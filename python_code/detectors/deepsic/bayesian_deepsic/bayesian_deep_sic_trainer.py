## Implement the LBD method "Learnable Bernoulli Dropout for Bayesian Deep Learning"
from typing import List

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.datasets.communication_blocks.modulator import MODULATION_NUM_MAPPING
from python_code.detectors.deepsic.bayesian_deepsic.masked_deep_sic_detector import MaskedDeepSICDetector
from python_code.detectors.deepsic.deepsic_trainer import DeepSICTrainer, NITERATIONS, EPOCHS
from python_code.utils.constants import Phase, ModulationType, HALF, LOGITS_INIT


class BayesianDeepSICTrainer(DeepSICTrainer):
    """
    The Black-Box Bayesian Approach Applied to DeepSIC
    """

    def __init__(self):
        self.ensemble_num = 5
        self.kl_scale = 5
        self.kl_beta = 1e-4
        self.arm_beta = 1
        self.classes_num = MODULATION_NUM_MAPPING[conf.modulation_type]
        self.hidden_size = conf.hidden_base_size * self.classes_num
        base_rx_size = conf.n_ant if conf.modulation_type == ModulationType.BPSK.name else 2 * conf.n_ant
        self.linear_input = base_rx_size + (self.classes_num - 1) * (conf.n_user - 1)  # from DeepSIC paper
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def __str__(self):
        return 'B-DeepSIC'

    def _initialize_detector(self):
        detectors_list = [
            [MaskedDeepSICDetector(self.linear_input, self.hidden_size, self.classes_num, self.kl_scale).to(DEVICE) for
             _ in range(NITERATIONS)]
            for _ in
            range(self.n_user)]  # 2D list for Storing the DeepSIC Networks
        flat_detectors_list = [detector for sublist in detectors_list for detector in sublist]
        self.detector = nn.ModuleList(flat_detectors_list)
        self.dropout_logits = [LOGITS_INIT * torch.rand([1, self.hidden_size], device=DEVICE)
                               for _ in range(self.n_user * NITERATIONS)]
        for dropout_logit in self.dropout_logits:
            dropout_logit.requires_grad = True

    def infer_model(self, single_model: nn.Module, dropout_logit: nn.Parameter, rx: torch.Tensor):
        """
        Trains a DeepSIC Network
        """
        y_total = self.preprocess(rx)
        return single_model(y_total, dropout_logit, phase=Phase.TRAIN)

    def calc_loss(self, tx_all: List[torch.Tensor], rx_all: List[torch.Tensor], iter: int,
                  only_bayesian_loss: bool):
        loss = 0
        for user in range(self.n_user):
            loss_var = self.infer_model(self.detector[user * NITERATIONS + iter],
                                        self.dropout_logits[user * NITERATIONS + iter],
                                        rx_all[user])
            # ARM Loss
            loss_term_arm_original = self.criterion(input=loss_var.arm_original, target=tx_all[user].long())
            loss_term_arm_tilde = self.criterion(input=loss_var.arm_tilde, target=tx_all[user].long())
            arm_delta = (loss_term_arm_tilde - loss_term_arm_original)
            grad_logit = arm_delta * (loss_var.u_list - HALF)
            arm_loss = torch.matmul(grad_logit, loss_var.dropout_logit.T)
            arm_loss = self.arm_beta * torch.mean(arm_loss)
            # KL Loss
            kl_term = self.kl_beta * loss_var.kl_term
            loss += arm_loss + kl_term
            # Frequentist loss
            if not only_bayesian_loss:
                fq_loss = self.criterion(input=loss_var.priors, target=tx_all[user].long())
                loss += fq_loss
        return loss

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        if self.train_from_scratch:
            self._initialize_detector()
        params = list(self.detector.parameters()) + self.dropout_logits
        self.optimizer = torch.optim.Adam(params, lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        for _ in range(EPOCHS):
            # Initializing the probabilities
            probs_vec = self._initialize_probs_for_training(tx)
            # Training the DeepSICNet for each user-symbol/iteration
            loss = 0
            for i in range(NITERATIONS):
                # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
                tx_all, rx_all = self.prepare_data_for_training(tx, rx, probs_vec)
                if i == NITERATIONS - 1:
                    break
                # Compute the ARM and KL losses
                loss += self.calc_loss(tx_all, rx_all, i, only_bayesian_loss=True)
                # Generating soft symbols for training purposes
                probs_vec = self.calculate_posteriors(self.detector, i + 1, probs_vec, rx)
            # adding the loss. In contrast to sequential learning - we do not update yet
            loss += self.calc_loss(tx_all, rx_all, NITERATIONS - 1, only_bayesian_loss=False)
            # back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def forward(self, rx: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
        with torch.no_grad():
            # detect and decode
            total_probs_vec = 0
            for ind_ensemble in range(self.ensemble_num):
                # detect and decode
                probs_vec = self._initialize_probs_for_infer(rx)
                for i in range(NITERATIONS):
                    probs_vec = self.calculate_posteriors(self.detector, i + 1, probs_vec, rx)
                total_probs_vec += probs_vec
            total_probs_vec /= self.ensemble_num
        detected_words, soft_confidences = self.compute_output(total_probs_vec)
        return detected_words, soft_confidences

    def calculate_posteriors(self, model: nn.ModuleList, i: int, probs_vec: torch.Tensor,
                             rx: torch.Tensor) -> torch.Tensor:
        """
        Propagates the probabilities through the learnt networks.
        """
        next_probs_vec = torch.zeros(probs_vec.shape).to(DEVICE)
        for user in range(self.n_user):
            idx = [user_i for user_i in range(self.n_user) if user_i != user]
            input = torch.cat((rx, probs_vec[:, idx].reshape(rx.shape[0], -1)), dim=1)
            preprocessed_input = self.preprocess(input)
            output = self.softmax(model[user * NITERATIONS + i - 1](preprocessed_input, self.dropout_logits[
                user * NITERATIONS + i - 1]))
            next_probs_vec[:, user] = output[:, 1:].reshape(next_probs_vec[:, user].shape)
        return next_probs_vec
