## Implement the LBD method "Learnable Bernoulli Dropout for Bayesian Deep Learning"
import collections
from typing import List

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.datasets.communication_blocks.modulator import MODULATION_NUM_MAPPING
from python_code.detectors.deepsic.bayesian_deepsic.masked_deep_sic_detector import LossVariable, \
    MaskedDeepSICDetector
from python_code.detectors.deepsic.deepsic_trainer import DeepSICTrainer, NITERATIONS, EPOCHS
from python_code.utils.constants import HALF, Phase, ModulationType


class BayesianDeepSICTrainer(DeepSICTrainer):
    """
    The Black-Box Bayesian Approach Applied to DeepSIC
    """

    def __init__(self):
        self.ensemble_num = 3
        self.kl_scale = 5
        self.kl_beta = 1e-2
        self.arm_beta = 5
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
        self.dropout_logits = [torch.rand([1, self.hidden_size], requires_grad=True, device=DEVICE)
                               for _ in range(self.n_user * NITERATIONS)]

    def calc_loss(self, est: List[List[LossVariable]], tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        loss = 0
        for ind_ensemble in range(self.ensemble_num):
            cur_est = est[ind_ensemble]
            for user in range(self.n_user):
                for iter in range(NITERATIONS):
                    cur_loss_var = cur_est[user][iter]
                    cur_tx = tx[user]
                    # point loss
                    loss += self.criterion(input=cur_loss_var.priors, target=cur_tx.long()) / self.ensemble_num
                    # ARM Loss
                    loss_term_arm_original = self.criterion(input=cur_loss_var.arm_original, target=cur_tx.long())
                    loss_term_arm_tilde = self.criterion(input=cur_loss_var.arm_tilde, target=cur_tx.long())
                    arm_delta = (loss_term_arm_tilde - loss_term_arm_original)
                    grad_logit = arm_delta * (cur_loss_var.u_list - HALF)
                    arm_loss = torch.matmul(grad_logit, cur_loss_var.dropout_logit.T)
                    arm_loss = self.arm_beta * torch.mean(arm_loss)
                    # KL Loss
                    kl_term = self.kl_beta * cur_loss_var.kl_term
                    loss += arm_loss + kl_term
        return loss

    def infer_model(self, single_model: nn.Module, dropout_logit: nn.Parameter, rx: torch.Tensor):
        """
        Trains a DeepSIC Network
        """
        y_total = self.preprocess(rx)
        return single_model(y_total, dropout_logit, phase=Phase.TRAIN)

    def infer_models(self, rx_all: List[torch.Tensor]):
        loss_vars = [[] for _ in range(self.n_user)]
        for user in range(self.n_user):
            for iter in range(NITERATIONS):
                loss_var = self.infer_model(self.detector[user * NITERATIONS + iter],
                                            self.dropout_logits[user * NITERATIONS + iter],
                                            rx_all[user])
                loss_vars[user].append(loss_var)
        return loss_vars

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        if self.train_from_scratch:
            self._initialize_detector()
        params = list(self.detector.parameters())  # + self.dropout_logits
        self.optimizer = torch.optim.Adam(params, lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        for _ in range(EPOCHS):
            total_loss_vars = {}
            for ind_ensemble in range(self.ensemble_num):
                # Initializing the probabilities
                probs_vec = self._initialize_probs_for_training(tx)
                # Training the DeepSICNet for each user-symbol/iteration
                for i in range(NITERATIONS):
                    # Generating soft symbols for training purposes
                    probs_vec = self.calculate_posteriors(self.detector, i, probs_vec, rx)
                # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
                tx_all, rx_all = self.prepare_data_for_training(tx, rx, probs_vec)
                total_loss_vars[ind_ensemble] = self.infer_models(rx_all)
            loss = self.calc_loss(total_loss_vars, tx_all)
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
