## Implement the LBD method "Learnable Bernoulli Dropout for Bayesian Deep Learning"

from itertools import chain

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.datasets.communication_blocks.modulator import MODULATION_NUM_MAPPING
from python_code.detectors.bayesian_deepsic.masked_deep_sic_detector import MaskedDeepSICDetector
from python_code.detectors.deepsic_trainer import DeepSICTrainer, EPOCHS
from python_code.utils.constants import Phase, ModulationType, HALF, LOGITS_INIT


class BayesianDeepSICTrainer(DeepSICTrainer):
    """
    The Black-Box Bayesian Approach Applied to DeepSIC
    """

    def __init__(self):
        self.ensemble_num = 3
        self.kl_scale = 1
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
        flat_detectors_list = [
            MaskedDeepSICDetector(self.linear_input, self.hidden_size, self.classes_num, self.kl_scale).to(DEVICE) for _
            in range(self.iterations * self.n_user)]
        self.detector = nn.ModuleList(flat_detectors_list)
        self.dropout_logits = [nn.Parameter(LOGITS_INIT * torch.ones(self.hidden_size).reshape(1, -1).to(DEVICE))
                               for _ in range(self.iterations * self.n_user)]

    def infer_model(self, single_model: nn.Module, dropout_logit: nn.Parameter, rx: torch.Tensor):
        """
        Trains a DeepSIC Network
        """
        y_total = self.preprocess(rx)
        return single_model(y_total, dropout_logit, phase=Phase.TRAIN)

    def calc_loss(self, all_loss_vars, tx, rx, probs_vec):
        loss = 0
        f_loss, arm_loss, kl_term = 0, 0, 0
        # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
        tx_all, rx_all = self.prepare_data_for_training(tx, rx, probs_vec)
        # calculate the loss based on the outputs of the network. Note that we cannot use the middle layers' outputs
        # for these calculations when assuming end-to-end training, similar to end-to-end training of DeepSIC. See
        # the original DeepSIC paper for more details.
        for iter in range(self.iterations):
            for user in range(self.n_user):
                loss_var = all_loss_vars[iter][user]
                # ARM Loss
                loss_term_arm_original = self.criterion(input=loss_var.arm_original, target=tx_all[user].long())
                loss_term_arm_tilde = self.criterion(input=loss_var.arm_tilde, target=tx_all[user].long())
                arm_delta = (loss_term_arm_tilde - loss_term_arm_original)
                grad_logit = arm_delta.unsqueeze(-1) * (loss_var.u - HALF)
                arm_loss_before_avg = grad_logit * loss_var.dropout_logits
                arm_loss += self.arm_beta * torch.mean(arm_loss_before_avg)
                kl_term += self.kl_beta * loss_var.kl_term
                # Frequentist loss
                f_loss += self.criterion(input=loss_var.priors, target=tx_all[user].long())
        loss += f_loss + arm_loss + kl_term
        return loss

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        if self.train_from_scratch:
            self._initialize_detector()
        total_parameters = self.detector.parameters()
        total_parameters = chain(total_parameters, self.dropout_logits)
        self.optimizer = torch.optim.Adam(total_parameters, lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        for _ in range(EPOCHS):
            # Initializing the probabilities
            probs_vec = self._initialize_probs_for_training(tx)
            all_loss_vars = []
            # Training the DeepSICNet for each user-symbol/iteration
            for i in range(self.iterations):
                # Generating soft symbols for training purposes
                probs_vec, loss_vars = self.calculate_posteriors(self.detector, i + 1, probs_vec, rx, phase=Phase.TRAIN)
                all_loss_vars.append(loss_vars)
            # adding the loss. In contrast to sequential learning - we do not update yet
            loss = self.calc_loss(all_loss_vars, tx.int(), rx, probs_vec)
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
                for i in range(self.iterations):
                    probs_vec = self.calculate_posteriors(self.detector, i + 1, probs_vec, rx)
                total_probs_vec += probs_vec
            total_probs_vec /= self.ensemble_num
        detected_words, soft_confidences = self.compute_output(total_probs_vec)
        return detected_words, soft_confidences

    def calculate_posteriors(self, model: nn.ModuleList, i: int, probs_vec: torch.Tensor,
                             rx: torch.Tensor, phase=Phase.TEST) -> torch.Tensor:
        """
        Propagates the probabilities through the learnt networks.
        """
        next_probs_vec = torch.zeros(probs_vec.shape).to(DEVICE)
        loss_vars = []
        for user in range(self.n_user):
            idx = [user_i for user_i in range(self.n_user) if user_i != user]
            input = torch.cat((rx, probs_vec[:, idx].reshape(rx.shape[0], -1)), dim=1)
            preprocessed_input = self.preprocess(input)
            current_dropout_logits = self.dropout_logits[user * self.iterations + i - 1]
            if phase == phase.TRAIN:
                loss_variable = model[user * self.iterations + i - 1](preprocessed_input, current_dropout_logits,
                                                                      phase=phase)
                loss_vars.append(loss_variable)
                probs = loss_variable.priors
            else:
                probs = model[user * self.iterations + i - 1](preprocessed_input, current_dropout_logits, phase=phase)
            output = self.softmax(probs)
            next_probs_vec[:, user] = output[:, 1:].reshape(next_probs_vec[:, user].shape)
        if phase == phase.TRAIN:
            return next_probs_vec, loss_vars
        return next_probs_vec
