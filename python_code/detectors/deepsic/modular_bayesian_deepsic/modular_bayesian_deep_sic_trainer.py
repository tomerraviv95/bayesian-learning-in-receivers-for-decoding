## Implements the LBD method "Learnable Bernoulli Dropout for Bayesian Deep Learning"
from typing import List

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.detectors.deepsic.deepsic_trainer import DeepSICTrainer, NITERATIONS, EPOCHS
from python_code.detectors.deepsic.modular_bayesian_deepsic.bayesian_deep_sic_detector import LossVariable, \
    BayesianDeepSICDetector
from python_code.utils.constants import HALF, Phase


class ModularBayesianDeepSICTrainer(DeepSICTrainer):
    """
    Our Proposed Approach, Each DeepSIC is applied with the Bayesian Approximation Individually
    """

    def __init__(self):
        self.ensemble_num = 3
        self.kl_scale = 5
        self.kl_beta = 1e-4
        self.arm_beta = 1
        super().__init__()

    def __str__(self):
        return 'MB-DeepSIC'

    def _initialize_detector(self):
        self.detector = [
            [BayesianDeepSICDetector(self.ensemble_num, self.kl_scale).to(DEVICE) for _ in range(NITERATIONS)] for _ in
            range(self.n_user)]  # 2D list for Storing the DeepSIC Networks

    def calc_loss(self, est: LossVariable, tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        loss = self.criterion(input=est.priors, target=tx.long())
        # ARM Loss
        arm_loss = 0
        for i in range(self.ensemble_num):
            loss_term_arm_original = self.criterion(input=est.arm_original[i], target=tx.long())
            loss_term_arm_tilde = self.criterion(input=est.arm_tilde[i], target=tx.long())
            arm_delta = (loss_term_arm_tilde - loss_term_arm_original)
            grad_logit = arm_delta * (est.u_list[i] - HALF)
            arm_loss += torch.matmul(grad_logit, est.dropout_logit.T)
        arm_loss = self.arm_beta *torch.mean(arm_loss)
        # KL Loss
        kl_term = self.kl_beta * est.kl_term
        loss += arm_loss + kl_term
        return loss

    def train_model(self, single_model: nn.Module, tx: torch.Tensor, rx: torch.Tensor):
        """
        Trains a DeepSIC Network
        """
        self.optimizer = torch.optim.Adam(single_model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        single_model = single_model.to(DEVICE)
        y_total = self.preprocess(rx)
        for _ in range(EPOCHS):
            soft_estimation = single_model(y_total, phase=Phase.TRAIN)
            self.run_train_loop(soft_estimation, tx)

    def train_models(self, model: List[List[BayesianDeepSICDetector]], i: int, tx_all: List[torch.Tensor],
                     rx_all: List[torch.Tensor]):
        for user in range(self.n_user):
            self.train_model(model[user][i], tx_all[user], rx_all[user])

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Main training function for DeepSIC evaluater. Initializes the probabilities, then propagates them through the
        network, training sequentially each network and not by end-to-end manner (each one individually).
        """
        if not conf.fading_in_channel:
            self._initialize_detector()
        # Initializing the probabilities
        probs_vec = self._initialize_probs_for_training(tx)
        # Training the DeepSICNet for each user-symbol/iteration
        for i in range(NITERATIONS):
            # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
            tx_all, rx_all = self.prepare_data_for_training(tx, rx, probs_vec)
            # Training the DeepSIC networks for the iteration>1
            self.train_models(self.detector, i, tx_all, rx_all)
            # Generating soft symbols for training purposes
            probs_vec = self.calculate_posteriors(self.detector, i + 1, probs_vec, rx)

