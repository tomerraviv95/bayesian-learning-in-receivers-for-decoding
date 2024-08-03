## Implements the LBD method "Learnable Bernoulli Dropout for Bayesian Deep Learning"
from typing import List

import torch
from torch import nn

from python_code import DEVICE
from python_code.detectors.deepsic_trainer import DeepSICTrainer, EPOCHS
from python_code.detectors.modular_bayesian_deepsic.bayesian_deep_sic_detector import LossVariable, \
    BayesianDeepSICDetector
from python_code.utils.constants import HALF, Phase


class ModularBayesianDeepSICTrainer(DeepSICTrainer):
    """
    Our Proposed Approach, Each DeepSIC is applied with the Bayesian Approximation Individually
    """

    def __init__(self):
        self.ensemble_num = 3
        self.kl_scale = 1
        self.kl_beta = 1e-4
        self.arm_beta = 1
        super().__init__()

    def __str__(self):
        return 'MB-DeepSIC'

    def _initialize_detector(self):
        self.detector = [
            [BayesianDeepSICDetector(self.ensemble_num, self.kl_scale).to(DEVICE) for _ in range(self.iterations)] for _
            in
            range(self.n_user)]  # 2D list for Storing the DeepSIC Networks

    def calc_loss(self, est: LossVariable, tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        loss = self.criterion(input=est.priors, target=tx.long())
        # ARM Loss
        loss_term_arm_original = self.criterion_arm(input=est.arm_original[0], target=tx.long())
        loss_term_arm_tilde = self.criterion_arm(input=est.arm_tilde[0], target=tx.long())
        arm_delta = (loss_term_arm_tilde - loss_term_arm_original)
        grad_logit = arm_delta.unsqueeze(-1) * (est.u - HALF)
        arm_loss = grad_logit * est.dropout_logits
        arm_loss = self.arm_beta * torch.mean(arm_loss)
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
        self.criterion_arm = torch.nn.CrossEntropyLoss(reduction='none')
        single_model = single_model.to(DEVICE)
        y_total = self.preprocess(rx)
        for _ in range(EPOCHS):
            est = single_model(y_total, phase=Phase.TRAIN)
            self.run_train_loop(est, tx)

    def train_models(self, model: List[List[BayesianDeepSICDetector]], i: int, tx_all: List[torch.Tensor],
                     rx_all: List[torch.Tensor]):
        for user in range(self.n_user):
            self.train_model(model[user][i], tx_all[user], rx_all[user])
