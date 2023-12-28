## Implements the LBD method "Learnable Bernoulli Dropout for Bayesian Deep Learning"
from typing import List

import torch
from torch import nn

from python_code import DEVICE
from python_code.detectors.deepsic.cmodular_bayesian_deepsic.cbayesian_deep_sic_detector import LossVariable, \
    CBayesianDeepSICDetector
from python_code.detectors.deepsic.deepsic_trainer import DeepSICTrainer, NITERATIONS, EPOCHS
from python_code.utils.constants import Phase


class CModularBayesianDeepSICTrainer(DeepSICTrainer):
    """
    Our Proposed Approach, Each DeepSIC is applied with the Bayesian Approximation Individually
    """

    def __init__(self):
        self.ensemble_num = 5
        super().__init__()

    def __str__(self):
        return 'CMB-DeepSIC'

    def _initialize_detector(self):
        self.detector = [
            [CBayesianDeepSICDetector(self.ensemble_num).to(DEVICE) for _ in range(NITERATIONS)] for _ in
            range(self.n_user)]  # 2D list for Storing the DeepSIC Networks

    def calc_loss(self, est: LossVariable, tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        loss = self.criterion(input=est[0], target=tx.long()) + est[1]
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
            log_loss = single_model(y_total, phase=Phase.TRAIN)
            est = [log_loss, single_model.regularisation()]
            self.run_train_loop(est, tx)

    def train_models(self, model: List[List[CBayesianDeepSICDetector]], i: int, tx_all: List[torch.Tensor],
                     rx_all: List[torch.Tensor]):
        for user in range(self.n_user):
            self.train_model(model[user][i], tx_all[user], rx_all[user])
