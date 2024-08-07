from typing import List

import torch
from torch import nn

from python_code import DEVICE
from python_code.detectors.deepsic_detector import DeepSICDetector
from python_code.detectors.deepsic_trainer import DeepSICTrainer, EPOCHS


class EndToEndDeepSICTrainer(DeepSICTrainer):
    def __init__(self):
        super().__init__()
        self.apply_dropout = False

    def __str__(self):
        return 'F-DeepSIC'

    def _initialize_detector(self):
        detectors_list = [[DeepSICDetector().to(DEVICE) for _ in range(self.iterations)] for _ in
                          range(self.n_user)]  # 2D list for Storing the DeepSIC Networks
        flat_detectors_list = [detector for sublist in detectors_list for detector in sublist]
        self.detector = nn.ModuleList(flat_detectors_list)

    def train_model(self, single_model: nn.Module, tx: torch.Tensor, rx: torch.Tensor):
        """
        Trains a DeepSIC Network
        """
        y_total = self.preprocess(rx)
        soft_estimation = single_model(y_total, apply_dropout=self.apply_dropout)
        loss = self.calc_loss(est=soft_estimation, tx=tx.int())
        return loss

    def train_models(self, tx_all: List[torch.Tensor], rx_all: List[torch.Tensor], iter: int):
        cur_loss = 0
        for user in range(self.n_user):
            cur_loss += self.train_model(self.detector[user * self.iterations + iter], tx_all[user], rx_all[user])
        return cur_loss

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Main training function for DeepSIC trainer. Initializes the probabilities, then propagates them through the
        network, training the entire networks by end-to-end manner.
        """
        if self.train_from_scratch:
            self._initialize_detector()
        self.optimizer = torch.optim.Adam(self.detector.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(EPOCHS):
            # Initializing the probabilities
            probs_vec = self._initialize_probs_for_training(tx)
            # Training the DeepSICNet for each user-symbol/iteration
            loss = 0
            for i in range(self.iterations):
                # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
                tx_all, rx_all = self.prepare_data_for_training(tx, rx, probs_vec)
                # Generating soft symbols for training purposes
                probs_vec = self.calculate_posteriors(self.detector, i + 1, probs_vec, rx)
            # adding the loss. In contrast to sequential learning - we do not update yet
            loss += self.train_models(tx_all, rx_all, self.iterations - 1)
            # back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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
            output = self.softmax(model[user * self.iterations + i - 1](preprocessed_input))
            next_probs_vec[:, user] = output[:, 1:].reshape(next_probs_vec[:, user].shape)
        return next_probs_vec
