import random
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import RMSprop, Adam, SGD

from python_code import DEVICE, conf

random.seed(conf.seed)
torch.manual_seed(conf.seed)
torch.cuda.manual_seed(conf.seed)
np.random.seed(conf.seed)


class Detector(nn.Module):
    """
    Implements the meta-evaluater class. Every evaluater must inherent from this base class.
    It implements the evaluation method, initializes the dataloader and the detector.
    It also defines some functions that every inherited evaluater must implement.
    """

    def __init__(self):
        super().__init__()
        self.train_from_scratch = not conf.fading_in_channel
        self._initialize_detector()
        self.softmax = torch.nn.Softmax(dim=1)

    def get_name(self):
        return self.__name__()

    def _initialize_detector(self):
        """
        Every evaluater must have some base detector
        """
        self.detector = None

    # calculate train loss
    def calc_loss(self, est: torch.Tensor, tx: torch.Tensor) -> torch.Tensor:
        """
         Every evaluater must have some loss calculation
        """
        pass

    # setup the optimization algorithm
    def deep_learning_setup(self, lr: float):
        """
        Sets up the optimizer and loss criterion
        """
        if conf.optimizer_type == 'Adam':
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.detector.parameters()),
                                  lr=lr)
        elif conf.optimizer_type == 'RMSprop':
            self.optimizer = RMSprop(filter(lambda p: p.requires_grad, self.detector.parameters()),
                                     lr=lr)
        elif conf.optimizer_type == 'SGD':
            self.optimizer = SGD(filter(lambda p: p.requires_grad, self.detector.parameters()),
                                 lr=lr)
        else:
            raise NotImplementedError("No such optimizer implemented!!!")
        if conf.loss_type == 'CrossEntropy':
            self.criterion = CrossEntropyLoss().to(DEVICE)
        elif conf.loss_type == 'MSE':
            self.criterion = MSELoss().to(DEVICE)
        else:
            raise NotImplementedError("No such loss function implemented!!!")

    # setup the optimization algorithm
    def calibration_deep_learning_setup(self):
        """
        Sets up the optimizer and loss criterion
        """
        if conf.optimizer_type == 'Adam':
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.detector.net.dropout_logits),
                                  lr=self.lr)
        elif conf.optimizer_type == 'RMSprop':
            self.optimizer = RMSprop(filter(lambda p: p.requires_grad, self.detector.net.dropout_logits),
                                     lr=self.lr)
        elif conf.optimizer_type == 'SGD':
            self.optimizer = SGD(filter(lambda p: p.requires_grad, self.detector.net.dropout_logits),
                                 lr=self.lr)
        else:
            raise NotImplementedError("No such optimizer implemented!!!")
        if conf.loss_type == 'CrossEntropy':
            self.criterion = CrossEntropyLoss().to(DEVICE)
        elif conf.loss_type == 'MSE':
            self.criterion = MSELoss().to(DEVICE)
        else:
            raise NotImplementedError("No such loss function implemented!!!")

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Every detector evaluater must have some function to adapt it online
        """
        pass

    def forward(self, rx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Every evaluater must have some forward pass for its detector
        """
        pass

    def run_train_loop(self, est: torch.Tensor, tx: torch.Tensor) -> float:
        # calculate loss
        loss = self.calc_loss(est=est, tx=tx)
        current_loss = loss.item()
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return current_loss
