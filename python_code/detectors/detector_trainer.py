import random
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from python_code import DEVICE, conf
from python_code.datasets.communication_blocks.modulator import MODULATION_DICT
from python_code.utils.constants import ModulationType
from python_code.utils.probs_utils import prob_to_BPSK_symbol, prob_to_QPSK_symbol, prob_to_EightPSK_symbol

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
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.detector.parameters()),
                              lr=lr)
        self.criterion = CrossEntropyLoss().to(DEVICE)

    # setup the optimization algorithm
    def calibration_deep_learning_setup(self):
        """
        Sets up the optimizer and loss criterion
        """
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.detector.net.dropout_logits),
                              lr=self.lr)
        self.criterion = CrossEntropyLoss().to(DEVICE)

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

    def compute_output(self, probs_vec):
        if conf.modulation_type == ModulationType.BPSK.name:
            symbols_word = prob_to_BPSK_symbol(probs_vec.float())
        elif conf.modulation_type == ModulationType.QPSK.name:
            symbols_word = prob_to_QPSK_symbol(probs_vec.float())
        elif conf.modulation_type == ModulationType.EightPSK.name:
            symbols_word = prob_to_EightPSK_symbol(probs_vec.float())
        else:
            raise ValueError("No such constellation!")
        detected_words = MODULATION_DICT[conf.modulation_type].demodulate(symbols_word)
        if conf.modulation_type == ModulationType.BPSK.name:
            new_probs_vec = torch.cat([probs_vec.unsqueeze(dim=2), (1 - probs_vec).unsqueeze(dim=2)], dim=2)
        elif conf.modulation_type in [ModulationType.QPSK.name, ModulationType.EightPSK.name]:
            new_probs_vec = torch.cat([probs_vec, (1 - probs_vec.sum(dim=2)).unsqueeze(dim=2)], dim=2)
        else:
            raise ValueError("No such constellation!")
        soft_confidences = torch.amax(new_probs_vec, dim=2)
        return detected_words, soft_confidences
