from typing import List

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.datasets.communication_blocks.modulator import MODULATION_DICT, MODULATION_NUM_MAPPING
from python_code.detectors.detector_trainer import Detector
from python_code.utils.constants import ModulationType, HALF
from python_code.utils.probs_utils import prob_to_EightPSK_symbol, prob_to_QPSK_symbol, prob_to_BPSK_symbol

NITERATIONS = 2
EPOCHS = 200

class DeepSICTrainer(Detector):

    def __init__(self):
        self.memory_length = 1
        self.n_user = conf.n_user
        self.n_ant = conf.n_ant
        self.lr = 5e-3
        super().__init__()

    def __str__(self):
        return 'DeepSIC'

    def _initialize_detector(self):
        pass

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        pass

    def calc_loss(self, est: torch.Tensor, tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        return self.criterion(input=est, target=tx.long())

    @staticmethod
    def preprocess(rx: torch.Tensor) -> torch.Tensor:
        if conf.modulation_type == ModulationType.BPSK.name:
            return rx.float()
        elif conf.modulation_type in [ModulationType.QPSK.name, ModulationType.EightPSK.name]:
            y_input = torch.view_as_real(rx[:, :conf.n_ant]).float().reshape(rx.shape[0], -1)
            return torch.cat([y_input, rx[:, conf.n_ant:].float()], dim=1)

    def forward(self, rx: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
        with torch.no_grad():
            # detect and decode
            probs_vec = self._initialize_probs_for_infer(rx)
            for i in range(NITERATIONS):
                probs_vec = self.calculate_posteriors(self.detector, i + 1, probs_vec, rx)
            detected_words, soft_confidences = self.compute_output(probs_vec)
            return detected_words, soft_confidences

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

    def prepare_data_for_training(self, tx: torch.Tensor, rx: torch.Tensor, probs_vec: torch.Tensor) -> [
        torch.Tensor, torch.Tensor]:
        """
        Generates the data for each user
        """
        tx_all = []
        rx_all = []
        for k in range(self.n_user):
            idx = [user_i for user_i in range(self.n_user) if user_i != k]
            current_y_train = torch.cat((rx, probs_vec[:, idx].reshape(rx.shape[0], -1)), dim=1)
            tx_all.append(tx[:, k])
            rx_all.append(current_y_train)
        return tx_all, rx_all

    def calculate_posteriors(self, model: List[List[nn.Module]], i: int, probs_vec: torch.Tensor,
                             rx: torch.Tensor) -> torch.Tensor:
        """
        Propagates the probabilities through the learnt networks.
        """
        next_probs_vec = torch.zeros(probs_vec.shape).to(DEVICE)
        for user in range(self.n_user):
            idx = [user_i for user_i in range(self.n_user) if user_i != user]
            input = torch.cat((rx, probs_vec[:, idx].reshape(rx.shape[0], -1)), dim=1)
            preprocessed_input = self.preprocess(input)
            with torch.no_grad():
                output = self.softmax(model[user][i - 1](preprocessed_input))
            next_probs_vec[:, user] = output[:, 1:].reshape(next_probs_vec[:, user].shape)
        return next_probs_vec

    def _initialize_probs_for_training(self, tx):
        if conf.modulation_type == ModulationType.BPSK.name:
            probs_vec = HALF * torch.ones(tx.shape).to(DEVICE)
        elif conf.modulation_type in [ModulationType.QPSK.name, ModulationType.EightPSK.name]:
            probs_vec = (1 / MODULATION_NUM_MAPPING[conf.modulation_type]) * torch.ones(tx.shape).to(DEVICE).unsqueeze(
                -1).repeat([1, 1, MODULATION_NUM_MAPPING[conf.modulation_type] - 1])
        else:
            raise ValueError("No such constellation!")
        return probs_vec

    def _initialize_probs_for_infer(self, rx):
        if conf.modulation_type == ModulationType.BPSK.name:
            probs_vec = HALF * torch.ones(rx.shape).to(DEVICE).float()
        elif conf.modulation_type in [ModulationType.QPSK.name, ModulationType.EightPSK.name]:
            probs_vec = (1 / MODULATION_NUM_MAPPING[conf.modulation_type]) * torch.ones(rx.shape).to(DEVICE).unsqueeze(
                -1)
            probs_vec = probs_vec.repeat([1, 1, MODULATION_NUM_MAPPING[conf.modulation_type] - 1]).float()
        else:
            raise ValueError("No such constellation!")
        return probs_vec
