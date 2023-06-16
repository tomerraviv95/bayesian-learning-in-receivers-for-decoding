from typing import Tuple

import torch

from python_code import conf
from python_code.channel.communication_blocks.modulator import MODULATION_NUM_MAPPING
from python_code.detectors.detector_trainer import Detector
from python_code.detectors.dnn.dnn_detector import DNNDetector
from python_code.utils.constants import Phase, ModulationType
from python_code.utils.probs_utils import calculate_mimo_states, get_bits_from_qpsk_symbols, \
    get_bits_from_eightpsk_symbols, calculate_symbols_from_states

EPOCHS = 400


class DNNTrainer(Detector):
    """Form the evaluater class.

    Keyword arguments:

    """

    def __init__(self):
        self.n_user = conf.n_user
        self.n_ant = conf.n_ant
        self.lr = 5e-3
        super().__init__()

    def __str__(self):
        return 'DNN Detector'

    def _initialize_detector(self):
        """
            Loads the DNN detector
        """
        self.detector = DNNDetector(self.n_user, self.n_ant)

    def calc_loss(self, est: torch.Tensor, tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        :param est: [1,transmission_length,n_states], each element is a probability
        :param tx: [1, transmission_length]
        :return: loss value
        """
        gt_states = calculate_mimo_states(self.n_ant, tx)
        loss = self.criterion(input=est, target=gt_states)
        return loss

    def forward(self, rx: torch.Tensor, probs_vec: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if conf.modulation_type == ModulationType.BPSK.name:
            rx = rx.float()
        elif conf.modulation_type in [ModulationType.QPSK.name, ModulationType.EightPSK.name]:
            rx = torch.view_as_real(rx).float().reshape(rx.shape[0], -1)

        soft_estimation = self.detector(rx, phase=Phase.TEST)
        estimated_states = torch.argmax(soft_estimation, dim=1)
        detected_words = calculate_symbols_from_states(self.n_ant, estimated_states).long()
        if conf.modulation_type == ModulationType.QPSK.name:
            detected_words = get_bits_from_qpsk_symbols(detected_words)
        if conf.modulation_type == ModulationType.EightPSK.name:
            detected_words = get_bits_from_eightpsk_symbols(detected_words)
        soft_confidences = torch.amax(torch.softmax(soft_estimation, dim=1), dim=1).unsqueeze(-1).repeat(
            [1, self.n_ant])
        if conf.modulation_type in [ModulationType.QPSK.name, ModulationType.EightPSK.name]:
            soft_confidences = torch.repeat_interleave(soft_confidences,
                                                       MODULATION_NUM_MAPPING[conf.modulation_type] // 2,
                                                       dim=0)
        return detected_words, soft_confidences

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Online training module - trains on the detected word.
        Start from the previous weights, or from scratch.
        :param tx: transmitted word
        :param rx: received word
        """
        if self.train_from_scratch:
            self._initialize_detector()
        self.deep_learning_setup(self.lr)

        if conf.modulation_type in [ModulationType.QPSK.name, ModulationType.EightPSK.name]:
            rx = torch.view_as_real(rx).float().reshape(rx.shape[0], -1)

        # run training loops
        loss = 0
        for i in range(EPOCHS):
            # pass through detector
            soft_estimation = self.detector(rx.float(), phase=Phase.TRAIN)
            current_loss = self.run_train_loop(est=soft_estimation, tx=tx)
            loss += current_loss
