from typing import Tuple

import torch

from python_code import conf
from python_code.detectors.detector_trainer import Detector
from python_code.detectors.dnn.dnn_detector import DNNDetector
from python_code.utils.constants import ModulationType

EPOCHS = 500


class DNNTrainer(Detector):
    """Form the trainer class.

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
        self.detector = DNNDetector(conf.n_user, conf.n_ant)

    def calc_loss(self, est: torch.Tensor, tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        return self.criterion(input=est, target=tx)

    def forward(self, rx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if conf.modulation_type == ModulationType.BPSK.name:
                rx = rx.float()
            elif conf.modulation_type in [ModulationType.QPSK.name, ModulationType.EightPSK.name]:
                rx = torch.view_as_real(rx).float().reshape(rx.shape[0], -1)
            soft_estimation = self.detector(rx)
            probs_vec = torch.softmax(soft_estimation, dim=2)[:, :, 1:]
            detected_words, soft_confidences = self.compute_output(probs_vec)
        return detected_words, soft_confidences

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Online training module - trains on the detected word.
        Start from the previous weights, or from scratch.
        :param tx: transmitted word
        :param rx: received word
        """
        if not conf.fading_in_channel:
            self._initialize_detector()
        self.deep_learning_setup(self.lr)

        if conf.modulation_type in [ModulationType.QPSK.name, ModulationType.EightPSK.name]:
            rx = torch.view_as_real(rx).float().reshape(rx.shape[0], -1)

        # run training loops
        loss = 0
        for i in range(EPOCHS):
            # pass through detector
            soft_estimation = self.detector(rx.float())
            current_loss = self.run_train_loop(est=soft_estimation.reshape(-1,4), tx=tx.long().reshape(-1))
            loss += current_loss