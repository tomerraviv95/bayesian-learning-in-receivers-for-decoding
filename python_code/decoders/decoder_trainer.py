import os

import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from dir_definitions import ECC_MATRICES_DIR, BP_WEIGHTS
from python_code import conf, DEVICE
from python_code.utils.coding_utils import get_code_pcm_and_gm

ITERATIONS = 5
EPOCHS = 400
LR = 1e-3


class DecoderTrainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)  # Single symbol probability inference
        self.odd_llr_mask_only = False
        self.even_mask_only = True
        self.multiloss_output_mask_only = True
        self.output_mask_only = True
        self.multi_loss_flag = True
        self.iteration_num = ITERATIONS
        self._code_bits = conf.code_bits
        self._message_bits = conf.message_bits
        self.code_pcm, self.code_gm = get_code_pcm_and_gm(conf.code_bits, conf.message_bits, ECC_MATRICES_DIR,
                                                          conf.code_type)
        self.neurons = int(np.sum(self.code_pcm))
        if not os.path.exists(BP_WEIGHTS):
            os.makedirs(BP_WEIGHTS)
        self.model_name = os.path.join(BP_WEIGHTS, str(self) + "_" + str(self._code_bits) + "_" + str(
            self._message_bits) + "_" + str(self.iteration_num))

    # setup the optimization algorithm
    def deep_learning_setup(self, lr: float):
        """
        Sets up the optimizer and loss criterion
        """
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr,
                              weight_decay=0.0005, betas=(0.5, 0.999))
        self.criterion = BCEWithLogitsLoss().to(DEVICE)

    def online_training(self, rx, s):
        self.initialize_layers()
        self.single_training(s, rx)
