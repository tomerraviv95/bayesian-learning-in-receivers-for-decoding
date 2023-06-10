import random
from typing import Tuple

import numpy as np
import torch
from torch import nn

from dir_definitions import ECC_MATRICES_DIR
from python_code import conf
from python_code.utils.constants import TANNER_GRAPH_CYCLE_REDUCTION, Phase
from python_code.utils.python_utils import load_code_parameters

random.seed(conf.seed)
torch.manual_seed(conf.seed)
torch.cuda.manual_seed(conf.seed)
np.random.seed(conf.seed)


class Decoder(nn.Module):
    """
    Implements the meta-trainer class. Every trainer must inherent from this base class.
    It implements the evaluation method, initializes the dataloader and the detector.
    It also defines some functions that every inherited trainer must implement.
    """

    def __init__(self):
        super(Decoder, self).__init__()
        # initialize matrices, datasets and detector
        self.odd_llr_mask_only = True
        self.even_mask_only = True
        self.multiloss_output_mask_only = True
        self.output_mask_only = False
        self.multi_loss_flag = True
        self.iteration_num = conf.iterations
        self._code_bits = conf.code_bits
        self._info_bits = conf.info_bits
        self.code_pcm, self.code_gm = load_code_parameters(self._code_bits, self._info_bits,
                                                           ECC_MATRICES_DIR, TANNER_GRAPH_CYCLE_REDUCTION)
        self.neurons = int(np.sum(self.code_pcm))

    def get_name(self):
        return self.__name__()

    def forward(self, rx: torch.Tensor, phase: Phase) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Every trainer must have some forward pass for its detector
        """
        pass

    # tx, rx = self.val_channel_dataset.__getitem__(snr_list=[conf.val_snr])
    # # detect data part after training on the pilot part
    # output_list = self.forward(rx, phase=Phase.VAL)
    # decoded_words = torch.round(torch.sigmoid(-output_list[-1]))
    # # calculate accuracy
    # ber, errors = calculate_ber(decoded_words, tx)
