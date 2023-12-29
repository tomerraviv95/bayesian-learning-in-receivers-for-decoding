from enum import Enum

import torch

from python_code import DEVICE

HALF = 0.5
CLIPPING_VAL = 15
LOGITS_INIT = torch.exp(torch.Tensor([0.85])).to(DEVICE)


class Phase(Enum):
    TRAIN = 'train'
    TEST = 'test'


class ChannelModes(Enum):
    MIMO = 'MIMO'


class ChannelModels(Enum):
    Synthetic = 'Synthetic'
    Cost2100 = 'Cost2100'


class DetectorType(Enum):
    seq_model = 'seq_model'
    end_to_end = 'end_to_end'
    bayesian = 'bayesian'
    model_based_bayesian = 'model_based_bayesian'


class DecoderType(Enum):
    bp = 'bp'
    wbp = 'wbp'
    modular_bayesian_wbp = 'modular_bayesian_wbp'
    bayesian_wbp = 'bayesian_wbp'


class ModulationType(Enum):
    BPSK = 'BPSK'
    QPSK = 'QPSK'
    EightPSK = 'EightPSK'
