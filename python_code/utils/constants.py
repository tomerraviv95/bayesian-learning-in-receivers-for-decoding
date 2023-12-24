from enum import Enum

HALF = 0.5
CLIPPING_VAL = 15
LOGITS_INIT = 1

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
    cmodel_based_bayesian = 'cmodel_based_bayesian'
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
