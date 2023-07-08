from enum import Enum

HALF = 0.5
CLIPPING_VAL = 10

class Phase(Enum):
    TRAIN = 'train'
    TEST = 'test'


class ChannelModes(Enum):
    MIMO = 'MIMO'


class ChannelModels(Enum):
    Synthetic = 'Synthetic'


class DetectorType(Enum):
    seq_model = 'seq_model'
    model_based_bayesian = 'model_based_bayesian'
    bayesian = 'bayesian'
    black_box = 'black_box'
    bayesian_black_box = 'bayesian_black_box'


class DecoderType(Enum):
    bp_model = 'bp_model'


class ModulationType(Enum):
    BPSK = 'BPSK'
    QPSK = 'QPSK'
    EightPSK = 'EightPSK'
