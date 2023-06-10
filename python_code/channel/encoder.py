import numpy as np

from dir_definitions import ECC_MATRICES_DIR
from python_code import conf
from python_code.utils.constants import TANNER_GRAPH_CYCLE_REDUCTION
from python_code.utils.python_utils import load_code_parameters


class Encoder:
    def __init__(self):
        self.code_pcm, self.code_gm = load_code_parameters(conf.code_bits, conf.message_bits,
                                                           ECC_MATRICES_DIR, TANNER_GRAPH_CYCLE_REDUCTION)
        self.encoding = lambda u: (np.dot(u, self.code_gm) % 2)

    def encode(self, u):
        return np.dot(u, self.code_gm) % 2
