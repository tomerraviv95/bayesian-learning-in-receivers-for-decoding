import numpy as np

from dir_definitions import ECC_MATRICES_DIR
from python_code import conf
from python_code.utils.coding_utils import get_code_pcm_and_gm


class Encoder:
    def __init__(self):
        self.code_pcm, self.code_gm = get_code_pcm_and_gm(conf.code_bits, conf.message_bits,
                                                            ECC_MATRICES_DIR, conf.code_type)
        self.encoding = lambda u: (np.dot(u, self.code_gm) % 2)

    def encode(self, mx):
        reshaped_mx = mx.reshape(-1, conf.message_bits)
        tx = (np.dot(reshaped_mx, self.code_gm) % 2)
        reshaped_tx = tx.reshape(-1)
        return reshaped_tx
