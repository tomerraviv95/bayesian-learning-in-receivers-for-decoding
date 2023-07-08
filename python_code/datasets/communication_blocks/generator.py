import numpy as np
from numpy.random import default_rng

from python_code import conf


class Generator:
    def __init__(self):
        self._bits_generator = default_rng(seed=conf.seed)

    def generate(self):
        pilots = self._bits_generator.integers(0, 2, size=(conf.pilots_length, conf.n_user))
        data = self._bits_generator.integers(0, 2, size=(conf.block_length - conf.pilots_length, conf.n_user))
        mx = np.concatenate([pilots, data])
        return mx
