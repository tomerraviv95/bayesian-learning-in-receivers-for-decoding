import concurrent.futures
from typing import List
from typing import Tuple

import numpy as np
import torch
from numpy.random import default_rng
from torch.utils.data import Dataset

from python_code import DEVICE
from python_code import conf
from python_code.datasets.communication_blocks.encoder import Encoder
from python_code.datasets.communication_blocks.modulator import BPSKModulator
from python_code.datasets.siso_channels.awgn_channel import AWGNChannel


class EccChannel:
    def __init__(self, codewords_num: int):
        self._codewords_num = codewords_num
        self._bits_generator = default_rng(seed=conf.seed)
        self.rate = float(conf.message_bits / conf.code_bits)
        self._encoder = Encoder()

    def _transmit(self, snr: float) -> Tuple[np.ndarray, np.ndarray]:
        mx = self._bits_generator.integers(0, 2, size=(self._codewords_num, conf.message_bits))
        tx = self._encoder.encode(mx)
        s = BPSKModulator.modulate(tx)
        rx = AWGNChannel(s=s, snr=snr, rate=self.rate, random=np.random.RandomState(conf.seed))
        return tx, rx

    def get_vectors(self, snr: float) -> Tuple[np.ndarray, np.ndarray]:
        # get datasets values
        tx, rx = self._transmit(snr)
        return tx, rx


class CodingDataset(Dataset):
    """
    Dataset object for the datasets. Used in training and evaluation.
    Returns (transmitted, received, channel_coefficients) batch.
    """

    def __init__(self, codewords_num: int):
        self.codewords_num = codewords_num
        self.channel_type = EccChannel(codewords_num)

    def get_snr_data(self, snr: float, database: list):
        if database is None:
            database = []
        tx, rx = self.channel_type.get_vectors(snr)
        database.append((tx, rx))

    def __getitem__(self, snr_list: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        database = []
        # do not change max_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            [executor.submit(self.get_snr_data, snr, database) for snr in snr_list]
        tx, rx = (np.concatenate(arrays) for arrays in zip(*database))
        tx, rx = torch.Tensor(tx).to(device=DEVICE), torch.from_numpy(rx).to(device=DEVICE).float()
        return tx, rx

    def __len__(self):
        return self.codewords_num
