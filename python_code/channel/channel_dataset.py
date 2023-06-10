import concurrent.futures
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset

from python_code import DEVICE, conf
from python_code.channel.encoder import Encoder
from python_code.channel.generator import Generator
from python_code.channel.modulator import MODULATION_DICT
from python_code.channel.transmitter import Transmitter
from python_code.utils.constants import ModulationType
from python_code.utils.python_utils import normalize_for_modulation


class ChannelModelDataset(Dataset):
    """
    Dataset object for the channel. Used in training and evaluation.
    Returns (transmitted, received, channel_coefficients) batch.
    """

    def __init__(self):
        self.blocks_num = conf.block_num
        self.generator = Generator()
        self.encoder = Encoder()
        self.modulator = MODULATION_DICT[conf.modulation_type]
        self.transmitter = Transmitter()

        # if conf.modulation_type == ModulationType.QPSK.name:
        #     tx = get_qpsk_symbols_from_bits(s)
        # if conf.modulation_type == ModulationType.EightPSK.name:
        #     tx = get_eightpsk_symbols_from_bits(s)

    def get_snr_data(self, snr: float, database: list):
        if database is None:
            database = []
        mx_full = np.empty((self.blocks_num, conf.block_length, conf.n_user))
        tx_full = np.empty((self.blocks_num, normalize_for_modulation(conf.block_length), conf.n_user))
        rx_full = np.empty((self.blocks_num, normalize_for_modulation(conf.block_length), conf.n_ant),
                           dtype=complex
                           if conf.modulation_type in [ModulationType.QPSK.name, ModulationType.EightPSK.name]
                           else float)
        # accumulate words until reaches desired number
        for index in range(conf.block_num):
            mx = self.generator.generate()
            tx = self.encoder.encode(mx)
            s = self.modulator.modulate(tx.T)
            rx = self.transmitter.transmit(s, snr, index)
            # accumulate
            mx_full[index] = mx
            tx_full[index] = tx
            rx_full[index] = rx

        database.append((mx_full, tx_full, rx_full))

    def __getitem__(self, snr_list: List[float]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        database = []
        # do not change max_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            [executor.submit(self.get_snr_data, snr, database) for snr in snr_list]
        mx, tx, rx = (np.concatenate(arrays) for arrays in zip(*database))
        mx, tx, rx = torch.Tensor(mx).to(device=DEVICE), torch.Tensor(tx).to(device=DEVICE), torch.from_numpy(rx).to(
            device=DEVICE)
        return mx, tx, rx

    def __len__(self):
        return self.block_length
