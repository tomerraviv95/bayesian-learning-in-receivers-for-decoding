import math
import random
from typing import List, Tuple

import numpy as np
import torch

from python_code import conf
from python_code.channel.channel_dataset import ChannelModelDataset
from python_code.channel.modulator import MODULATION_NUM_MAPPING
from python_code.decoders.bp_decoder import BPDecoder
from python_code.detectors.detector import Detector
from python_code.utils.constants import ModulationType
from python_code.utils.metrics import calculate_ber
from python_code.utils.probs_utils import get_bits_from_qpsk_symbols, get_bits_from_eightpsk_symbols

random.seed(conf.seed)
torch.manual_seed(conf.seed)
torch.cuda.manual_seed(conf.seed)
np.random.seed(conf.seed)


class Evaluator(object):
    """
    Implements the evaluation pipeline. Start with initializing the dataloader, detector and decoder.
    Then, drawing from the triplets of message, transmitted and received words it runs the received words through
    the pipeline of detection and decoding. At each stage, we also calculate the BER at that given step (after detection
    and after decoding).
    """

    def __init__(self):
        self.constellation_bits = int(math.log2(MODULATION_NUM_MAPPING[conf.modulation_type]))
        # initialize matrices, datasets and detector
        self._initialize_dataloader()
        self._initialize_detector()
        self._initialize_decoder()

    def _initialize_detector(self):
        """
        Every trainer must have some base detector
        """
        self.detector = Detector()

    def _initialize_decoder(self):
        """
        Every trainer must have some base decoder
        """
        self.decoder = BPDecoder()

    def _initialize_dataloader(self):
        """
        Sets up the data loader - a generator from which we draw batches, in iterations
        """
        self.channel_dataset = ChannelModelDataset()

    def evaluate(self) -> Tuple[List[float], List[float]]:
        """
        The online evaluation run. Main function for running the experiments of sequential transmission of pilots and
        data blocks for the paper.
        :return: list of ber per timestep
        """
        detection_bers, decoding_bers = [], []
        # draw words for a given snr
        message_words, transmitted_words, received_words = self.channel_dataset.__getitem__(snr_list=[conf.snr])
        # detect sequentially
        for block_ind in range(conf.blocks_num):
            print('*' * 20)
            # get current word and channel
            mx, tx, rx = message_words[block_ind], transmitted_words[block_ind], received_words[block_ind]
            # split words into data and pilot part
            mx_pilot = mx[:conf.pilot_size // self.constellation_bits]
            mx_data = mx[conf.pilot_size // self.constellation_bits:]
            tx_pilot = tx[:conf.pilot_size // self.constellation_bits]
            tx_data = tx[conf.pilot_size // self.constellation_bits:]
            rx_pilot = rx[:conf.pilot_size // self.constellation_bits]
            rx_data = rx[conf.pilot_size // self.constellation_bits:]
            # run online training on the pilots part if desired
            if conf.is_online_training:
                self.detector._online_training(tx_pilot, rx_pilot)
            # detect data part after training on the pilot part
            detected_words, (confident_bits, confidence_word) = self.detector.forward(rx_data)
            # calculate accuracy for detection
            detection_ber = self.calculate_detection_ber(detected_words, rx, tx_data)
            detection_bers.append(detection_ber)
            print(f'current: {block_ind, detection_ber}')
            # use detected soft values to calculate the final message
            decoded_words = self.decoder.forward(confidence_word)
            # calculate accuracy for decoding
            decoded_ber = calculate_ber(decoded_words, mx_data)
            decoding_bers.append(decoded_ber)
            print(f'current: {block_ind, decoded_ber}')
        return decoding_bers, detection_bers

    def calculate_detection_ber(self, detected_word, rx, tx_data):
        detection_target = tx_data[:, :rx.shape[1]]
        if conf.modulation_type == ModulationType.QPSK.name:
            detection_target = get_bits_from_qpsk_symbols(detection_target)
        if conf.modulation_type == ModulationType.EightPSK.name:
            detection_target = get_bits_from_eightpsk_symbols(detection_target)
        detection_ber = calculate_ber(detected_word, detection_target)
        return detection_ber
