import math
import random
from collections import namedtuple
from typing import List, Tuple

import numpy as np
import torch

from python_code import conf, DEVICE
from python_code.datasets.channel_dataset import ChannelModelDataset
from python_code.datasets.communication_blocks.modulator import MODULATION_NUM_MAPPING, BPSKModulator
from python_code.datasets.siso_channels.awgn_channel import compute_channel_llr, compute_channel_sigma
from python_code.decoders import DECODERS_TYPE_DICT
from python_code.detectors import DETECTORS_TYPE_DICT
from python_code.utils.constants import ModulationType, HALF
from python_code.utils.metrics import calculate_error_rate, calculate_reliability_and_ece
from python_code.utils.probs_utils import get_qpsk_symbols_from_bits, get_eightpsk_symbols_from_bits

random.seed(conf.seed)
torch.manual_seed(conf.seed)
torch.cuda.manual_seed(conf.seed)
np.random.seed(conf.seed)

MetricOutput = namedtuple(
    "MetricOutput",
    "ser_list ber_list ece_list"
)


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
        Every evaluater must have some base detector
        """
        self.detector = DETECTORS_TYPE_DICT[conf.detector_type]()

    def _initialize_decoder(self):
        """
        Every evaluater must have some base decoder
        """
        self.decoder = DECODERS_TYPE_DICT[conf.decoder_type]()

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
        print(f"Detecting using {str(self.detector)}, decoding using {str(self.decoder)}")
        self.decoder.train_model()
        ser_list, ber_list, ece_list = [], [], []
        # draw words for a given snr
        message_words, transmitted_words, received_words = self.channel_dataset.__getitem__(snr_list=[conf.snr])
        # detect sequentially
        for block_ind in range(conf.blocks_num):
            print('*' * 20)
            print(f'current: {block_ind}')
            # get current word and datasets
            mx, tx, rx = message_words[block_ind], transmitted_words[block_ind], received_words[block_ind]
            # split words into data and pilot part
            uncoded_pilots_end_ind = int(conf.pilots_length)
            pilots_end_ind = int(conf.pilots_length // self.constellation_bits / conf.message_bits * conf.code_bits)
            mx_pilot, tx_pilot, rx_pilot = mx[:uncoded_pilots_end_ind], tx[:pilots_end_ind], rx[:pilots_end_ind]
            mx_data, tx_data, rx_data = mx[uncoded_pilots_end_ind:], tx[pilots_end_ind:], rx[pilots_end_ind:]
            # run online training on the pilots part
            self.detector._online_training(tx_pilot, rx_pilot)
            # detect data part after training on the pilot part
            detected_words, soft_confidences = self.detector.forward(rx_data)
            # calculate accuracy for detection
            detected_symbols_words = self.get_detected_symbols_words(detected_words)
            ser, _ = calculate_error_rate(detected_symbols_words, tx_data)
            ser_list.append(ser)
            print(f'symbol error rate: {ser}')
            # calculate ece measure
            ece = self.calculate_ece(tx_data, detected_symbols_words, soft_confidences)
            ece_list.append(ece)
            print(f'expected calibration error (ECE): {ece}')
            # use detected soft values to calculate the final message
            ber = self.calculate_ber(soft_confidences, detected_words, mx_data)
            ber_list.append(ber)
            print(f'bit error rate: {ber}')
        metrics_output = MetricOutput(ber_list=ber_list, ser_list=ser_list, ece_list=ece_list)
        return metrics_output

    @staticmethod
    def get_detected_symbols_words(detected_words):
        if conf.modulation_type == ModulationType.BPSK.name:
            detected_symbols_words = detected_words
        if conf.modulation_type == ModulationType.QPSK.name:
            detected_symbols_words = torch.Tensor(get_qpsk_symbols_from_bits(detected_words.cpu().numpy())).to(DEVICE)
        if conf.modulation_type == ModulationType.EightPSK.name:
            detected_symbols_words = torch.Tensor(get_eightpsk_symbols_from_bits(detected_words.cpu().numpy())).to(
                DEVICE)
        return detected_symbols_words

    def calculate_ber(self, confidence_word, detected_words, mx_data):
        if conf.modulation_type in [ModulationType.QPSK.name, ModulationType.EightPSK.name]:
            confidence_word = torch.repeat_interleave(confidence_word,
                                                      int(math.log2(MODULATION_NUM_MAPPING[conf.modulation_type])),
                                                      dim=0)
        rate = float(conf.message_bits / conf.code_bits)
        sigma = compute_channel_sigma(rate, conf.snr)
        to_decode_word = compute_channel_llr(BPSKModulator.modulate(detected_words) * (confidence_word - HALF), sigma)
        decoded_words = torch.zeros_like(mx_data)
        no_coding_words = torch.zeros_like(mx_data)
        for user in range(conf.n_user):
            current_to_decode = to_decode_word[:, user].reshape(-1, conf.code_bits)
            decoded_word = self.decoder.forward(current_to_decode)
            no_coding_word = current_to_decode < 0
            message_decoded_word = decoded_word[:, conf.message_bits:]
            message_no_coding_word = no_coding_word[:, conf.message_bits:]
            decoded_words[:, user] = message_decoded_word.reshape(-1)
            no_coding_words[:, user] = message_no_coding_word.reshape(-1)
        decoded_ber, _ = calculate_error_rate(decoded_words, mx_data)
        no_coding_ber, _ = calculate_error_rate(no_coding_words, mx_data)
        return decoded_ber

    @staticmethod
    def calculate_ece(tx_data, detected_symbols_words, soft_confidences):
        correct_values = soft_confidences[torch.eq(tx_data, detected_symbols_words)].tolist()
        error_values = soft_confidences[~torch.eq(tx_data, detected_symbols_words)].tolist()
        binning_regions = np.linspace(start=0, stop=1, num=9)
        ece_measure, avg_acc_per_bin, avg_confidence_per_bin = calculate_reliability_and_ece(correct_values,
                                                                                             error_values,
                                                                                             binning_regions)
        return ece_measure


if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate()
