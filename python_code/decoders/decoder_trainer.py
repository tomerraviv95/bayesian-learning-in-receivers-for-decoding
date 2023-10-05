import os

import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from dir_definitions import ECC_MATRICES_DIR, BP_WEIGHTS
from python_code import conf, DEVICE
from python_code.datasets.coding_dataset import CodingDataset
from python_code.utils.coding_utils import get_code_pcm_and_gm
from python_code.utils.metrics import calculate_error_rate

ITERATIONS = 5
SNR_START = 4
SNR_END = 7
CODEWORDS_NUM = 100
MIN_EVAL_ERRORS = 500


class DecoderTrainer(nn.Module):
    def __init__(self):
        super().__init__()
        self._initialize_dataloader()
        self.softmax = torch.nn.Softmax(dim=1)  # Single symbol probability inference
        self.odd_llr_mask_only = True
        self.even_mask_only = True
        self.multiloss_output_mask_only = True
        self.output_mask_only = False
        self.multi_loss_flag = True
        self.iteration_num = ITERATIONS
        self._code_bits = conf.code_bits
        self._message_bits = conf.message_bits
        self.code_pcm, self.code_gm = get_code_pcm_and_gm(conf.code_bits, conf.message_bits, ECC_MATRICES_DIR,
                                                          conf.code_type)
        self.neurons = int(np.sum(self.code_pcm))
        if not os.path.exists(BP_WEIGHTS):
            os.makedirs(BP_WEIGHTS)
        self.model_name = os.path.join(BP_WEIGHTS, str(self) + "_" + str(self._code_bits) + "_" + str(
            self._message_bits) + "_" + str(self.iteration_num))

    # setup the optimization algorithm
    def deep_learning_setup(self, lr: float):
        """
        Sets up the optimizer and loss criterion
        """
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        self.criterion = BCEWithLogitsLoss().to(DEVICE)

    def _initialize_dataloader(self):
        """
        Sets up the data loader - a generator from which we draw batches, in iterations
        """
        self.train_dataset = CodingDataset(codewords_num=CODEWORDS_NUM)
        self.val_dataset = CodingDataset(codewords_num=CODEWORDS_NUM)

    def train_model(self):
        if os.path.exists(self.model_name + '.pth'):
            print("Loading existing model")
            self.load_state_dict(torch.load(self.model_name + '.pth'))
            return
        print(f"Training {self.__str__()} on AWGN Channel")
        avg_ber = self.eval()
        print(f"Training Loop {0}, BER {avg_ber}")
        for run_ind in range(self.total_runs):
            s, rx = self.train_dataset.__getitem__(snr_list=list(range(SNR_START, SNR_END + 1)))
            # train the decoder
            self.single_training(s, rx)
            avg_ber = self.eval()
            print(f"Training Loop {run_ind + 1}, BER {avg_ber}")
            torch.save(self.state_dict(), self.model_name + '.pth')

    def eval(self) -> float:
        """
        The evaluation running on multiple pairs of transmitted and received blocks.
        :return: list of ber per block
        """
        total_ber = []
        total_errors = 0
        with torch.no_grad():
            while total_errors < MIN_EVAL_ERRORS:
                tx, rx = self.val_dataset.__getitem__(snr_list=[SNR_END])
                # detect data part after training on the pilot part
                output = self.forward(rx)
                # calculate accuracy
                ber, errors = calculate_error_rate(output, tx)
                total_ber.append(ber)
                total_errors += errors
        avg_ber = sum(total_ber) / len(total_ber)
        return avg_ber
