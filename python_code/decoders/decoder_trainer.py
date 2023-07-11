import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import Adam, RMSprop, SGD

from dir_definitions import ECC_MATRICES_DIR
from python_code import conf, DEVICE
from python_code.datasets.coding_dataset import CodingDataset
from python_code.utils.coding_utils import get_code_pcm_and_gm

ITERATIONS = 5
SNR_START = 4
SNR_END = 7
TOTAL_RUNS = 5
CODEWORDS_NUM = 50

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
        self.code_pcm, self.code_gm = get_code_pcm_and_gm(conf.code_bits, conf.message_bits,
                                                          ECC_MATRICES_DIR, conf.code_type)
        self.neurons = int(np.sum(self.code_pcm))

    # setup the optimization algorithm
    def deep_learning_setup(self, lr: float):
        """
        Sets up the optimizer and loss criterion
        """
        if conf.optimizer_type == 'Adam':
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                  lr=lr)
        elif conf.optimizer_type == 'RMSprop':
            self.optimizer = RMSprop(filter(lambda p: p.requires_grad, self.parameters()),
                                     lr=lr)
        elif conf.optimizer_type == 'SGD':
            self.optimizer = SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                 lr=lr)
        else:
            raise NotImplementedError("No such optimizer implemented!!!")
        if conf.loss_type == 'CrossEntropy':
            self.criterion = CrossEntropyLoss().to(DEVICE)
        elif conf.loss_type == 'MSE':
            self.criterion = MSELoss().to(DEVICE)
        else:
            raise NotImplementedError("No such loss function implemented!!!")

    def _initialize_dataloader(self):
        """
        Sets up the data loader - a generator from which we draw batches, in iterations
        """
        self.train_dataset = CodingDataset(codewords_num=CODEWORDS_NUM)

    def train_model(self):
        print(f"Training {self.__str__()} on AWGN Channel")
        for run_ind in range(TOTAL_RUNS):
            tx, rx = self.train_dataset.__getitem__(snr_list=list(range(SNR_START, SNR_END + 1)))
            # train the decoder
            self.single_training(tx, rx)
            print(f"Training Loop {1 + run_ind}")
