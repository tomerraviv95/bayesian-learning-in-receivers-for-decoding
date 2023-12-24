## Implement the LBD method "Learnable Bernoulli Dropout for Bayesian Deep Learning"

from typing import Union

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.datasets.communication_blocks.modulator import MODULATION_NUM_MAPPING
from python_code.utils import condrop
from python_code.utils.bayesian_utils import dropout_ori, dropout_tilde, entropy, LossVariable
from python_code.utils.constants import Phase, ModulationType, LOGITS_INIT
from python_code.utils.condrop import ConcreteDropout
from python_code.utils.probs_utils import sigmoid


@condrop.concrete_regulariser
class CBayesianDeepSICDetector(nn.Module):

    def __init__(self, ensemble_num):
        super(CBayesianDeepSICDetector, self).__init__()
        classes_num = MODULATION_NUM_MAPPING[conf.modulation_type]
        hidden_size = conf.hidden_base_size * classes_num
        base_rx_size = conf.n_ant if conf.modulation_type == ModulationType.BPSK.name else 2 * conf.n_ant
        linear_input = base_rx_size + (classes_num - 1) * (conf.n_user - 1)  # from DeepSIC paper
        self.ensemble_num = ensemble_num
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(linear_input, hidden_size)
        w, d = 1e-6, 1e-4
        self.cd1 = ConcreteDropout(weight_regulariser=w, dropout_regulariser=d,
                                   init_max=sigmoid(LOGITS_INIT)+0.001,init_min=sigmoid(LOGITS_INIT)-0.001)
        self.fc2 = nn.Linear(hidden_size, classes_num)

    def forward(self, rx: torch.Tensor, phase: Phase = Phase.TEST) -> Union[LossVariable, torch.Tensor]:
        # train
        if phase == Phase.TRAIN:
            # first layer
            x = self.fc1(rx)
            log_probs = self.cd1(x, nn.Sequential(self.fc2, self.activation))
            return log_probs
        # test
        log_probs = 0
        for ind_ensemble in range(self.ensemble_num):
            # first layer
            x = self.fc1(rx)
            log_probs += self.cd1(x, nn.Sequential(self.fc2, self.activation))
        return log_probs / self.ensemble_num


