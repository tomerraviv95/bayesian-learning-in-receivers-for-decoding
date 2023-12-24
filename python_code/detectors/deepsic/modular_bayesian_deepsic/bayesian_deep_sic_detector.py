## Implement the LBD method "Learnable Bernoulli Dropout for Bayesian Deep Learning"

from typing import Union

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.datasets.communication_blocks.modulator import MODULATION_NUM_MAPPING
from python_code.utils.bayesian_utils import dropout_ori, dropout_tilde, entropy, LossVariable
from python_code.utils.constants import Phase, ModulationType, LOGITS_INIT


class BayesianDeepSICDetector(nn.Module):

    def __init__(self, ensemble_num, kl_scale):
        super(BayesianDeepSICDetector, self).__init__()
        classes_num = MODULATION_NUM_MAPPING[conf.modulation_type]
        hidden_size = conf.hidden_base_size * classes_num
        base_rx_size = conf.n_ant if conf.modulation_type == ModulationType.BPSK.name else 2 * conf.n_ant
        linear_input = base_rx_size + (classes_num - 1) * (conf.n_user - 1)  # from DeepSIC paper
        self.activation = nn.ReLU()
        self.ensemble_num = ensemble_num
        self.kl_scale = kl_scale
        self.fc1 = nn.Linear(linear_input, hidden_size)
        self.fc2 = nn.Linear(hidden_size, classes_num)
        self.dropout_logit = nn.Parameter(LOGITS_INIT * torch.ones(hidden_size).reshape(1, -1).to(DEVICE))

    def forward(self, rx: torch.Tensor, phase: Phase = Phase.TEST) -> Union[LossVariable, torch.Tensor]:
        log_probs = 0
        arm_original, arm_tilde, kl_term = [], [], 0

        if phase == Phase.TEST:
            for ind_ensemble in range(self.ensemble_num):
                # first layer
                x = self.activation(self.fc1(rx))
                u = torch.rand(x.shape).to(DEVICE)
                x_after_dropout = dropout_ori(x, self.dropout_logit, u)
                # second layer
                out = self.fc2(x_after_dropout)
                # if in train phase, keep parameters in list and compute the tilde output for arm loss calculation
                log_probs += out
            return log_probs / self.ensemble_num
        else:
            # first layer
            x = self.activation(self.fc1(rx))
            u = torch.rand(x.shape).to(DEVICE)
            x_after_dropout = dropout_ori(x, self.dropout_logit, u)
            # second layer
            out = self.fc2(x_after_dropout)
            # if in train phase, keep parameters in list and compute the tilde output for arm loss calculation
            log_probs += out
            # compute first variable output
            arm_original.append(out)
            # compute second variable output
            x_tilde = dropout_tilde(x, self.dropout_logit, u)
            out_tilde = self.fc2(x_tilde)
            arm_tilde.append(out_tilde)
            # KL term
            scaling1 = (self.kl_scale ** 2 / 2) * (torch.sigmoid(self.dropout_logit).reshape(-1))
            first_layer_kl = scaling1 * torch.norm(self.fc1.weight, dim=1) ** 2
            H1 = entropy(torch.sigmoid(self.dropout_logit).reshape(-1))
            kl_term = torch.mean(first_layer_kl - H1)
            return LossVariable(priors=log_probs, arm_original=arm_original, arm_tilde=arm_tilde,
                                u=u, kl_term=kl_term, dropout_logit=self.dropout_logit)
