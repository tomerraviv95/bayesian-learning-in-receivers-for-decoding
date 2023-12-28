## Implement the LBD method "Learnable Bernoulli Dropout for Bayesian Deep Learning"
import torch
from torch import nn

from python_code import DEVICE
from python_code.utils.bayesian_utils import dropout_ori, dropout_tilde, entropy, LossVariable, dropout
from python_code.utils.constants import Phase


class MaskedDeepSICDetector(nn.Module):

    def __init__(self, linear_input, hidden_size, classes_num, kl_scale):
        super(MaskedDeepSICDetector, self).__init__()
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(linear_input, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, classes_num)
        self.kl_scale = kl_scale

    def forward(self, rx: torch.Tensor, dropout_logits: torch.Tensor, phase: Phase = Phase.TEST) -> LossVariable:
        # add KL term if training
        if phase == Phase.TRAIN:
            x = self.activation(self.fc1(rx))
            x = self.activation(self.fc2(x))
            u = torch.rand(x.shape).to(DEVICE)
            x_after_dropout = dropout_ori(x, dropout_logits, u)
            out = self.fc3(x_after_dropout)
            # tilde
            x_tilde = dropout_tilde(x, dropout_logits, u)
            out_tilde = self.fc3(x_tilde)
            # KL term
            scaling1 = (self.kl_scale ** 2 / 2) * (torch.sigmoid(dropout_logits).reshape(-1))
            first_layer_kl = scaling1 * torch.norm(self.fc3.weight, dim=0) ** 2
            H1 = entropy(torch.sigmoid(dropout_logits).reshape(-1))
            kl_term = torch.mean(first_layer_kl - H1)
            return LossVariable(priors=out, u=u, arm_original=out, arm_tilde=out_tilde,
                                dropout_logit=dropout_logits, kl_term=kl_term)
        x = self.activation(self.fc1(rx))
        x = self.activation(self.fc2(x))
        u = torch.rand(x.shape).to(DEVICE)
        x = dropout(x, dropout_logits, u)
        out = self.fc3(x)
        return out
