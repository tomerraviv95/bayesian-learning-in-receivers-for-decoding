import torch
from torch import nn

from python_code import conf, DEVICE
from python_code.datasets.communication_blocks.modulator import MODULATION_NUM_MAPPING
from python_code.utils.bayesian_utils import dropout_ori
from python_code.utils.constants import ModulationType, LOGITS_INIT


class DeepSICDetector(nn.Module):
    """
    The DeepSIC Network Architecture

    ===========Architecture=========
    DeepSICNet(
      (fullyConnectedLayer): Linear(...)
      (reluLayer): ReLU()
      (fullyConnectedLayer2): Linear(...)
    ================================
    """

    def __init__(self):
        super(DeepSICDetector, self).__init__()
        classes_num = MODULATION_NUM_MAPPING[conf.modulation_type]
        hidden_size = conf.hidden_base_size * classes_num
        base_rx_size = conf.n_ant if conf.modulation_type == ModulationType.BPSK.name else 2 * conf.n_ant
        linear_input = base_rx_size + (classes_num - 1) * (conf.n_user - 1)  # from DeepSIC paper
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(linear_input, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, classes_num)
        self.dropout_logits = LOGITS_INIT * torch.ones([1, hidden_size], device=DEVICE)

    def forward(self, rx: torch.Tensor, apply_dropout: bool = False) -> torch.Tensor:
        x = self.activation(self.fc1(rx))
        x = self.activation(self.fc2(x))
        if apply_dropout:
            u = torch.rand(x.shape).to(DEVICE)
            x = dropout_ori(x, self.dropout_logits, u)
        out1 = self.fc3(x)
        return out1
