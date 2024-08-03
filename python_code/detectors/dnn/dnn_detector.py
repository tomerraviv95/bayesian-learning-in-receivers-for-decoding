import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.datasets.communication_blocks.modulator import MODULATION_NUM_MAPPING
from python_code.utils.constants import ModulationType

HIDDEN_SIZE = 32


class DNNDetector(nn.Module):
    """
    The DNNDetector Network Architecture
    """

    def __init__(self, n_user, n_ant):
        super(DNNDetector, self).__init__()
        self.n_user = n_user
        self.n_ant = n_ant
        self.base_rx_size = self.n_ant if conf.modulation_type == ModulationType.BPSK.name else 2 * self.n_ant
        self.initialize_dnn()

    def initialize_dnn(self):
        layers = [nn.Linear(self.base_rx_size, HIDDEN_SIZE),
                  nn.ReLU(),
                  nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                  nn.ReLU(),
                  nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                  nn.ReLU()]
        self.net = nn.Sequential(*layers).to(DEVICE)
        self.project_head = nn.Linear(HIDDEN_SIZE, MODULATION_NUM_MAPPING[conf.modulation_type] * self.n_ant).to(DEVICE)

    def forward(self, rx: torch.Tensor) -> torch.Tensor:
        embedding = self.net(rx)
        out = self.project_head(embedding).reshape(-1, self.n_ant, MODULATION_NUM_MAPPING[conf.modulation_type])
        return out
