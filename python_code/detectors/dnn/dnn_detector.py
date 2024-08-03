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
                  nn.ReLU(),
                  ]
        self.net = nn.Sequential(*layers).to(DEVICE)
        self.projecion_heads = [nn.Linear(HIDDEN_SIZE, MODULATION_NUM_MAPPING[conf.modulation_type]).to(DEVICE) for _ in
                                range(self.n_ant)]

    def forward(self, rx: torch.Tensor) -> torch.Tensor:
        embedding = self.net(rx)
        soft_estimation = []
        for head in self.projecion_heads:
            soft_estimation.append(head(embedding).unsqueeze(1))
        return torch.cat(soft_estimation, dim=1)
