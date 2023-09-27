from typing import Tuple

import numpy as np

from python_code import conf
from python_code.datasets.mimo_channels.cost_channel import Cost2100MIMOChannel
from python_code.datasets.mimo_channels.sed_channel import SEDChannel
from python_code.utils.constants import ChannelModels

MIMO_CHANNELS_DICT = {ChannelModels.Synthetic.name: SEDChannel,
                      ChannelModels.Cost2100.name: Cost2100MIMOChannel}


class Transmitter:
    def transmit(self, s: np.ndarray, snr: float, index: int) -> Tuple[np.ndarray, np.ndarray]:
        if conf.channel_model == ChannelModels.Synthetic.name:
            h = SEDChannel.calculate_channel(conf.n_ant, conf.n_user, index, conf.fading_in_channel)
        elif conf.channel_model == ChannelModels.Cost2100.name:
            h = Cost2100MIMOChannel.calculate_channel(conf.n_ant, conf.n_user, index, conf.fading_in_channel)
        else:
            raise ValueError("No such channel model!!!")
        # pass through datasets
        rx = MIMO_CHANNELS_DICT[conf.channel_model].transmit(s=s, h=h, snr=snr)
        return rx.T
