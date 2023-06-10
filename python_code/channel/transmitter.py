from typing import Tuple

import numpy as np

from python_code import conf
from python_code.channel.mimo_channels.sed_channel import SEDChannel


class Transmitter:

    def transmit(self, s: np.ndarray, snr: float, index: int) -> Tuple[np.ndarray, np.ndarray]:
        h = SEDChannel.calculate_channel(conf.n_ant, conf.n_user, index, conf.fading_in_channel)
        # pass through channel
        rx = SEDChannel.transmit(s=s, h=h, snr=snr)
        return rx.T
