import numpy as np

from python_code import conf
from python_code.utils.constants import ModulationType

H_COEF = 0.8


class SEDChannel:
    @staticmethod
    def calculate_channel(n_ant: int, n_user: int, frame_ind: int, fading: bool) -> np.ndarray:
        H_row = np.array([i for i in range(n_ant)])
        H_row = np.tile(H_row, [n_user, 1]).T
        H_column = np.array([i for i in range(n_user)])
        H_column = np.tile(H_column, [n_ant, 1])
        H = np.exp(-np.abs(H_row - H_column))
        if fading:
            H = SEDChannel._add_fading(H, n_ant, frame_ind)
        return H

    @staticmethod
    def _add_fading(H: np.ndarray, n_ant: int, frame_ind: int) -> np.ndarray:
        degs_array = np.array([51, 39, 33, 21])
        fade_mat = H_COEF + (1 - H_COEF) * np.cos(2 * np.pi * frame_ind / degs_array)
        fade_mat = np.tile(fade_mat.reshape(1, -1), [n_ant, 1])
        return H * fade_mat

    @staticmethod
    def transmit(s: np.ndarray, h: np.ndarray, snr: float) -> np.ndarray:
        """
        The MIMO SED Channel
        :param s: to transmit symbol words
        :param snr: signal-to-noise value
        :param h: datasets function
        :return: received word
        """
        conv = SEDChannel._compute_channel_signal_convolution(h, s)
        var = 10 ** (-0.1 * snr)
        if conf.modulation_type == ModulationType.BPSK.name:
            w = np.sqrt(var) * np.random.randn(conf.n_ant, s.shape[1])
        else:
            w_real = np.sqrt(var) / 2 * np.random.randn(conf.n_ant, s.shape[1])
            w_imag = np.sqrt(var) / 2 * np.random.randn(conf.n_ant, s.shape[1]) * 1j
            w = w_real + w_imag
        y = conv + w
        if conf.linear:
            return y
        # non-linear output - simulates a saturation phenomena at the receiver
        return np.tanh(y)

    @staticmethod
    def _compute_channel_signal_convolution(h: np.ndarray, s: np.ndarray) -> np.ndarray:
        conv = np.matmul(h, s)
        return conv
