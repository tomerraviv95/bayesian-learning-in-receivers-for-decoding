import numpy as np


def AWGNChannel(s, snr, rate, random):
    """
        Input: s - Transmitted codeword, snr - dB, rate - Code rate, use_llr - Return llr
        Output: rx - Codeword with AWGN noise
    """
    [row, col] = s.shape
    sigma = compute_channel_sigma(rate, snr)
    rx = s + sigma * random.normal(0.0, 1.0, (row, col))
    llr = compute_channel_llr(rx, sigma)
    return llr


def compute_channel_llr(rx, sigma):
    llr = rx * 2 / (sigma ** 2)
    return llr


def compute_channel_sigma(rate, snr):
    sigma = np.sqrt(0.5 * ((10 ** ((snr + 10 * np.log10(rate)) / 10)) ** (-1)))
    return sigma
