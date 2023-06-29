from enum import Enum
from typing import Tuple, List, Dict

from python_code.utils.constants import DetectorType


class PlotType(Enum):
    QPSK_BER_VS_SER = 'QPSK_BER_VS_SER'
    QPSK_SNR = 'QPSK_SNR'


def get_config(plot_type: PlotType) -> Tuple[List[Dict], list, str, str, List[int]]:
    if plot_type == PlotType.QPSK_BER_VS_SER:
        params_dicts = [
            {'snr': 6, 'detector_type': DetectorType.black_box.name},
            {'snr': 7, 'detector_type': DetectorType.black_box.name},
            {'snr': 8, 'detector_type': DetectorType.black_box.name},
            {'snr': 9, 'detector_type': DetectorType.black_box.name},
            {'snr': 6, 'detector_type': DetectorType.seq_model.name},
            {'snr': 7, 'detector_type': DetectorType.seq_model.name},
            {'snr': 8, 'detector_type': DetectorType.seq_model.name},
            {'snr': 9, 'detector_type': DetectorType.seq_model.name},
            {'snr': 6, 'detector_type': DetectorType.bayesian.name},
            {'snr': 7, 'detector_type': DetectorType.bayesian.name},
            {'snr': 8, 'detector_type': DetectorType.bayesian.name},
            {'snr': 9, 'detector_type': DetectorType.bayesian.name},
            {'snr': 6, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 7, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 8, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 9, 'detector_type': DetectorType.model_based_bayesian.name},
        ]
        xlabel, ylabel = 'SER', 'BER'
        to_plot_by_values = range(8, 9)
    elif plot_type == PlotType.QPSK_SNR:
        params_dicts = [
            {'snr': 6, 'detector_type': DetectorType.seq_model.name},
            {'snr': 7, 'detector_type': DetectorType.seq_model.name},
            {'snr': 8, 'detector_type': DetectorType.seq_model.name},
            {'snr': 9, 'detector_type': DetectorType.seq_model.name},
            {'snr': 10, 'detector_type': DetectorType.seq_model.name},
            {'snr': 11, 'detector_type': DetectorType.seq_model.name},
            {'snr': 6, 'detector_type': DetectorType.bayesian.name},
            {'snr': 7, 'detector_type': DetectorType.bayesian.name},
            {'snr': 8, 'detector_type': DetectorType.bayesian.name},
            {'snr': 9, 'detector_type': DetectorType.bayesian.name},
            {'snr': 10, 'detector_type': DetectorType.bayesian.name},
            {'snr': 11, 'detector_type': DetectorType.bayesian.name},
            {'snr': 6, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 7, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 8, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 9, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 10, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 11, 'detector_type': DetectorType.model_based_bayesian.name}
        ]
        xlabel, ylabel = 'SNR [dB]', 'BER'
        to_plot_by_values = range(6, 10)
    else:
        raise ValueError('No such plot type!!!')

    return params_dicts, xlabel, ylabel, to_plot_by_values
