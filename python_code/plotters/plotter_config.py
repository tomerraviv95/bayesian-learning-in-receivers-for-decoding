from enum import Enum
from typing import Tuple, List, Dict

from python_code.utils.constants import DetectorType, DecoderType


class PlotType(Enum):
    detection_comparison_by_SNR_QPSK = 'detection_comparison_by_SNR_QPSK'
    detection_comparison_by_SNR_EightPSK = 'detection_comparison_by_SNR_EightPSK'
    detection_comparison_by_SNR_cost_QPSK = 'detection_comparison_by_SNR_cost_QPSK'
    detection_comparison_by_SNR_cost_EightPSK = 'detection_comparison_by_SNR_cost_EightPSK'

    decoding_comparison_by_SNR = 'decoding_comparison_by_SNR'
    final_comparison_by_SNR = 'final_comparison_by_SNR'
    final_comparison_by_users = 'final_comparison_by_users'
    ber_by_ser = 'ber_by_ser'
    ece_by_pilots_length = 'ece_by_pilots_length'


def get_config(plot_type: PlotType) -> Tuple[List[Dict], List[int]]:
    ## Decoding with BP, detection methods vary
    if plot_type == PlotType.detection_comparison_by_SNR_QPSK:
        params_dicts = [
            {'snr': 4, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 6, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 8, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 10, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 4, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'QPSK'},
            {'snr': 6, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'QPSK'},
            {'snr': 8, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'QPSK'},
            {'snr': 10, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'QPSK'},
            {'snr': 4, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 6, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 8, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 10, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'QPSK'},
        ]
        to_plot_by_values = range(4, 11, 2)
    elif plot_type == PlotType.detection_comparison_by_SNR_EightPSK:
        params_dicts = [
            {'snr': 8, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 10, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 12, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 14, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 8, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'EightPSK'},
            {'snr': 10, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'EightPSK'},
            {'snr': 12, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'EightPSK'},
            {'snr': 14, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'EightPSK'},
            {'snr': 8, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 10, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 12, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 14, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'EightPSK'},
        ]
        to_plot_by_values = range(8, 15, 2)
    elif plot_type == PlotType.detection_comparison_by_SNR_cost_QPSK:
        params_dicts = [
            {'snr': 4, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 6, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 8, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 10, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 4, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'QPSK'},
            {'snr': 6, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'QPSK'},
            {'snr': 8, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'QPSK'},
            {'snr': 10, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'QPSK'},
            {'snr': 4, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 6, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 8, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 10, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'QPSK'},
        ]
        to_plot_by_values = range(4, 11, 2)
    elif plot_type == PlotType.detection_comparison_by_SNR_cost_EightPSK:
        params_dicts = [
            {'snr': 8, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 10, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 12, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 14, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 8, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'EightPSK'},
            {'snr': 10, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'EightPSK'},
            {'snr': 12, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'EightPSK'},
            {'snr': 14, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'EightPSK'},
            {'snr': 8, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 10, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 12, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 14, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'EightPSK'},
        ]
        to_plot_by_values = range(8, 15, 2)
    ## Detection with DeepSIC, decoding methods vary
    elif plot_type == PlotType.decoding_comparison_by_SNR:
        params_dicts = [
            {'snr': 5, 'decoder_type': DecoderType.wbp.name},
            {'snr': 6, 'decoder_type': DecoderType.wbp.name},
            {'snr': 7, 'decoder_type': DecoderType.wbp.name},
            {'snr': 8, 'decoder_type': DecoderType.wbp.name},
            {'snr': 5, 'decoder_type': DecoderType.bayesian_wbp.name},
            {'snr': 6, 'decoder_type': DecoderType.bayesian_wbp.name},
            {'snr': 7, 'decoder_type': DecoderType.bayesian_wbp.name},
            {'snr': 8, 'decoder_type': DecoderType.bayesian_wbp.name},
            {'snr': 5, 'decoder_type': DecoderType.modular_bayesian_wbp.name},
            {'snr': 6, 'decoder_type': DecoderType.modular_bayesian_wbp.name},
            {'snr': 7, 'decoder_type': DecoderType.modular_bayesian_wbp.name},
            {'snr': 8, 'decoder_type': DecoderType.modular_bayesian_wbp.name},
        ]
        to_plot_by_values = range(5, 9)
    ## Detection and decoding methods vary with SNR
    elif plot_type == PlotType.final_comparison_by_SNR:
        params_dicts = [
            {'snr': 5, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.seq_model.name},
            {'snr': 6, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.seq_model.name},
            {'snr': 7, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.seq_model.name},
            {'snr': 8, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.seq_model.name},
            {'snr': 5, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.seq_model.name},
            {'snr': 6, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.seq_model.name},
            {'snr': 7, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.seq_model.name},
            {'snr': 8, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.seq_model.name},
            {'snr': 5, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 6, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 7, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 8, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 5, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 6, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 7, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 8, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.model_based_bayesian.name},
        ]
        to_plot_by_values = range(5, 9)
    ## Detection and decoding methods vary with users and antennas
    elif plot_type == PlotType.final_comparison_by_users:
        params_dicts = [
            {'n_user': 2, 'n_ant': 2, 'decoder_type': DecoderType.wbp.name,
             'detector_type': DetectorType.seq_model.name},
            {'n_user': 4, 'n_ant': 4, 'decoder_type': DecoderType.wbp.name,
             'detector_type': DetectorType.seq_model.name},
            {'n_user': 6, 'n_ant': 6, 'decoder_type': DecoderType.wbp.name,
             'detector_type': DetectorType.seq_model.name},
            {'n_user': 8, 'n_ant': 8, 'decoder_type': DecoderType.wbp.name,
             'detector_type': DetectorType.seq_model.name},
            {'n_user': 2, 'n_ant': 2, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.seq_model.name},
            {'n_user': 4, 'n_ant': 4, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.seq_model.name},
            {'n_user': 6, 'n_ant': 6, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.seq_model.name},
            {'n_user': 8, 'n_ant': 8, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.seq_model.name},
            {'n_user': 2, 'n_ant': 2, 'decoder_type': DecoderType.wbp.name,
             'detector_type': DetectorType.model_based_bayesian.name},
            {'n_user': 4, 'n_ant': 4, 'decoder_type': DecoderType.wbp.name,
             'detector_type': DetectorType.model_based_bayesian.name},
            {'n_user': 6, 'n_ant': 6, 'decoder_type': DecoderType.wbp.name,
             'detector_type': DetectorType.model_based_bayesian.name},
            {'n_user': 8, 'n_ant': 8, 'decoder_type': DecoderType.wbp.name,
             'detector_type': DetectorType.model_based_bayesian.name},
            {'n_user': 2, 'n_ant': 2, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.model_based_bayesian.name},
            {'n_user': 4, 'n_ant': 4, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.model_based_bayesian.name},
            {'n_user': 6, 'n_ant': 6, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.model_based_bayesian.name},
            {'n_user': 8, 'n_ant': 8, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.model_based_bayesian.name},
        ]
        to_plot_by_values = [2, 4, 6, 8]
    elif plot_type == PlotType.ber_by_ser:
        params_dicts = [
            {'detector_type': DetectorType.seq_model.name, 'pilots_length': 128},
            {'detector_type': DetectorType.seq_model.name, 'pilots_length': 256},
            {'detector_type': DetectorType.seq_model.name, 'pilots_length': 384},
            {'detector_type': DetectorType.seq_model.name, 'pilots_length': 512},
            {'detector_type': DetectorType.seq_model.name, 'pilots_length': 640},
            {'detector_type': DetectorType.seq_model.name, 'pilots_length': 768},
            {'detector_type': DetectorType.seq_model.name, 'pilots_length': 896},
            {'detector_type': DetectorType.bayesian.name, 'pilots_length': 128},
            {'detector_type': DetectorType.bayesian.name, 'pilots_length': 256},
            {'detector_type': DetectorType.bayesian.name, 'pilots_length': 384},
            {'detector_type': DetectorType.bayesian.name, 'pilots_length': 512},
            {'detector_type': DetectorType.bayesian.name, 'pilots_length': 640},
            {'detector_type': DetectorType.bayesian.name, 'pilots_length': 768},
            {'detector_type': DetectorType.bayesian.name, 'pilots_length': 896},
            {'detector_type': DetectorType.model_based_bayesian.name, 'pilots_length': 128},
            {'detector_type': DetectorType.model_based_bayesian.name, 'pilots_length': 256},
            {'detector_type': DetectorType.model_based_bayesian.name, 'pilots_length': 384},
            {'detector_type': DetectorType.model_based_bayesian.name, 'pilots_length': 512},
            {'detector_type': DetectorType.model_based_bayesian.name, 'pilots_length': 640},
            {'detector_type': DetectorType.model_based_bayesian.name, 'pilots_length': 768},
            {'detector_type': DetectorType.model_based_bayesian.name, 'pilots_length': 896},
        ]
        to_plot_by_values = None
    elif plot_type == PlotType.ece_by_pilots_length:
        params_dicts = [
            {'detector_type': DetectorType.seq_model.name, 'pilots_length': 128},
            {'detector_type': DetectorType.seq_model.name, 'pilots_length': 256},
            {'detector_type': DetectorType.seq_model.name, 'pilots_length': 384},
            {'detector_type': DetectorType.seq_model.name, 'pilots_length': 512},
            {'detector_type': DetectorType.seq_model.name, 'pilots_length': 640},
            {'detector_type': DetectorType.seq_model.name, 'pilots_length': 768},
            {'detector_type': DetectorType.seq_model.name, 'pilots_length': 896},
            {'detector_type': DetectorType.bayesian.name, 'pilots_length': 128},
            {'detector_type': DetectorType.bayesian.name, 'pilots_length': 256},
            {'detector_type': DetectorType.bayesian.name, 'pilots_length': 384},
            {'detector_type': DetectorType.bayesian.name, 'pilots_length': 512},
            {'detector_type': DetectorType.bayesian.name, 'pilots_length': 640},
            {'detector_type': DetectorType.bayesian.name, 'pilots_length': 768},
            {'detector_type': DetectorType.bayesian.name, 'pilots_length': 896},
            {'detector_type': DetectorType.model_based_bayesian.name, 'pilots_length': 128},
            {'detector_type': DetectorType.model_based_bayesian.name, 'pilots_length': 256},
            {'detector_type': DetectorType.model_based_bayesian.name, 'pilots_length': 384},
            {'detector_type': DetectorType.model_based_bayesian.name, 'pilots_length': 512},
            {'detector_type': DetectorType.model_based_bayesian.name, 'pilots_length': 640},
            {'detector_type': DetectorType.model_based_bayesian.name, 'pilots_length': 768},
            {'detector_type': DetectorType.model_based_bayesian.name, 'pilots_length': 896},
        ]
        to_plot_by_values = [128, 256, 384, 512, 640, 768, 896]
    else:
        raise ValueError('No such plot type!!!')

    return params_dicts, to_plot_by_values
