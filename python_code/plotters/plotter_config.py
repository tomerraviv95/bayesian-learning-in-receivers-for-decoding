from enum import Enum
from typing import Tuple, List, Dict

from python_code.utils.constants import DetectorType, DecoderType


class PlotType(Enum):
    detection_comparison_by_SNR = 'detection_comparison_by_SNR'
    decoding_comparison_by_SNR = 'decoding_comparison_by_SNR'
    final_comparison_by_SNR = 'final_comparison_by_SNR'
    final_comparison_by_users = 'final_comparison_by_users'


def get_config(plot_type: PlotType) -> Tuple[List[Dict], list, str, str, List[int]]:
    ## Decoding with BP, detection methods vary
    if plot_type == PlotType.detection_comparison_by_SNR:
        params_dicts = [
            {'snr': 5, 'detector_type': DetectorType.seq_model.name},
            {'snr': 6, 'detector_type': DetectorType.seq_model.name},
            {'snr': 7, 'detector_type': DetectorType.seq_model.name},
            {'snr': 8, 'detector_type': DetectorType.seq_model.name},
            {'snr': 5, 'detector_type': DetectorType.bayesian.name},
            {'snr': 6, 'detector_type': DetectorType.bayesian.name},
            {'snr': 7, 'detector_type': DetectorType.bayesian.name},
            {'snr': 8, 'detector_type': DetectorType.bayesian.name},
            {'snr': 5, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 6, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 7, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 8, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 5, 'detector_type': DetectorType.black_box.name},
            {'snr': 6, 'detector_type': DetectorType.black_box.name},
            {'snr': 7, 'detector_type': DetectorType.black_box.name},
            {'snr': 8, 'detector_type': DetectorType.black_box.name},
            {'snr': 5, 'detector_type': DetectorType.bayesian_black_box.name},
            {'snr': 6, 'detector_type': DetectorType.bayesian_black_box.name},
            {'snr': 7, 'detector_type': DetectorType.bayesian_black_box.name},
            {'snr': 8, 'detector_type': DetectorType.bayesian_black_box.name},
        ]
        xlabel, ylabel = 'SNR [dB]', 'BER'
        to_plot_by_values = range(5, 9)
    ## Detection with DeepSIC, decoding methods vary
    elif plot_type == PlotType.decoding_comparison_by_SNR:
        params_dicts = [
            {'snr': 5, 'decoder_type': DecoderType.bp.name},
            {'snr': 6, 'decoder_type': DecoderType.bp.name},
            {'snr': 7, 'decoder_type': DecoderType.bp.name},
            {'snr': 8, 'decoder_type': DecoderType.bp.name},
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
        xlabel, ylabel = 'SNR [dB]', 'BER'
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
        xlabel, ylabel = 'SNR [dB]', 'BER'
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
        xlabel, ylabel = 'Number of Users and Antennas', 'BER'
        to_plot_by_values = [2, 4, 6, 8]
    else:
        raise ValueError('No such plot type!!!')

    return params_dicts, xlabel, ylabel, to_plot_by_values
