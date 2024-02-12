from enum import Enum
from typing import Tuple, List, Dict

from python_code.utils.constants import DetectorType, DecoderType


class PlotType(Enum):
    detection_comparison_by_SNR_QPSK = 'detection_comparison_by_SNR_QPSK'
    detection_comparison_by_SNR_EightPSK = 'detection_comparison_by_SNR_EightPSK'
    detection_comparison_by_SNR_cost_QPSK = 'detection_comparison_by_SNR_cost_QPSK'
    detection_comparison_by_SNR_cost_EightPSK = 'detection_comparison_by_SNR_cost_EightPSK'
    ber_by_ece = 'ber_by_ece'
    iterations_ablation = 'iterations_ablation'
    final_comparison_by_SNR_EightPSK = 'final_comparison_by_SNR_EightPSK'
    final_comparison_by_SNR_cost_EightPSK = 'final_comparison_by_SNR_cost_EightPSK'


def get_config(plot_type: PlotType) -> Tuple[List[Dict], List[int]]:
    ## Decoding with BP, detection methods vary
    # Figure 3a
    if plot_type == PlotType.detection_comparison_by_SNR_QPSK:
        params_dicts = [
            {'snr': 6, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'QPSK'},
            {'snr': 8, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'QPSK'},
            {'snr': 10, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'QPSK'},
            {'snr': 12, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'QPSK'},
            {'snr': 6, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 8, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 10, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 12, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 6, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 8, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 10, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 12, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'QPSK'},
        ]
        to_plot_by_values = range(6, 13, 2)
    # Figure 3b
    elif plot_type == PlotType.detection_comparison_by_SNR_EightPSK:
        params_dicts = [
            {'snr': 10, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'EightPSK'},
            {'snr': 12, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'EightPSK'},
            {'snr': 14, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'EightPSK'},
            {'snr': 16, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'EightPSK'},
            {'snr': 10, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 12, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 14, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 16, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 10, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 12, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 14, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 16, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'EightPSK'},
        ]
        to_plot_by_values = range(10, 17, 2)
    # Figure 4a
    elif plot_type == PlotType.detection_comparison_by_SNR_cost_QPSK:
        params_dicts = [
            {'snr': 4, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'QPSK'},
            {'snr': 6, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'QPSK'},
            {'snr': 8, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'QPSK'},
            {'snr': 10, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'QPSK'},
            {'snr': 4, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 6, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 8, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 10, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 4, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 6, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 8, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'QPSK'},
            {'snr': 10, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'QPSK'},
        ]
        to_plot_by_values = range(4, 11, 2)
    # Figure 4b
    elif plot_type == PlotType.detection_comparison_by_SNR_cost_EightPSK:
        params_dicts = [
            {'snr': 10, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'EightPSK'},
            {'snr': 12, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'EightPSK'},
            {'snr': 14, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'EightPSK'},
            {'snr': 16, 'detector_type': DetectorType.seq_model.name, 'modulation_type': 'EightPSK'},
            {'snr': 10, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 12, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 14, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 16, 'detector_type': DetectorType.bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 10, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 12, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 14, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'EightPSK'},
            {'snr': 16, 'detector_type': DetectorType.model_based_bayesian.name, 'modulation_type': 'EightPSK'},
        ]
        to_plot_by_values = range(10, 17, 2)
    # Figure 5
    elif plot_type == PlotType.ber_by_ece:
        params_dicts = [
            {'snr': 8, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.seq_model.name},
            {'snr': 10, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.seq_model.name},
            {'snr': 12, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.seq_model.name},
            {'snr': 14, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.seq_model.name},
            {'snr': 8, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 10, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 12, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 14, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.model_based_bayesian.name},
        ]
        to_plot_by_values = range(8, 15, 2)
    # Figure 6
    elif plot_type == PlotType.iterations_ablation:
        params_dicts = [
            {'detector_type': DetectorType.seq_model.name, 'deepsic_iterations': 1},
            {'detector_type': DetectorType.seq_model.name, 'deepsic_iterations': 2},
            {'detector_type': DetectorType.seq_model.name, 'deepsic_iterations': 3},
            {'detector_type': DetectorType.bayesian.name, 'deepsic_iterations': 1},
            {'detector_type': DetectorType.bayesian.name, 'deepsic_iterations': 2},
            {'detector_type': DetectorType.bayesian.name, 'deepsic_iterations': 3},
            {'detector_type': DetectorType.model_based_bayesian.name, 'deepsic_iterations': 1},
            {'detector_type': DetectorType.model_based_bayesian.name, 'deepsic_iterations': 2},
            {'detector_type': DetectorType.model_based_bayesian.name, 'deepsic_iterations': 3},
        ]
        to_plot_by_values = [1, 2, 3, 4]
    ## Detection and decoding methods vary with snr
    elif plot_type == PlotType.final_comparison_by_SNR_EightPSK:
        params_dicts = [
            {'snr': 10, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.seq_model.name},
            {'snr': 12, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.seq_model.name},
            {'snr': 14, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.seq_model.name},
            {'snr': 16, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.seq_model.name},
            {'snr': 10, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.seq_model.name},
            {'snr': 12, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.seq_model.name},
            {'snr': 14, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.seq_model.name},
            {'snr': 16, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.seq_model.name},
            {'snr': 10, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 12, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 14, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 16, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 10, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 12, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 14, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 16, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.model_based_bayesian.name},
        ]
        to_plot_by_values = range(10, 17, 2)
    elif plot_type == PlotType.final_comparison_by_SNR_cost_EightPSK:
        params_dicts = [
            {'snr': 10, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.seq_model.name},
            {'snr': 12, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.seq_model.name},
            {'snr': 14, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.seq_model.name},
            {'snr': 16, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.seq_model.name},
            {'snr': 10, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.seq_model.name},
            {'snr': 12, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.seq_model.name},
            {'snr': 14, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.seq_model.name},
            {'snr': 16, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.seq_model.name},
            {'snr': 10, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 12, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 14, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 16, 'decoder_type': DecoderType.wbp.name, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 10, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 12, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 14, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 16, 'decoder_type': DecoderType.modular_bayesian_wbp.name,
             'detector_type': DetectorType.model_based_bayesian.name},
        ]
        to_plot_by_values = range(10, 17, 2)
    else:
        raise ValueError('No such plot type!!!')

    return params_dicts, to_plot_by_values
