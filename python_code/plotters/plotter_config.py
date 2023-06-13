from enum import Enum
from typing import Tuple, List, Dict

from python_code.utils.constants import DetectorType


class PlotType(Enum):
    ## The Three Figures for the Paper
    basic = 'basic'


def get_config(plot_type: PlotType) -> Tuple[List[Dict], list, str, str]:
    if plot_type == PlotType.basic:
        params_dicts = [
            # {'snr': 9, 'detector_type': DetectorType.black_box.name},
            {'snr': 10, 'detector_type': DetectorType.black_box.name},
            # {'snr': 11, 'detector_type': DetectorType.black_box.name},
            # {'snr': 9, 'detector_type': DetectorType.seq_model.name},
            {'snr': 10, 'detector_type': DetectorType.seq_model.name},
            # {'snr': 11, 'detector_type': DetectorType.seq_model.name},
        ]
        xlabel, ylabel = 'Detection BERs', 'Decoding BERs'
    else:
        raise ValueError('No such plot type!!!')

    return params_dicts, xlabel, ylabel
