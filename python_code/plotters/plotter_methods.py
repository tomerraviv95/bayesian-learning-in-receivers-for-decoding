import os
from collections import namedtuple
from typing import Tuple, List, Dict, Union

from dir_definitions import CONFIG_RUNS_DIR
from python_code.evaluator import Evaluator
from python_code.plotters.plotter_utils import get_all_plots
from python_code.utils.config_singleton import Config

RunParams = namedtuple(
    "RunParams",
    "run_over trial_num",
    defaults=[False, 1]
)


def set_method_name(conf: Config, params_dict: Dict[str, Union[int, str]]) -> str:
    """
    Set values of params dict to current config. And return the field and their respective values as the name of the run,
    used to save as pkl file for easy access later.
    :param conf: config file.
    :param save_by_name: the desired augmentation scheme name
    :param params_dict: the run params
    :return: name of the run
    """
    name = ''
    for field, value in params_dict.items():
        conf.set_value(field, value)
        name += f'_{field}_{value}'
    return name


def gather_plots_by_trials(all_curves: List[Tuple[str, List, List, List]], conf: Config, name: str, run_over: bool,
                           trial_num: int, evaluater: Evaluator):
    """
    Run the experiments #trial_num times, averaging over the whole run's aggregated ser.
    """
    metric_outputs = []
    method_name = str(evaluater.detector) + ', ' + str(evaluater.decoder)
    for trial in range(trial_num):
        conf.set_value('seed', 1 + trial)
        evaluater.__init__()
        metric_output = get_all_plots(evaluater, run_over=run_over,
                                      save_by_name=name,
                                      trial=trial)
        metric_outputs.append(metric_output)
    all_curves.append((method_name, metric_outputs))


def compute_for_method(all_curves: List[Tuple[float, str]], params_dict: Dict[str, Union[int, str]],
                       run_params_obj: RunParams, plot_type_name: str):
    conf = Config()
    conf.load_config(os.path.join(CONFIG_RUNS_DIR, f'{plot_type_name}.yaml'))
    name = set_method_name(conf, params_dict)
    name = f'{plot_type_name}' + name
    evaluater = Evaluator()
    gather_plots_by_trials(all_curves, conf, name, run_params_obj.run_over, run_params_obj.trial_num, evaluater)
