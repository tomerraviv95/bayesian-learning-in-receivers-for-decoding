import datetime
import os
from enum import Enum
from typing import List, Tuple, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from dir_definitions import FIGURES_DIR, PLOTS_DIR
from python_code.evaluator import Evaluator
from python_code.plotters.plotter_config import PlotType
from python_code.utils.python_utils import load_pkl, save_pkl

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['font.size'] = 8
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = [9.5, 6.45]
mpl.rcParams['axes.titlesize'] = 28
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'


def get_linestyle(method_name: str) -> str:
    if 'MB-D' in method_name:
        return 'solid'
    elif 'B-D' in method_name:
        return 'dashed'
    elif 'F-D' in method_name:
        return 'dotted'
    else:
        raise ValueError('No such detector!!!')


def get_color(method_name: str) -> str:
    if 'MB-D' in method_name:
        return 'blue'
    elif 'B-D' in method_name:
        return 'black'
    elif 'F-D' in method_name:
        return 'red'
    else:
        raise ValueError('No such method!!!')


def get_marker(method_name: str) -> str:
    if 'MB-W' in method_name:
        return 'o'
    elif 'B-W' in method_name:
        return 'X'
    elif 'F-W' in method_name:
        return 's'
    else:
        raise ValueError('No such method!!!')


class LEGEND_TYPE(Enum):
    FULL = 'FULL'
    DETECTION_ONLY = 'DETECTION_ONLY'


def get_all_plots(dec: Evaluator, run_over: bool, save_by_name: str, trial=None) -> Tuple[List[float], List[float]]:
    print(save_by_name)
    # set the path to saved plot results for a single method (so we do not need to run anew each time)
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    file_name = save_by_name
    if trial is not None:
        file_name = file_name + '_' + str(trial)
    plots_path = os.path.join(PLOTS_DIR, file_name)
    print(plots_path)
    # if plot already exists, and the run_over flag is false - load the saved plot
    if os.path.isfile(plots_path + '_ber.pkl') and not run_over:
        print("Loading plots")
        metric_output = load_pkl(plots_path, type='ber')
    else:
        # otherwise - run again
        print("Calculating fresh")
        metric_output = dec.evaluate()
        save_pkl(plots_path, metric_output, type='ber')
    return metric_output


def plot_dict_vs_list(values_dict: Dict[str, List[float]], to_plot_by_values: List[int], xlabel: str, ylabel: str,
                      plot_type: PlotType, legend_type: LEGEND_TYPE, xticks: List[int] = None, loc='lower left'):
    # path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))

    # extract names from simulated plots
    plt.figure()
    # plots all methods
    print("Plotting")
    for method_name in values_dict.keys():
        print(method_name)
        if legend_type is LEGEND_TYPE.FULL:
            label = method_name.replace(', ', '/').replace('-DeepSIC', '').replace('-WBP', '')
        elif legend_type is LEGEND_TYPE.DETECTION_ONLY:
            label = method_name.split(',')[0].replace('-DeepSIC', '')
        else:
            label = ''
        plt.plot(to_plot_by_values, values_dict[method_name],
                 label=label,
                 color=get_color(method_name),
                 marker=get_marker(method_name), markersize=11,
                 linestyle=get_linestyle(method_name), linewidth=2.2)

    if xticks is not None:
        plt.xticks(to_plot_by_values, xticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(which='both', ls='--')
    plt.legend(loc=loc, prop={'size': 18})
    plt.yscale('log')
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, f'{ylabel}_versus_{xlabel}_{plot_type.name}.png'),
                bbox_inches='tight')


def plot_dict_vs_dict(values_dict: Dict[str, List[float]], to_plot_by_values: Dict[str, List[float]], xlabel: str,
                      ylabel: str, plot_type: PlotType, loc='lower left'):
    # path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))

    # extract names from simulated plots
    plt.figure()

    # plots all methods
    print("Plotting")
    for method_name in values_dict.keys():
        print(method_name)
        plt.scatter(to_plot_by_values[method_name], values_dict[method_name],
                    label=method_name.replace(', ', '/').replace('-DeepSIC', '').replace('-WBP', ''),
                    color=get_color(method_name))
        plt.plot(np.linspace(0, 1, 10000), np.mean(values_dict[method_name]) * np.ones(10000),
                 color=get_color(method_name), linestyle='--')
        plt.text(0.1, np.mean(values_dict[method_name]) + 0.3e-3, "average",
                 verticalalignment='center', size=24)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(which='both', ls='--')
    plt.legend(loc=loc, prop={'size': 18})
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, f'{ylabel}_versus_{xlabel}_{plot_type.name}.png'),
                bbox_inches='tight')
