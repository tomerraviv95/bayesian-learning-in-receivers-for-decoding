import datetime
import os
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
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'


def get_linestyle(method_name: str) -> str:
    if 'Model-Based Bayesian' in method_name:
        return 'solid'
    elif 'Bayesian DeepSIC' in method_name:
        return 'dashed'
    elif 'DeepSIC' in method_name:
        return 'dotted'
    elif 'Bayesian DNN' in method_name:
        return '-.'
    elif 'DNN' in method_name:
        return '-.'
    else:
        raise ValueError('No such detector!!!')


def get_marker(method_name: str) -> str:
    if 'Model-Based Bayesian' in method_name:
        return 'o'
    elif 'Bayesian DeepSIC' in method_name:
        return 'X'
    elif 'DeepSIC' in method_name:
        return 's'
    elif 'Bayesian DNN' in method_name:
        return 'p'
    elif 'DNN' in method_name:
        return 'p'
    else:
        raise ValueError('No such method!!!')


def get_color(method_name: str) -> str:
    if 'Model-Based Bayesian' in method_name:
        return 'blue'
    elif 'Bayesian DeepSIC' in method_name:
        return 'black'
    elif 'DeepSIC' in method_name:
        return 'red'
    elif 'Bayesian DNN' in method_name:
        return 'purple'
    elif 'DNN' in method_name:
        return 'green'
    else:
        raise ValueError('No such method!!!')


def get_all_plots(dec: Evaluator, run_over: bool, method_name: str, trial=None) -> Tuple[List[float], List[float]]:
    print(method_name)
    # set the path to saved plot results for a single method (so we do not need to run anew each time)
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    file_name = method_name
    if trial is not None:
        file_name = file_name + '_' + str(trial)
    plots_path = os.path.join(PLOTS_DIR, file_name)
    print(plots_path)
    # if plot already exists, and the run_over flag is false - load the saved plot
    if os.path.isfile(plots_path + '_ber.pkl') and not run_over:
        print("Loading plots")
        ber_total = load_pkl(plots_path, type='ber')
    else:
        # otherwise - run again
        print("Calculating fresh")
        ber_total = dec.evaluate()
        save_pkl(plots_path, ber_total, type='ber')
    return ber_total


def get_mean_decoding_sers(all_curves):
    values_to_plot_dict = {method_name: [] for method_name in set([curve[0] for curve in all_curves])}
    for method_name, ser in all_curves:
        current_decoded_sers = ser[0].decoding_bers
        values_to_plot_dict[method_name].append(sum(current_decoded_sers) / len(current_decoded_sers))
    return values_to_plot_dict


def plot_by_snr(all_curves: List[Tuple[np.ndarray, np.ndarray, str]], xlabel: str, ylabel: str, plot_type: PlotType,
                to_plot_by_values: List[int]):
    # path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))

    # extract names from simulated plots
    plt.figure()
    means_sers_dict = get_mean_decoding_sers(all_curves)

    # plots all methods
    for method_name, sers in means_sers_dict.items():
        print(method_name)
        plt.plot(to_plot_by_values, means_sers_dict[method_name],
                 label=method_name,
                 color=get_color(method_name),
                 marker=get_marker(method_name), markersize=11,
                 linestyle=get_linestyle(method_name), linewidth=2.2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(which='both', ls='--')
    plt.legend(loc='lower left', prop={'size': 18})
    plt.yscale('log')
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, f'ser_versus_snrs_{plot_type}.png'), bbox_inches='tight')


def plot_by_det_ber(all_curves: List[Tuple[np.ndarray, np.ndarray, str]], xlabel: str, ylabel: str,
                    plot_type: PlotType, to_plot_by_values: List[int]):
    # path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))

    # extract names from simulated plots
    plt.figure()
    sers_dict = get_all_sers_dict(all_curves)

    # plots all methods
    for method_name, sers in sers_dict.items():
        print(method_name)
        plt.scatter(sers['detection_bers'], sers['decoding_bers'],
                    label=method_name,
                    color=get_color(method_name),
                    marker=get_marker(method_name),
                    linestyle=get_linestyle(method_name))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(which='both', ls='--')
    plt.legend(loc='upper left', prop={'size': 18})
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, f'ser_versus_snrs_{plot_type}.png'), bbox_inches='tight')


def get_all_sers_dict(all_curves: List[Tuple[float, str]]) -> Tuple[
    str, Dict[str, List[np.ndarray]]]:
    values_to_plot_dict = {method_name: {'detection_bers': [], 'decoding_bers': []} for method_name in
                           set([curve[0] for curve in all_curves])}
    for method_name, ser in all_curves:
        non_zero_idxs = np.nonzero(np.array(ser[0].decoding_bers))[0]
        values_to_plot_dict[method_name]['detection_bers'].extend(list(np.array(ser[0].detection_bers)[non_zero_idxs]))
        values_to_plot_dict[method_name]['decoding_bers'].extend(list(np.array(ser[0].decoding_bers)[non_zero_idxs]))
    return values_to_plot_dict
