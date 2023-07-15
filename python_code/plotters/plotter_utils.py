import datetime
import os
from collections import OrderedDict
from typing import List, Tuple

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


def get_mean_ser_list(all_curves):
    values_to_plot_dict = OrderedDict()
    for method_name, metric_outputs in all_curves:
        if method_name not in values_to_plot_dict.keys():
            values_to_plot_dict[method_name] = []
        current_ser_list = []
        for metric_output in metric_outputs:
            current_ser_list.extend(metric_output.ser_list)
        values_to_plot_dict[method_name].append(sum(current_ser_list) / len(current_ser_list))
    return values_to_plot_dict


def get_mean_ber_list(all_curves):
    values_to_plot_dict = OrderedDict()
    for method_name, metric_outputs in all_curves:
        if method_name not in values_to_plot_dict.keys():
            values_to_plot_dict[method_name] = []
        current_ber_list = []
        for metric_output in metric_outputs:
            current_ber_list.extend(metric_output.ber_list)
        values_to_plot_dict[method_name].append(sum(current_ber_list) / len(current_ber_list))
    return values_to_plot_dict


def get_mean_ece_list(all_curves):
    values_to_plot_dict = OrderedDict()
    for method_name, metric_outputs in all_curves:
        if method_name not in values_to_plot_dict.keys():
            values_to_plot_dict[method_name] = []
        current_ece_list = []
        for metric_output in metric_outputs:
            current_ece_list.extend(metric_output.ece_list)
        values_to_plot_dict[method_name].append(sum(current_ece_list) / len(current_ece_list))
    return values_to_plot_dict


def plot_by_ber(all_curves: List[Tuple[np.ndarray, np.ndarray, str]], xlabel: str, ylabel: str, plot_type: PlotType,
                to_plot_by_values: List[int], loc='lower left'):
    # path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))

    # extract names from simulated plots
    plt.figure()
    means_bers_dict = get_mean_ber_list(all_curves)

    # plots all methods
    print("Plotting BER")
    for method_name in means_bers_dict.keys():
        print(method_name)
        plt.plot(to_plot_by_values, means_bers_dict[method_name],
                 label=method_name.replace(', ', '/'),
                 color=get_color(method_name),
                 marker=get_marker(method_name), markersize=11,
                 linestyle=get_linestyle(method_name), linewidth=2.2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(which='both', ls='--')
    plt.legend(loc=loc, prop={'size': 18})
    plt.yscale('log')
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, f'ber_versus_{xlabel}_{plot_type.name}.png'),
                bbox_inches='tight')


def plot_by_ser(all_curves: List[Tuple[np.ndarray, np.ndarray, str]], xlabel: str, ylabel: str, plot_type: PlotType,
                to_plot_by_values: List[int]):
    # path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))

    # extract names from simulated plots
    plt.figure()
    means_sers_dict = get_mean_ser_list(all_curves)

    # plots all methods
    print("Plotting SER")
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
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, f'ser_versus_{xlabel}_{plot_type.name}.png'),
                bbox_inches='tight')


def plot_by_ece(all_curves: List[Tuple[np.ndarray, np.ndarray, str]], xlabel: str, ylabel: str, plot_type: PlotType,
                to_plot_by_values: List[int], loc='lower left'):
    # path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))

    # extract names from simulated plots
    plt.figure()
    means_ece_dict = get_mean_ece_list(all_curves)
    means_bers_dict = get_mean_ber_list(all_curves)

    # plots all methods
    print("Plotting BER")
    for method_name in means_ece_dict.keys():
        print(method_name)
        plt.plot(means_ece_dict[method_name], means_bers_dict[method_name],
                 label=method_name.replace(', ', '/'),
                 color=get_color(method_name),
                 marker=get_marker(method_name), markersize=11,
                 linestyle=get_linestyle(method_name), linewidth=2.2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(which='both', ls='--')
    plt.legend(loc=loc, prop={'size': 18})
    plt.yscale('log')
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, f'ber_versus_{xlabel}_{plot_type.name}.png'),
                bbox_inches='tight')
