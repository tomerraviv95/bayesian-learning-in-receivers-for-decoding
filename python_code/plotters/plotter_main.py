from collections import OrderedDict

from python_code.plotters.plotter_config import get_config, PlotType
from python_code.plotters.plotter_methods import compute_for_method, RunParams
from python_code.plotters.plotter_utils import plot_dict_vs_list, plot_dict_vs_dict, LEGEND_TYPE


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


## Plotter for the Paper's Figures
if __name__ == '__main__':
    run_over = False  # whether to run over previous results
    trial_num = 1  # number of trials per point estimate, used to reduce noise by averaging results of multiple runs
    plot_type = PlotType.decoding_comparison_by_code_length
    print(plot_type.name)
    run_params_obj = RunParams(run_over=run_over, trial_num=trial_num)
    params_dicts, to_plot_by_values = get_config(plot_type)
    all_curves = []

    for params_dict in params_dicts:
        print(params_dict)
        compute_for_method(all_curves, params_dict, run_params_obj, plot_type.name)

    if plot_type is PlotType.decoding_comparison_by_code_length:
        means_bers_dict = get_mean_ber_list(all_curves)
        plot_dict_vs_list(values_dict=means_bers_dict, xlabel='Code Length', ylabel='BER',
                          plot_type=plot_type, to_plot_by_values=to_plot_by_values, loc='upper right',
                          legend_type=LEGEND_TYPE.FULL, xticks=to_plot_by_values)
    elif plot_type is PlotType.ber_by_ser:
        means_bers_dict = get_mean_ber_list(all_curves)
        means_sers_dict = get_mean_ser_list(all_curves)
        plot_dict_vs_dict(values_dict=means_bers_dict, to_plot_by_values=means_sers_dict, xlabel='SER', ylabel='BER',
                          plot_type=plot_type, loc='lower right')
    elif plot_type is PlotType.ece_by_pilots_length:
        means_ece_dict = get_mean_ece_list(all_curves)
        plot_dict_vs_list(values_dict=means_ece_dict, xlabel='Number of Pilots', ylabel='ECE', plot_type=plot_type,
                          to_plot_by_values=to_plot_by_values, loc='upper right',
                          legend_type=LEGEND_TYPE.DETECTION_ONLY, xticks=to_plot_by_values)
    elif plot_type in [PlotType.detection_comparison_by_SNR_QPSK,
                       PlotType.detection_comparison_by_SNR_EightPSK,
                       PlotType.detection_comparison_by_SNR_cost_QPSK,
                       PlotType.detection_comparison_by_SNR_cost_EightPSK]:
        means_sers_dict = get_mean_ser_list(all_curves)
        plot_dict_vs_list(values_dict=means_sers_dict, xlabel='SNR [dB]', ylabel='SER', plot_type=plot_type,
                          to_plot_by_values=to_plot_by_values, legend_type=LEGEND_TYPE.DETECTION_ONLY)
    else:
        means_bers_dict = get_mean_ber_list(all_curves)
        plot_dict_vs_list(values_dict=means_bers_dict, xlabel='SNR [dB]', ylabel='BER', plot_type=plot_type,
                          to_plot_by_values=to_plot_by_values, legend_type=LEGEND_TYPE.FULL)
