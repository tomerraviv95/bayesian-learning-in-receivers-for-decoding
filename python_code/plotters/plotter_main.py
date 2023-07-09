from python_code.plotters.plotter_config import get_config, PlotType
from python_code.plotters.plotter_methods import compute_for_method, RunParams
from python_code.plotters.plotter_utils import plot_by_ber, plot_by_ser

## Plotter for the Paper's Figures
if __name__ == '__main__':
    run_over = False  # whether to run over previous results
    trial_num = 1  # number of trials per point estimate, used to reduce noise by averaging results of multiple runs
    # for plot_type in [PlotType.final_comparison_by_SNR,
    #                   PlotType.detection_comparison_by_SNR,
    #                   PlotType.decoding_comparison_by_SNR,
    #                   PlotType.final_comparison_by_users]:
    plot_type = PlotType.final_comparison_by_SNR
    print(plot_type.name)
    run_params_obj = RunParams(run_over=run_over, trial_num=trial_num)
    params_dicts, xlabel, ylabel, to_plot_by_values = get_config(plot_type)
    all_curves = []

    for params_dict in params_dicts:
        print(params_dict)
        compute_for_method(all_curves, params_dict, run_params_obj, plot_type.name)
    if plot_type is not PlotType.final_comparison_by_users:
        plot_by_ber(all_curves=all_curves, xlabel=xlabel, ylabel='BER', plot_type=plot_type,
                    to_plot_by_values=to_plot_by_values)
        plot_by_ser(all_curves=all_curves, xlabel=xlabel, ylabel='SER', plot_type=plot_type,
                    to_plot_by_values=to_plot_by_values)
    else:
        plot_by_ber(all_curves=all_curves, xlabel=xlabel, ylabel='BER', plot_type=plot_type,
                    to_plot_by_values=to_plot_by_values)
