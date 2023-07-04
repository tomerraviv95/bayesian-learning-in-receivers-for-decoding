from python_code.plotters.plotter_config import get_config, PlotType
from python_code.plotters.plotter_methods import compute_for_method, RunParams
from python_code.plotters.plotter_utils import plot_ber_vs_ser, plot_ber_vs_snr, plot_ser_vs_snr

## Plotter for the Paper's Figures
if __name__ == '__main__':
    run_over = False  # whether to run over previous results
    trial_num = 1  # number of trials per point estimate, used to reduce noise by averaging results of multiple runs
    plot_type = PlotType.QPSK_SNR  # Choose the plot among the three Figures
    print(plot_type.name)
    run_params_obj = RunParams(run_over=run_over, trial_num=trial_num)
    params_dicts, xlabel, ylabel, to_plot_by_values = get_config(plot_type)
    all_curves = []

    for params_dict in params_dicts:
        print(params_dict)
        compute_for_method(all_curves, params_dict, run_params_obj, plot_type.name)

    if plot_type in [PlotType.QPSK_SNR]:
        plot_ber_vs_snr(all_curves=all_curves, xlabel=xlabel, ylabel='BER', plot_type=plot_type,
                        to_plot_by_values=to_plot_by_values)
        plot_ser_vs_snr(all_curves=all_curves, xlabel=xlabel, ylabel='SER', plot_type=plot_type,
                        to_plot_by_values=to_plot_by_values)
    elif plot_type in [PlotType.QPSK_BER_VS_SER]:
        plot_ber_vs_ser(all_curves, xlabel, ylabel, plot_type, to_plot_by_values)
    else:
        raise ValueError("No such graph type!!!")
