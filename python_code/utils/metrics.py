from typing import List, Tuple

import numpy as np
import torch

SENSITIVITY = 1e-3


def calculate_error_rate(prediction: torch.Tensor, target: torch.Tensor) -> Tuple[float, int]:
    """
    Returns the calculated ber of the prediction and the target (ground truth transmitted word)
    """
    prediction = prediction.long()
    target = target.long()
    equal_bits = torch.eq(prediction, target).float()
    bits_acc = torch.mean(equal_bits).item()
    non_equal_bits = 1 - equal_bits
    errors = int(torch.sum(non_equal_bits).item())
    return 1 - bits_acc, errors


def calculate_reliability_and_ece(correct_values_list: List[float], error_values_list: List[float],
                                  values: List[float]) -> Tuple[float, List[float], List[float]]:
    """
    Input is two lists, of the correctly detected and incorrectly detected confidence values
    Computes the two lists of accuracy and confidences (red and blue bar plots in paper), the ECE measure and the
    normalized frequency count per bin (green bar plot in paper)
    """
    correct_values_array = np.array(correct_values_list)
    error_values_array = np.array(error_values_list)
    avg_confidence_per_bin, avg_acc_per_bin, inbetween_indices_number_list = [], [], []
    total_values = len(correct_values_array) + len(error_values_array)
    # calculate the mean accuracy and mean confidence for the given range
    for val_j, val_j_plus_1 in zip(values[:-1], values[1:]):
        avg_confidence_value_in_bin, avg_acc_value_in_bin = 0, 0
        inbetween_correct_indices = np.logical_and(val_j <= correct_values_array,
                                                   correct_values_array <= val_j_plus_1)
        inbetween_errored_indices = np.logical_and(val_j <= error_values_array, error_values_array <= val_j_plus_1)
        inbetween_indices_number = inbetween_correct_indices.sum() + inbetween_errored_indices.sum()
        if total_values * SENSITIVITY < inbetween_indices_number:
            correct_values = correct_values_array[inbetween_correct_indices]
            errored_values = error_values_array[inbetween_errored_indices]
            avg_acc_value_in_bin = len(correct_values) / (len(correct_values) + len(errored_values))
            avg_confidence_value_in_bin = np.mean(np.concatenate([correct_values, errored_values]))
        avg_acc_per_bin.append(avg_acc_value_in_bin)
        avg_confidence_per_bin.append(avg_confidence_value_in_bin)
        inbetween_indices_number_list.append(inbetween_indices_number)
    # calculate ECE
    confidence_acc_diff = np.abs(np.array(avg_confidence_per_bin) - np.array(avg_acc_per_bin))
    ece_measure = np.sum(np.array(inbetween_indices_number_list) * confidence_acc_diff) / sum(
        inbetween_indices_number_list)
    return ece_measure, avg_acc_per_bin, avg_confidence_per_bin
