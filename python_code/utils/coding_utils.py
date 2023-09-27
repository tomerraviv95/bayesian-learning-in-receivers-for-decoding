import os
from enum import Enum

import numpy as np


class CODE_TYPE(Enum):
    POLAR = 'POLAR'
    BCH = 'BCH'


def row_reduce(mat, ncols=None):
    assert mat.ndim == 2
    ncols = mat.shape[1] if ncols is None else ncols
    mat_row_reduced = mat.copy()
    p = 0
    for j in range(ncols):
        idxs = p + np.nonzero(mat_row_reduced[p:, j])[0]
        if idxs.size == 0:
            continue
        mat_row_reduced[[p, idxs[0]], :] = mat_row_reduced[[idxs[0], p], :]
        idxs = np.nonzero(mat_row_reduced[:, j])[0].tolist()
        idxs.remove(p)
        mat_row_reduced[idxs, :] = mat_row_reduced[idxs, :] ^ mat_row_reduced[p, :]
        p += 1
        if p == mat_row_reduced.shape[0]:
            break
    return mat_row_reduced, p


def get_generator(pc_matrix_):
    assert pc_matrix_.ndim == 2
    pc_matrix = pc_matrix_.copy().astype(bool).transpose()
    pc_matrix_I = np.concatenate((pc_matrix, np.eye(pc_matrix.shape[0], dtype=bool)), axis=-1)
    pc_matrix_I, p = row_reduce(pc_matrix_I, ncols=pc_matrix.shape[1])
    return row_reduce(pc_matrix_I[p:, pc_matrix.shape[1]:])[0]


def get_standard_form(pc_matrix):
    pc_matrix = pc_matrix.astype(bool)
    next_col = min(pc_matrix.shape)
    for ii in range(min(pc_matrix.shape)):
        while True:
            rows_ones = ii + np.where(pc_matrix[ii:, ii])[0]
            if len(rows_ones) == 0:
                new_shift = np.arange(ii, min(pc_matrix.shape) - 1).tolist() + [min(pc_matrix.shape) - 1, next_col]
                old_shift = np.arange(ii + 1, min(pc_matrix.shape)).tolist() + [next_col, ii]
                pc_matrix[:, new_shift] = pc_matrix[:, old_shift]
                next_col += 1
            else:
                break
        pc_matrix[[ii, rows_ones[0]], :] = pc_matrix[[rows_ones[0], ii], :]
        other_rows = pc_matrix[:, ii].copy()
        other_rows[ii] = False
        pc_matrix[other_rows] = pc_matrix[other_rows] ^ pc_matrix[ii]
    return pc_matrix.astype(int)


def get_code_pcm_and_gm(bits_num, message_bits_num, ecc_mat_path, code_type):
    pc_matrix_path = os.path.join(ecc_mat_path, f'{code_type}_{bits_num}_{message_bits_num}')
    if code_type in [CODE_TYPE.POLAR.name, CODE_TYPE.BCH.name]:
        code_pcm = np.loadtxt(pc_matrix_path + '.txt')
    else:
        raise Exception(f'Code of type {code_type} is not supported!!!')
    code_gm = get_generator(code_pcm)
    assert np.all(np.mod((np.matmul(code_gm, code_pcm.transpose())), 2) == 0) and np.sum(code_gm) > 0
    return code_pcm.astype(np.float32), code_gm.astype(np.float32)
