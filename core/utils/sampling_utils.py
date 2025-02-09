import os
import torch
import scipy.io as sio
import numpy as np


# Functions related to sampling and evaluation
def min_max_normalization(x_max, x_min, x):
    return (x - x_min) / (x_max - x_min)

def add_x(x_candidate, ind_lf, it, EXAMPLE):
    # Select additional sample (the sample which maximizes the learning function value)
    data_dim = x_candidate.shape[1]
    x_added = x_candidate[ind_lf, :]
    x_added = x_added.view(1, data_dim)
    # x_added = x_added * (x_max - x_min) + x_min  # undo normalization
    
    numpy_array = x_added.numpy()
    folder_path = os.path.join(EXAMPLE, "data/sampling_plan")
    os.makedirs(folder_path, exist_ok=True)
    full_path = os.path.join(folder_path, f'x{it}.mat')
    sio.savemat(full_path, {'x': numpy_array})
    
    return x_added

def evaluate_g(x_added, it, limit_state_function, EXAMPLE):
    folder_path = os.path.join(EXAMPLE, "data/sampling_plan")
    file_name = f'g{it-1}.mat'
    full_path = os.path.join(folder_path, file_name)
    if os.path.isfile(full_path):
        mat_contents = sio.loadmat(full_path)
        g_added = torch.Tensor(mat_contents['g']).view(-1)
    else:
        g_added = torch.Tensor(limit_state_function(x_added))
        sio.savemat(full_path, {'g': g_added.numpy()})
    return g_added
