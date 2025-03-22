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
    
    numpy_array = x_added.detach().cpu().numpy()
    folder_path = os.path.join("examples", EXAMPLE, "data", "sampling_plan")
    os.makedirs(folder_path, exist_ok=True)
    full_path = os.path.join(folder_path, f'x{it}.mat')
    sio.savemat(full_path, {'x': numpy_array})
    
    return x_added

def evaluate_g(x_added, it, limit_state_function, EXAMPLE):
    # Evaluate limit state function
    folder_path = os.path.join("examples", EXAMPLE, "data", "sampling_plan")
    file_name = f'g{it-1}.mat'
    full_path = os.path.join(folder_path, file_name)
    if os.path.isfile(full_path):
        mat_contents = sio.loadmat(full_path)
        g_added = torch.Tensor(mat_contents['g']).view(-1)
        if torch.cuda.is_available():
            g_added = g_added.cuda()
    else:
        g_added = torch.Tensor(limit_state_function(x_added))
        if torch.cuda.is_available():
            g_added = g_added.cuda()
        sio.savemat(full_path, {'g': g_added.detach().cpu().numpy()})
    return g_added

def keep_best(Data, OptData):
    Data.model, Data.likelihood = OptData.model, OptData.likelihood
    Data.train_losses, Data.val_losses = OptData.train_losses, OptData.val_losses
    Data.train_x, Data.val_x = OptData.train_x, OptData.val_x
    Data.train_g, Data.val_g = OptData.train_g, OptData.val_g
    Data.x_max, Data.x_min = OptData.x_max, OptData.x_min
    return Data
