import pickle
import os
from pathlib import Path
import torch
import numpy as np
import scipy.io as sio
from core.hyper_params_opt.optimization_variables import optimization_variables


# Functions related to saving and loading data
def save_load_initial(EXAMPLE, Data, Params, limit_state_function, hyper_params_opt):
    folder_path = os.path.join("examples", EXAMPLE, "data", "initial")
    file = os.path.join(folder_path, 'Data.pkl')
    if os.path.exists(file):
        with open(file, 'rb') as file:
            loaded_Data = pickle.load(file)
            
            # Check hyperparameters optimization
            if not loaded_Data.f_opt:
                # Check initial sampling plan evaluation
                if not loaded_Data.g:
                    Data.g = limit_state_function(Data.x)
                    with open(file, 'wb') as file: pickle.dump(Data, file)
                else:
                    Data = loaded_Data
                    
                Data = hyper_params_opt(Data, Params)
                Data.model, Data.likelihood, Data.act_fun = None, None, None
                with open(file, 'wb') as file: pickle.dump(Data, file)
    else:
        os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
        Data.g = limit_state_function(Data.x)
        with open(file, 'wb') as file: pickle.dump(Data, file)
        
        Data = hyper_params_opt(Data, Params)
        Data.model, Data.likelihood, Data.act_fun = None, None, None
        with open(file, 'wb') as file: pickle.dump(Data, file)

    return Data


def pickle_save(data_dict, EXAMPLE):
    folder_path = os.path.join("examples", EXAMPLE, "data")
    os.makedirs(folder_path, exist_ok=True)
    for key, data in data_dict.items():
        file_path = os.path.join(folder_path, f'{key}.pkl')
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)


def pickle_load(EXAMPLE, keys):
    folder_path = os.path.join("examples", EXAMPLE, "data")
    loaded_data = {}
    for key in keys:
        file_path = os.path.join(folder_path, f'{key}.pkl')
        with open(file_path, 'rb') as file:
            loaded_data[key] = pickle.load(file)
    return loaded_data


def save_bests(it, Data, Params, EXAMPLE):
    # Define the folder path for saving files
    folder_path = os.path.join("examples", EXAMPLE, "data", "best_models")
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    # Save the model state using torch
    torch_model_likelihood_path = os.path.join(folder_path, f'best_model_and_likelihood_{it}.pth')
    torch.save({
        'model_state_dict': Data.model.state_dict(),
        'likelihood_state_dict': Data.likelihood.state_dict(),
    }, torch_model_likelihood_path)

    Data = optimization_variables(Data, Params, get_best=True)

    folder_path = os.path.join("examples", EXAMPLE, "data", "variables")
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    # Save the .mat file inside the example's data folder
    file_name = 'variables_best_' + Data.act_fun.__name__ \
        + '_SN_' + str(Params.surrogate.spectral_normalization) \
        + str(it) + '.mat'
    full_path = os.path.join(folder_path, file_name)

    sio.savemat(full_path, {
        'training_losses': np.array(Data.train_losses),
        'validation_losses': np.array(Data.val_losses),
        'x_opt': Data.x_opt.numpy(),
        'f_opt': Data.f_opt.numpy(),
        'layer_sizes': np.array(Data.layer_sizes),
        'train_x': np.array(Data.train_x),
        'val_x': np.array(Data.val_x),
        'train_g': np.array(Data.train_g),
        'val_g': np.array(Data.val_g),
        'x_max': np.array(Data.x_max),
        'x_min': np.array(Data.x_min),
        })
