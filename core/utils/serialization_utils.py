import pickle
import os
import torch

# Functions related to saving and loading data using pickle
def pickle_save(data_dict, example):
    folder_path = os.path.join(example, "data")
    os.makedirs(folder_path, exist_ok=True)
    for key, data in data_dict.items():
        file_path = os.path.join(folder_path, f'{key}.pkl')
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

def pickle_load(example, keys):
    folder_path = os.path.join(example, "data")
    loaded_data = {}
    for key in keys:
        file_path = os.path.join(folder_path, f'{key}.pkl')
        with open(file_path, 'rb') as file:
            loaded_data[key] = pickle.load(file)
    return loaded_data

def save_bests(best_model, best_likelihood, it, EXAMPLE):
    folder_path = os.path.join(EXAMPLE, "data/best_models")
    os.makedirs(folder_path, exist_ok=True)
    torch_model_path = os.path.join(folder_path, f'best_model_and_likelihood_{it}.pth')
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'likelihood_state_dict': best_likelihood.state_dict(),
    }, torch_model_path)