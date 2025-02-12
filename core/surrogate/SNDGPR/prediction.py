import torch
from gpytorch.settings import use_toeplitz, fast_pred_var

def predict(Data, x_candidate):
    Data.model.eval()
    Data.likelihood.eval()
    with torch.no_grad(), use_toeplitz(False), fast_pred_var():
        return Data.model(x_candidate)
