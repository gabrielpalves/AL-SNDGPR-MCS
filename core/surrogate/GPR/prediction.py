import torch
from gpytorch.settings import use_toeplitz, fast_pred_var

def prediction(model, likelihood, x_candidate):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), use_toeplitz(False), fast_pred_var():
        return model(x_candidate)
