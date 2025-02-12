import torch
import numpy as np
from scipy.special import erf
from gpytorch.settings import use_toeplitz, fast_pred_var


def expected_improvement(x, model):
    """
    Function to calculate the Expected Improvement using Monte Carlo Integration.
    
    Parameters:
        x (array): Individual under evaluation.
        y (array): Valor mínimo obtido até o momento.
    
    Returns:
        array: The value of the Expected Improvement.
    """
    # Get the minimum value obtained so far
    ymin = torch.min(model.train_outputs)
    # ymin = torch.Tensor(ymin)

    # Calculate the prediction value and the variance (Ssqr)
    with torch.no_grad(), use_toeplitz(False), fast_pred_var():
        preds = model(x)
    f = preds.mean
    s = preds.variance

    # Check for any errors that are less than zero (due to numerical error)
    s[s < 0] = 0  # Set negative variances to zero

    # Calculate the RMSE (Root Mean Square Error)
    s = torch.sqrt(s)

    # Calculation of Expected Improvement
    term1 = (ymin - f) * (0.5 + 0.5 * erf((ymin - f) / (s * torch.sqrt( torch.from_numpy(np.array([2])) ))))
    term2 = (1 / torch.sqrt(2 * torch.from_numpy(np.array([np.pi])))) * s * torch.exp(-((ymin - f) ** 2) / (2 * s ** 2))
    
    return -(term1 + term2)
