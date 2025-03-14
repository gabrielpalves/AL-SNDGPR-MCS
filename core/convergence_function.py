import torch
from scipy.special import ndtri


def stop_Pf(g, g_mean, gs, Params):
    """Stop Pf convergence criteria"""
    N = Params.config.n_initial
    n = Params.reliability.n
    ALPHA = Params.reliability.alpha

    # Estimate the current Pf, Pf+ and Pf-
    Pf = (torch.sum(g_mean <= 0) + torch.sum(g[N+1:] <= 0))/n
    Pf_plus = (
        torch.sum(g_mean - gs*ndtri(1-ALPHA/2) <= 0) +
        torch.sum(g[N+1:] <= 0)
        )/n
    Pf_minus = (
        torch.sum(g_mean + gs*ndtri(1-ALPHA/2) <= 0) +
        torch.sum(g[N+1:] <= 0)
        )/n

    delta = 0.10

    # Check whether convergence has been reached
    if (Pf_plus - Pf_minus) / Pf <= delta:
        return True
    return False


def stop_U(g, g_mean, gs, Params):
    """Stop U convergence criteria"""
    U = torch.abs(g_mean/gs)
    min_U = torch.min(U)

    if min_U >= 2:
        return True
    
    return False
