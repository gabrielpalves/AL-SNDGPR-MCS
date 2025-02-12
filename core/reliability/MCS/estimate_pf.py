import torch
from scipy.special import ndtri


def estimate_Pf(g, g_mean, gs, Params):
    N = Params.config.n_initial
    n = Params.reliability.n
    ALPHA = Params.reliability.alpha
    
    Pf = (torch.sum(g_mean <= 0) + torch.sum(g[N+1:] <= 0))/n
    Pf_plus = (
        torch.sum(g_mean - gs*ndtri(1-ALPHA/2) <= 0) +
        torch.sum(g[N+1:] <= 0)
        )/n
    Pf_minus = (
        torch.sum(g_mean + gs*ndtri(1-ALPHA/2) <= 0) +
        torch.sum(g[N+1:] <= 0)
        )/n
    
    return Pf, Pf_plus, Pf_minus
