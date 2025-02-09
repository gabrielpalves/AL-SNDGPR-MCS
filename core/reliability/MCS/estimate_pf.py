import torch
from scipy.special import ndtri


def estimate_Pf(g, g_mean, gs, N, N_MC, ALPHA):
    Pf = (torch.sum(g_mean <= 0) + torch.sum(g[N+1:] <= 0))/N_MC
    Pf_plus = (
        torch.sum(g_mean - gs*ndtri(1-ALPHA/2) <= 0) +
        torch.sum(g[N+1:] <= 0)
        )/N_MC
    Pf_minus = (
        torch.sum(g_mean + gs*ndtri(1-ALPHA/2) <= 0) +
        torch.sum(g[N+1:] <= 0)
        )/N_MC
    
    return Pf, Pf_plus, Pf_minus
