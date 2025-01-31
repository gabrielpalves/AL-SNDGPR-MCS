import torch


def stop_Pf(g, g_mean, gs, N, MC_sample_size):
    """Stop Pf convergence criteria"""
    # Estimate the current Pf, Pf+ and Pf-
    Pf = (torch.sum(g_mean <= 0) + torch.sum(g[N+1:] <= 0)) / MC_sample_size
    Pf_plus = (
        torch.sum(g_mean-2*gs <= 0) + torch.sum(g[N+1:] <= 0)
        ) / MC_sample_size
    Pf_minus = (
        torch.sum(g_mean+2*gs <= 0) + torch.sum(g[N+1:] <= 0)
        ) / MC_sample_size

    delta = 0.10

    # Check whether convergence has been reached
    if (Pf_plus - Pf_minus) / Pf <= delta:
        return True
    return False


def stop_U(g, g_mean, gs, N, MC_sample_size):
    """Stop U convergence criteria"""
    U = torch.abs(g_mean/gs)
    min_U = torch.min(U)
    print(f'min U: {min_U:.3f}')
    if min_U >= 2:
        return True
    
    return False
