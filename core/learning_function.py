import torch


def U(g_mean, g_sigma):
    """U function"""
    return -torch.abs(g_mean / g_sigma)

def evaluate_lf(preds, learning_function):
    """Evaluate learning function for all predicted points"""
    g_mean = preds.mean
    gs2 = preds.variance
    gs = torch.sqrt(gs2)
    lf = learning_function(g_mean, gs)
    
    return g_mean, gs, torch.argmax(lf)
