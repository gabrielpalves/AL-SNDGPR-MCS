import torch


def limit_state_function(x):
    y = 86 \
        - ((torch.square(x[:, 0]) + 4) * (x[:, 1] - 1))/20 \
        + torch.cos(5*x[:, 0]) \
        - torch.sum(torch.square(x), dim=1)
    return y
