import numpy as np
from torch.nn import ReLU, ELU, Tanh, Sigmoid, GELU
from torch import argmin

def optimization_variables(Data, Params):
    x_opt = Data.x_opt
    if x_opt.shape[0] > 1:
        x_opt = x_opt[argmin(Data.f_opt), :]

    L = x_opt[0]
    r = x_opt[1]
    act_fun = 0
    if x_opt.shape[0] > 2:
        act_fun = x_opt[2]
    
    bounds = Params.optimization.bounds_opt
        
    # BOUNDS
    L = L*(bounds[0][1] - bounds[0][0]) + bounds[0][0]
    r = r*(bounds[1][1] - bounds[1][0]) + bounds[1][0]
    act_fun = act_fun*(bounds[2][1] - bounds[2][0]) + bounds[2][0]
    
    r = int(np.round(r))
    L = int(np.round(L))
    act_fun = int(np.round(act_fun))
    
    act_fun_list = [
        ReLU,
        ELU,
        Tanh,
        GELU,
        Sigmoid
    ]
    
    act_fun = act_fun_list[act_fun]

    # Randomly sample hyperparameters
    layer_sizes = []
    D = Data.x.shape[1]  # dimension of the problem
    rho = np.log(r/D) / L
    for i in range(1, L+1):
        layer_sizes.append(int(D * np.exp(rho * i)))  # Eq. 20
    
    # unique values
    myset = set(layer_sizes)
    layer_sizes = list(myset)
    
    # decreasing number of neurons per layer in architecture
    layer_sizes.sort(reverse=True)

    return layer_sizes, act_fun
