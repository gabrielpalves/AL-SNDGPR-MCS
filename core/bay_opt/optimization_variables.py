import numpy as np
from torch.nn import ReLU, ELU, Tanh, Sigmoid, GELU


def optimization_variables(BOUNDS_BAY_OPT, X, x, SPECTRAL_NORMALIZATION=True):
    """
    BOUNDS_BAY_OPT -> bounds for bayesian optimization (EGO)
    X -> variables from EGO (hyperparameters optimization)
    x -> sampling plan of EGRA (AL-SNDGPR-MCS)
    """
    L, r, act_fun = X[:len(BOUNDS_BAY_OPT)]
        
    # BOUNDS
    L = L*(BOUNDS_BAY_OPT[0][1] - BOUNDS_BAY_OPT[0][0]) + BOUNDS_BAY_OPT[0][0]
    r = r*(BOUNDS_BAY_OPT[1][1] - BOUNDS_BAY_OPT[1][0]) + BOUNDS_BAY_OPT[1][0]
    act_fun = act_fun*(BOUNDS_BAY_OPT[2][1] - BOUNDS_BAY_OPT[2][0]) + BOUNDS_BAY_OPT[2][0]
    
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
    D = x.shape[1]  # dimension of the problem
    rho = np.log(r/D) / L
    for i in range(1, L+1):
        layer_sizes.append(int(D * np.exp(rho * i)))  # Eq. 20
    
    # unique values
    myset = set(layer_sizes)
    layer_sizes = list(myset)
    
    # decreasing number of neurons per layer in architecture
    layer_sizes.sort(reverse=True)

    return layer_sizes, act_fun
