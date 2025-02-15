import torch
import numpy as np
import matlab.engine


def limit_state_function(x):
    """Call vectorized progressive collapse MATLAB function"""
    eng = matlab.engine.start_matlab()
    x = np.array(x)
    x = x.tolist()
    x = matlab.double(x)
    g = eng.colapsoParalelizado(x, nargout=1)
    eng.quit()

    if isinstance(g, float):
        return torch.Tensor([g])
    return torch.Tensor(g)
