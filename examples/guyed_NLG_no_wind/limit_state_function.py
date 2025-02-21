import torch
import numpy as np
import matlab.engine


def limit_state_function(x):
    """Call vectorized progressive collapse MATLAB function"""
    eng = matlab.engine.start_matlab()
    g = torch.zeros(x.shape[0])
    for i in range(x.shape[0]):
        xx = x[i]
        xx = np.array(xx)
        xx = xx.tolist()
        xx = matlab.double(xx)
        gg = eng.falha_guyed_no_wind(xx, nargout=1)
        gg = torch.Tensor([gg])
        g[i] = gg[-1]
    eng.quit()

    return g
