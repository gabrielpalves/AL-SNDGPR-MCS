import itertools
import torch
from core.hyper_params_opt.objective_function import obj_fun
from core.utils.sampling_utils import keep_best
from core.configs import RuntimeData


def grid_search(Data, Params):  # Data, Params
    bounds = Params.optimization.bounds_opt

    # Generate possible values for each variable based on bounds
    possibilities = [torch.linspace(0, 1, steps=ub+1-lb).tolist() for lb, ub in bounds]

    # Compute the Cartesian product to generate all combinations
    grid = list(itertools.product(*possibilities))

    # Convert to a Torch tensor (optional)
    x_opt = torch.tensor(grid, dtype=torch.float32)

    OptData = RuntimeData(
        x=Data.x,
        g=Data.g
    )

    f_opt, OptData = obj_fun(x_opt, OptData, Params)

    Data = keep_best(Data, OptData)
    Data.x_opt, Data.f_opt = x_opt, f_opt

    return Data
