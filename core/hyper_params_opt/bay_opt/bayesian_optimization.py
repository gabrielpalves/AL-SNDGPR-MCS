import torch
from sklearn.model_selection import train_test_split

from core.initial_sampling_plan import LHS
from core.hyper_params_opt.objective_function import obj_fun
from core.hyper_params_opt.bay_opt.bsa import bsa
from core.hyper_params_opt.bay_opt.expected_improvement import expected_improvement
from core.hyper_params_opt.optimization_variables import optimization_variables
from core.configs import RuntimeData
from core.surrogate.GPR.train import train_model as train_model_EGO
from core.utils.sampling_utils import keep_best

# -------------------- Helper Functions --------------------
def update_model(x_EGO, f_EGO, data, params):
    """
    Trains or updates the surrogate model using the current sampling data.

    Args:
        x_EGO (torch.Tensor): Sampling points.
        f_EGO (torch.Tensor): Objective function evaluations at sampling points.
        data: Runtime data.
        params: Optimization parameters.

    Returns:
        Tuple: The trained model, likelihood, and training loss.
    """
    train_x, val_x, train_g, val_g = train_test_split(x_EGO, f_EGO, random_state=params.config.seed, test_size=0.20)
    data.train_x, data.val_x, data.train_g, data.val_g = train_x, val_x, train_g, val_g
    best_loss, data = train_model_EGO(data, params)
    data.model.eval()
    data.likelihood.eval()
    return best_loss, data

def log_iteration_progress(it, f_EGO, best_loss):
    """
    Logs the progress of the current iteration.

    Args:
        it (int): Current iteration number.
        f_EGO (torch.Tensor): Function evaluations.
        best_loss (List[float]): Best training and validation losses.
    """
    print(f'\nIteration {it}. Best of DGPR: {torch.min(f_EGO):.2f}. GPR model: Train loss = {best_loss[0]:.2f}; Val. loss = {best_loss[-1]:.2f}')


# -------------------- Bayesian Optimization --------------------
def bayesian_optimization(Data, Params):
    """
    Data: Runtime data
    Params: Params specific to the example
    """
    bounds = Params.optimization.bounds_opt
    dim = len(bounds)
    n = Params.optimization.n_initial_ego
    n_infill = Params.optimization.n_infill_ego
    
    # Calculate BSA popsize and epoch:
    ## Generate possible values for each variable based on bounds
    possibilities = torch.Tensor([torch.arange(lb, ub + 1).sum() for lb, ub in bounds])
    possibilities = torch.prod(possibilities)
    
    bsa_popsz = int(torch.log(possibilities) + 1)*2  # min. 2 of pop
    bsa_epoch = int(possibilities / bsa_popsz + 2)
    
    bsa_bounds = (tuple((0, 1) for _ in range(dim)))

    seed = Params.config.seed
    spectral_normalization = Params.surrogate.spectral_normalization

    # Initial sampling plan
    x_EGO = LHS([{} for _ in range(dim)], n, seed)

    # Evaluate initial sampling plan
    OptData = RuntimeData(
        x=Data.x,
        g=Data.g,
        x_opt=x_EGO,
        f_opt=f_EGO
        )
    f_EGO, OptData = obj_fun(x_EGO, OptData, Params)
    f_EGO = f_EGO.view(n)
    Data = keep_best(Data, OptData)

    # Train model
    best_loss, OptData = update_model(x_EGO, f_EGO, OptData, Params)

    it = 0
    log_iteration_progress(it, f_EGO, best_loss)
    while it < n_infill:
        it += 1

        # Search for the maximum expected improvement
        new_point = bsa(expected_improvement, bounds=bsa_bounds,
                        popsize=bsa_popsz, epoch=bsa_epoch, data=OptData.model)
        x_new = torch.from_numpy(new_point.x)
        # EI = new_point.y

        # Objective function at the new point
        f_new, OptData = obj_fun(x_new, OptData, Params)
        f_new = f_new.view(-1)
        x_new = x_new.view(-1, dim)
        OptData.f_opt, OptData.x_opt = f_new, x_new

        print(f'Iteration {it} of {n_infill}')

        # New best
        if f_new < torch.min(f_EGO):
            print(f'New best: {float(f_new):.2f} at position {it}')
            Data = keep_best(Data, OptData)

        # Add new values to the initial sampling
        x_EGO = torch.cat((x_EGO, x_new), 0)
        x_EGO = x_EGO.to(torch.float32)
        f_EGO = torch.cat((f_EGO, f_new), 0)

        # Update model
        best_loss, OptData = update_model(x_EGO, f_EGO, OptData, Params)
        log_iteration_progress(it, f_EGO, best_loss)

        # if abs(EI) < TOL_MIN_EI:
        #     print('Optimization finished. Minimum tolerance achieved.')
        #     break

    print(f'f*: {torch.min(f_EGO):.2f}; x*: {x_EGO[torch.argmin(f_EGO), :]}\n')

    # FINAL RESULT OF BAYESIAN OPTIMIZATION
    Data.x_opt, Data.f_opt = x_EGO, f_EGO
    layer_sizes, act_fun = optimization_variables(Data, Params, get_best=True)

    print(f'Loss: {torch.min(f_EGO):.2f}')
    print(f'SN: {spectral_normalization}')
    print(f'Hyperparameters: {layer_sizes}')
    print(f'act_fun: {act_fun.__name__}')

    return Data
