import random

import torch
import numpy as np
from scipy.special import ndtri

from core.hyper_params_opt.optimization_variables import optimization_variables
from core.K_fold_CV import kfold_train
from core.utils.sampling_utils import min_max_normalization, add_x, evaluate_g
from core.utils.serialization_utils import save_bests, pickle_save
from core.utils.plot_utils import plot_losses, print_info
from core.utils.import_utils import load_core_modules, load_example_modules, \
    load_surrogate_modules, load_reliability_modules, load_optimization_modules
from core.learning_function import evaluate_lf
from core.configs import RuntimeData


def AL(EXAMPLE):
    # Load important modules
    RVs, limit_state_function, Params = load_example_modules(EXAMPLE)
    initial_sampling_plan, learning_function, convergence_function \
        = load_core_modules(Params)
    predict, _ = load_surrogate_modules(Params)
    estimate_Pf, sampling_plan = load_reliability_modules(Params)
    hyper_params_opt = load_optimization_modules(Params)
    
    # Set seed for reproducibility
    SEED = Params.config.seed
    # For NumPy
    np.random.seed(SEED)

    # For PyTorch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU

    # For Python's built-in random module
    random.seed(SEED)

    # Ensuring reproducibility in cuDNN using PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Initiate samples
    Data = RuntimeData()
    Data.x = initial_sampling_plan(RVs, Params.config.n_initial, Params.config.seed)
    Data.x_candidate = sampling_plan(RVs, Params.reliability.n)  # sampling plan to predict
    Data.g = limit_state_function(Data.x)

    # Define important variables
    data_dim = Data.x.shape[1]
    estimate_Pf_all = []
    estimate_Pf_allp = []
    estimate_Pf_allm = []
    estimate_N_samples_added = []
    N_samples_added_total = 0
    converged = False
    it = 0

    # Begin AL loop
    while True:
        print(f'\nIteration {it}')
        
        if it == 0:  # optimize hyperparameters
            Data = hyper_params_opt(Data, Params)
        else:  # use already optimized hyperparameters
            Data = optimization_variables(Data, Params, get_best=True)
            _, _, Data = kfold_train(Data, Params)

        # Save variables and plot loss
        save_bests(it, Data, Params, EXAMPLE)
        plot_losses(it, Data)
        
        # Predict MC responses (only the sample which are not contained in the Kriging yet)
        x_candidate_normalized = min_max_normalization(Data.x_max, Data.x_min, Data.x_candidate)
        preds = predict(Data, x_candidate_normalized)
        
        # Evaluate learning function
        g_mean, gs, ind_lf = evaluate_lf(preds, learning_function)
        
        # Select additional sample (the sample which maximizes the learning function value)
        x_added = add_x(Data.x_candidate, ind_lf, it, EXAMPLE)
        
        # Estimate Pf
        Pf, Pf_plus, Pf_minus = estimate_Pf(Data.g, g_mean, gs, Params)
        
        # Append results
        estimate_Pf_all.append(Pf)
        estimate_Pf_allp.append(Pf_plus)
        estimate_Pf_allm.append(Pf_minus)
        estimate_N_samples_added.append(N_samples_added_total)
        
        # Print some info
        print_info(Params, it, Pf, Pf_plus, Pf_minus)
        
        # Check if maximum number of points were added
        if N_samples_added_total >= Params.config.n_infill: break
        it += 1
        
        # Convergence criterion
        if converged and N_samples_added_total != 0:
            if convergence_function(Data.g, g_mean, gs, Params): break
            converged = False
        else:
            converged = convergence_function(Data.g, g_mean, gs, Params)
        
        # Limit state function evaluation
        g_added = evaluate_g(x_added, it, limit_state_function, EXAMPLE)
        
        # Adjust sampling plans
        Data.x = torch.cat((Data.x, x_added), 0)
        Data.g = torch.cat((Data.g, g_added), 0)
        Data.x_candidate = torch.cat((Data.x_candidate[:ind_lf], Data.x_candidate[ind_lf+1:]))
        N_samples_added_total = N_samples_added_total + 1
        
    # Store results
    # Estimate failure probability
    estimate_Pf_0 = estimate_Pf_all[-1]

    # Estimate the covariance
    estimate_CoV = torch.sqrt((1-estimate_Pf_0) / estimate_Pf_0 / Params.reliability.n)

    # Store the results
    Results = {
        'Pf': estimate_Pf_0,
        'Beta': -ndtri(estimate_Pf_0),
        'CoV': estimate_CoV,
        'Model_Evaluations': N_samples_added_total + Params.config.n_initial,
        'Pf_CI': estimate_Pf_0 * np.array([
            1 + ndtri(Params.reliability.alpha/2)*estimate_CoV,
            1 + ndtri(1-Params.reliability.alpha/2)*estimate_CoV
            ]),
        }
    Results['Beta_CI'] = torch.flip(-ndtri(Results['Pf_CI']), [0])

    History = {
        'Pf': estimate_Pf_all,
        'Pf_Upper': estimate_Pf_allp,
        'Pf_Lower': estimate_Pf_allm,
        'N_Samples': estimate_N_samples_added,
        'N_Init': Params.config.n_initial,
        'X': Data.x,
        'G': Data.g,
        'MC_Sample': Data.x_candidate,
    }

    data2save = {
        "Results": Results,
        "History": History,
        "Params": Params
        }
    pickle_save(data2save, EXAMPLE)
    
    return Results, History, Params
