import torch

from core.hyper_params_opt.bay_opt import bayesian_optimization
from core.utils.utils import load_core_modules, load_example_modules

def AL(data, params, hyperparams, optparams):
    RVs, limit_state_function = load_example_modules(EXAMPLE)
    initial_sampling_plan, learning_function, evaluate_lf, convergence_function = load_core_modules(
    SAMPLING_PLAN_STRATEGY, LEARNING_FUNCTION, CONVERGENCE_FUNCTION
    )
    
    x = initial_sampling_plan(N, RVs, SEED)
    x_candidate = sampling_plan(N_PREDICT, RVs)  # sampling plan to predict

    data_dim = x.shape[1]
    estimate_Pf_all = []
    estimate_Pf_allp = []
    estimate_Pf_allm = []
    estimate_N_samples_added = []
    N_samples_added_total = 0
    converged = False
    it = 0

    while True:
        print(f'\nIteration {it}')
        
        if it == 0:
            f_EGO, x_EGO, model, likelihood, train_losses, val_losses, train_x, val_x, train_g, val_g, x_max, x_min \
                = bayesian_optimization(
                    x, g, x_candidate, N, N_MC, ALPHA, N_INITIAL_EGO, N_INFILL_EGO, DIM_EGO, \
                    TRAINING_ITERATIONS_EGO, BOUNDS_BSA, BSA_POPSIZE, BSA_EPOCH, \
                    SEED, TRAINING_ITERATIONS, BOUNDS_OPT, SPECTRAL_NORMALIZATION, \
                    VALIDATION_SPLIT, LEARNING_RATE, Params
                    )
                
        else:
            layer_sizes, act_fun = optimization_variables(BOUNDS_OPT, x_EGO[torch.argmin(f_EGO), :], x, SPECTRAL_NORMALIZATION)
            
            _, _, model, likelihood, train_losses, val_losses, train_x, val_x, train_g, val_g, x_max, x_min, fold \
                = kfold_train(
                x, g, x_candidate, TRAINING_ITERATIONS, LEARNING_RATE, layer_sizes, act_fun, \
                    N, N_MC, ALPHA, SPECTRAL_NORMALIZATION, Params, n_splits=VALIDATION_SPLIT, SEED=SEED
                )
        
        # Save variables and plot loss
        save_bests(model, likelihood, train_losses, val_losses, x, train_x, val_x, train_g, val_g,
                x_EGO, f_EGO, x_max, x_min, it, BOUNDS_OPT, SPECTRAL_NORMALIZATION, EXAMPLE)
        plot_losses(train_losses, val_losses, it)
        
        # Predict MC responses (only the sample which are not contained in the Kriging yet)
        x_candidate_normalized = min_max_normalization(x_max, x_min, x_candidate)
        preds = MC_prediction(model, likelihood, x_candidate_normalized)
        
        # Evaluate learning function
        g_mean, gs, ind_lf = evaluate_lf(preds, learning_function)
        
        # Select additional sample (the sample which maximizes the learning function value)
        x_added = add_x(x_candidate, ind_lf, it, EXAMPLE)
        
        # Estimate Pf
        Pf, Pf_plus, Pf_minus = estimate_Pf(g, g_mean, gs, N, N_MC, ALPHA)
        
        estimate_Pf_all.append(Pf)
        estimate_Pf_allp.append(Pf_plus)
        estimate_Pf_allm.append(Pf_minus)
        estimate_N_samples_added.append(N_samples_added_total)
        
        # Print some info
        print_info(N, N_INFILL, it, Pf, Pf_plus, Pf_minus)
        
        # Check if maximum number of points were added
        if N_samples_added_total >= N_INFILL: break
        it += 1
        
        # Convergence criterion
        if converged and N_samples_added_total != 0:
            if convergence_function(g, g_mean, gs, N, N_MC): break
            converged = False
        else:
            converged = convergence_function(g, g_mean, gs, N, N_MC)
        
        g_added = evaluate_g(x_added, it, limit_state_function, EXAMPLE)
        
        x = torch.cat((x, x_added), 0)
        g = torch.cat((g, g_added), 0)
        x_candidate = torch.cat((x_candidate[:ind_lf], x_candidate[ind_lf+1:]))
        N_samples_added_total = N_samples_added_total + 1
        
    # Store results
    # Estimate failure probability
    estimate_Pf_0 = (torch.sum(g_mean <= 0) + torch.sum(g[N+1:] <= 0))/N_MC

    # Estimate the covariance
    estimate_CoV = torch.sqrt((1-estimate_Pf_0) / estimate_Pf_0 / N_MC)

    # Store the results
    Results = {
        'Pf': estimate_Pf_0,
        'Beta': -ndtri(estimate_Pf_0),
        'CoV': estimate_CoV,
        'Model_Evaluations': N_samples_added_total + N,
        'Pf_CI': estimate_Pf_0 * np.array([
            1 + ndtri(ALPHA/2)*estimate_CoV,
            1 + ndtri(1-ALPHA/2)*estimate_CoV
            ]),
        }
    Results['Beta_CI'] = torch.flip(-ndtri(Results['Pf_CI']), [0])

    History = {
        'Pf': estimate_Pf_all,
        'Pf_Upper': estimate_Pf_allp,
        'Pf_Lower': estimate_Pf_allm,
        'N_Samples': estimate_N_samples_added,
        'N_Init': N,
        'X': x,
        'G': g,
        'MC_Sample': x_candidate,
    }

    data2save = {
        "Results": Results,
        "History": History,
        "Params": Params
        }
    pickle_save(data2save, EXAMPLE)
    
    return Results, History