import torch
import numpy as np
from scipy.stats import qmc
from sklearn.model_selection import train_test_split

from core.bay_opt.objective_function import obj_fun
from core.bay_opt.bsa import bsa
from core.bay_opt.train import train_model_EGO
from core.bay_opt.expected_improvement import expected_improvement
from core.bay_opt.optimization_variables import optimization_variables


def bayesian_optimization(
    x, g, x_candidate, N, N_MC, ALPHA, \
    N_INITIAL_EGO, N_INFILL_EGO, DIM_EGO, TRAINING_ITERATIONS_EGO, \
    BOUNDS_BSA, BSA_POPSIZE, BSA_EPOCH, \
    SEED, TRAINING_ITERATIONS, BOUNDS_BAY_OPT, SPECTRAL_NORMALIZATION, \
    VALIDATION_SPLIT, LEARNING_RATE,  Params
    ):
    
    x_EGO = qmc.LatinHypercube(d=DIM_EGO, optimization="random-cd", seed=SEED)
    lhs_sample = x_EGO.random(n=N_INITIAL_EGO)
    x_EGO = torch.from_numpy(lhs_sample)
    x_EGO = x_EGO.to(torch.float32)
    
    f_EGO, best_model, best_likelihood, best_train_losses, best_val_losses, \
          train_x, val_x, train_g, val_g, best_x_max, best_x_min = \
        obj_fun(x_EGO, x, g, x_candidate, N, N_MC, ALPHA, BOUNDS_BAY_OPT, SPECTRAL_NORMALIZATION, VALIDATION_SPLIT,
            TRAINING_ITERATIONS, LEARNING_RATE, SEED, Params)
    f_EGO = f_EGO.view(N_INITIAL_EGO)
    
    overall_best_model = best_model
    overall_best_likelihood = best_likelihood
    overall_best_train_losses = best_train_losses
    overall_best_val_losses = best_val_losses
    overall_best_train_x, overall_best_val_x = train_x, val_x
    overall_best_train_g, overall_best_val_g = train_g, val_g
    overall_best_x_max, overall_best_x_min = best_x_max, best_x_min

    train_x_EGO, val_x_EGO, train_g_EGO, val_g_EGO = train_test_split(
        x_EGO, f_EGO, random_state=SEED, test_size=0.20)

    model, likelihood, best_loss = train_model_EGO(
        train_x_EGO, train_g_EGO, val_x_EGO, val_g_EGO, TRAINING_ITERATIONS_EGO)
    model.eval()
    likelihood.eval()

    it = 0
    print(f'\nIteration {it}. Best of DGPR: {torch.min(f_EGO)}. GPR model: Train loss = {best_loss[0]}; Val. loss = {best_loss[-1]}')

    while it < N_INFILL_EGO:
        it += 1
        
        # Search for the maximum expected improvement
        new_point = bsa(expected_improvement, bounds=BOUNDS_BSA,
                        popsize=BSA_POPSIZE, epoch=BSA_EPOCH, data=model)
        x_new = torch.from_numpy(new_point.x)
        # EI = new_point.y

        # Objective function at the new point
        f_new, best_model, best_likelihood, best_train_losses, best_val_losses, \
            train_x, val_x, train_g, val_g, best_x_max, best_x_min = obj_fun(
                x_new, x, g, x_candidate, N, N_MC, ALPHA, BOUNDS_BAY_OPT, SPECTRAL_NORMALIZATION, VALIDATION_SPLIT,
            TRAINING_ITERATIONS, LEARNING_RATE, SEED, Params
                )
        f_new = f_new.view(-1)
        
        print(f'Iteration {it} of {N_INFILL_EGO}')
        if f_new < torch.min(f_EGO):
            print(f'New best: {float(f_new):.2f} at position {it}')
            overall_best_model = best_model
            overall_best_likelihood = best_likelihood
            overall_best_train_losses = best_train_losses
            overall_best_val_losses = best_val_losses
            overall_best_train_x, overall_best_val_x = train_x, val_x
            overall_best_train_g, overall_best_val_g = train_g, val_g
            overall_best_x_max, overall_best_x_min = best_x_max, best_x_min
        
        # Add new values to the initial sampling
        x_EGO = torch.cat((x_EGO, torch.from_numpy(np.array([np.asarray(x_new)]))), 0)
        x_EGO = x_EGO.to(torch.float32)
        f_EGO = torch.cat((f_EGO, f_new), 0)
        
        # Update model
        train_x_EGO, val_x_EGO, train_g_EGO, val_g_EGO = train_test_split(x_EGO, f_EGO, random_state=SEED, test_size=0.20)
        model, likelihood, best_loss = train_model_EGO(train_x_EGO, train_g_EGO, val_x_EGO, val_g_EGO, TRAINING_ITERATIONS)
        model.eval()
        likelihood.eval()
        
        print(f'\nIteration {it}. Best of DGPR: {torch.min(f_EGO)}. GPR model: Train loss = {best_loss[0]}; Val. loss = {best_loss[-1]}')
        
        # if abs(EI) < TOL_MIN_EI:
        #     print('Optimization finished. Minimum tolerance achieved.')
        #     break

    print(f'f*: {torch.min(f_EGO):.2f}; x*: {x_EGO[torch.argmin(f_EGO), :]}\n')
    
    # FINAL RESULT OF BAYESIAN OPTIMIZATION
    X = x_EGO[torch.argmin(f_EGO), :]
    layer_sizes, act_fun = optimization_variables(BOUNDS_BAY_OPT, X, x, SPECTRAL_NORMALIZATION)

    print(f'Loss: {torch.min(f_EGO):.2f}, Hyperparameters: {layer_sizes}, SN: {SPECTRAL_NORMALIZATION}, act_fun: {act_fun.__name__}')
    
    return f_EGO, x_EGO, overall_best_model, overall_best_likelihood, overall_best_train_losses, overall_best_val_losses, \
        overall_best_train_x, overall_best_val_x, overall_best_train_g, overall_best_val_g, overall_best_x_max, overall_best_x_min
