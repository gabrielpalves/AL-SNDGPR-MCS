import torch
from K_fold_train import kfold_train
from .optimization_variables import optimization_variables


def obj_fun(xx, x, g, x_candidate, BOUNDS_BAY_OPT, SPECTRAL_NORMALIZATION, VALIDATION_SPLIT,
            TRAINING_ITERATIONS, LEARNING_RATE):
    """
    Inputs:
        L -> number of layers
        r -> size of the latent layer
    Output:
        Best K-Fold validation
    """
    
    if len(xx.shape) == 1:
        xx = torch.reshape(xx, (1, xx.shape[0]))
    
    fobj_all = torch.zeros((xx.shape[0]))
    best_loss = 1e2
    
    for idx, X in enumerate(xx):
        layer_sizes, act_fun = optimization_variables(BOUNDS_BAY_OPT, X, x, SPECTRAL_NORMALIZATION)

        print(f'Hyperparameters: {layer_sizes}, SN: {SPECTRAL_NORMALIZATION}, act_fun: {act_fun.__name__}')
        
        # Train the model with the sampled hyperparameters
        fobj, loss, model, likelihood, train_losses, val_losses, train_x, val_x, train_g, val_g, x_max, x_min, fold = \
            kfold_train(
            x, g, x_candidate, TRAINING_ITERATIONS, LEARNING_RATE, layer_sizes, act_fun, SPECTRAL_NORMALIZATION, n_splits=VALIDATION_SPLIT
            )
        print(f'obj fun (avg loss): {fobj:.2f} -> best fold: {fold}\n\n')
        
        fobj_all[idx] = fobj
        
        if fobj < best_loss:
            if best_loss < 1e2:
                print(f'New best found at OFE {idx}')
            best_loss = fobj
            best_model = model
            best_likelihood = likelihood
            best_train_losses = train_losses
            best_val_losses = val_losses
            best_train_x, best_val_x, best_train_g, best_val_g = train_x, val_x, train_g, val_g
            best_x_max, best_x_min = x_max, x_min

    return fobj_all, best_model, best_likelihood, best_train_losses, best_val_losses, \
        best_train_x, best_val_x, best_train_g, best_val_g, best_x_max, best_x_min
