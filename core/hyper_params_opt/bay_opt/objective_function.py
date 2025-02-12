import torch
from core.K_fold_CV import kfold_train
from core.hyper_params_opt import optimization_variables


def obj_fun(xx, Data, Params):
    """Outputs: Best K-Fold validation and its Runtime Data"""
    if len(xx.shape) == 1:
        xx = torch.reshape(xx, (1, xx.shape[0]))
    
    fobj_all = torch.zeros((xx.shape[0]))
    best_loss = 1e4
    
    for idx, X in enumerate(xx):
        Data.x_opt = X
        layer_sizes, act_fun = optimization_variables(Data, Params)

        print(f'Hyperparameters: {layer_sizes}, \
SN: {Params.surrogate.spectral_normalization}, \
act_fun: {act_fun.__name__}')
        
        # Train the model with the sampled hyperparameters
        fobj, fold, KData = kfold_train(layer_sizes, act_fun, Data, Params)

        print(f'obj fun (avg loss): {fobj:.2f} -> best fold: {fold}\n\n')
        
        fobj_all[idx] = fobj
        
        if fobj < best_loss:
            if best_loss < 1e4:
                print(f'New best found at OFE {idx}')
            best_loss = fobj
            Data = KData

    return fobj_all, Data
