from sklearn.model_selection import KFold
from .utils.import_utils import load_surrogate_modules
from .utils.sampling_utils import min_max_normalization, keep_best
from core.configs import RuntimeData


def kfold_train(layer_sizes, act_fun, Data, Params):
    
    _, train_model = load_surrogate_modules(Params)
    n_splits = Params.surrogate.validation_split
    
    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=Params.config.seed
        )
    
    KData = RuntimeData(
        x=Data.x,
        g=Data.g,
        x_opt=Data.x_opt
        )
    
    avg_losses = []
    it = 0
    best_loss = 1e8

    # Split data into training and validation using KFold
    x, g = KData.x, KData.g
    for train_idx, val_idx in kf.split(x):
        it += 1
        # Training and validation sets
        train_idx = train_idx.tolist()
        val_idx = val_idx.tolist()
        train_x, val_x = x[train_idx], x[val_idx]
        train_g, val_g = g[train_idx], g[val_idx]
        
        # Normalize based on training data:
        x_max = train_x.max(dim=0)[0]
        x_min = train_x.min(dim=0)[0]
        train_x = min_max_normalization(x_max, x_min, train_x)
        val_x = min_max_normalization(x_max, x_min, val_x)
        
        KData.train_x, KData.val_x, KData.train_g, KData.val_g = train_x, val_x, train_g, val_g
        KData.x_min, KData.x_max = x_min, x_max

        # Train the model using the training set
        success, attempts, max_attempts = False, 0, 2
        while not success and attempts < max_attempts:
            try:
                model, likelihood, avg_loss, train_losses, val_losses \
                    = train_model(KData, Params)
                success = True  # Training was successful, exit loop
                print(f"  Training succeeded after {attempts + 1} attempt(s).")
            except Exception as e:
                attempts += 1
                print(f"  Error occurred during training attempt {attempts}: {e}")
                print("  Retrying training...")

                if attempts >= max_attempts:
                    print("  Max retry attempts reached. Training failed.")
                    success = False
                    break
        
        if not success:
            continue

        # Update best model
        print(f'avg loss = {avg_loss:.4f}')
        if avg_loss < best_loss:
            print(f'  New best model found at fold {it}:')
            print(f'    layer sizes: {layer_sizes}, act fun: {act_fun}\n')
            best_loss, best_fold = avg_loss, it
            KData.model, KData.likelihood = model, likelihood
            KData.train_losses, KData.val_losses = train_losses, val_losses
            Data = keep_best(Data, KData)

        avg_losses.append(avg_loss)
        print(f'Fold: {it}, Avg. Loss: {avg_loss}\n')
    
    if len(avg_losses) == 0:
        return 1e8
    
    # Average validation loss over all folds
    avg_avg_loss = sum(avg_losses) / n_splits
    print(f'Average Loss across {n_splits} folds: {avg_avg_loss}\n')
    
    return avg_avg_loss, best_fold, Data
