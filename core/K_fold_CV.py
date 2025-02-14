from sklearn.model_selection import KFold
from .utils.import_utils import load_surrogate_modules
from .utils.sampling_utils import min_max_normalization, keep_best
from core.configs import RuntimeData


def kfold_train(Data, Params):
    
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
        x_opt=Data.x_opt,
        layer_sizes=Data.layer_sizes,
        act_fun=Data.act_fun
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
        KData.train_x, KData.val_x = x[train_idx], x[val_idx]
        KData.train_g, KData.val_g = g[train_idx], g[val_idx]
        
        # Normalize based on training data:
        KData.x_max = KData.train_x.max(dim=0)[0]
        KData.x_min = KData.train_x.min(dim=0)[0]
        KData.train_x = min_max_normalization(KData.x_max, KData.x_min, KData.train_x)
        KData.val_x = min_max_normalization(KData.x_max, KData.x_min, KData.val_x)

        # Train the model using the training set
        success, attempts, max_attempts = False, 0, 2
        while not success and attempts < max_attempts:
            try:
                avg_loss, KData = train_model(KData, Params)
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
            print(f'    layer sizes: {KData.layer_sizes}, act fun: {KData.act_fun}\n')
            best_loss, best_fold = avg_loss, it
            Data = keep_best(Data, KData)

        avg_losses.append(avg_loss)
        print(f'Fold: {it}, Avg. Loss: {avg_loss}\n')
    
    if len(avg_losses) == 0:
        return 1e8
    
    # Average validation loss over all folds
    avg_avg_loss = sum(avg_losses) / n_splits
    print(f'Average Loss across {n_splits} folds: {avg_avg_loss}\n')
    
    return avg_avg_loss, best_fold, Data
