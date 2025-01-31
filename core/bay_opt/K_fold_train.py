from sklearn.model_selection import KFold
from ..utils import min_max_normalization


def kfold_train(x, g, x_candidate, epochs, lr, layer_sizes, activation_fn,
                train_model, MC_prediction, evaluate_lf, estimate_Pf, learning_function,
                N, N_MC, ALPHA, SPECTRAL_NORMALIZATION, n_splits=6, SEED=42):
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    avg_losses = []
    it = 0
    
    best_loss = 1e2

    # Split data into training and validation using KFold
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
        x_candidate_normalized = min_max_normalization(x_max, x_min, x_candidate)

        # Train the model using the training set
        success, attempts, max_attempts = False, 0, 2
        while not success and attempts < max_attempts:
            try:
                model, likelihood, avg_loss, model_training_losses, model_validation_losses = train_model(
                    train_x, train_g, val_x, val_g, epochs, lr, layer_sizes, activation_fn, SPECTRAL_NORMALIZATION)
                success = True  # Training was successful, exit loop
                print(f"  Training succeeded after {attempts + 1} attempt(s).")
            except Exception as e:
                attempts += 1
                print(f"  Error occurred during training attempt {attempts}: {e}")
                print("  Retrying training...")

                if attempts >= max_attempts:
                    print("  Max retry attempts reached. Training failed.")
                    # Optionally, continue with the next fold if training fails
                    success = False
                    break
        
        if not success:
            continue
        
        # Predict MC responses (only the sample which are not contained in the Kriging yet)
        preds = MC_prediction(model, likelihood, x_candidate_normalized)
        
        # Evaluate learning function
        g_mean, gs, _ = evaluate_lf(preds, learning_function)
        
        # Estimate Pf
        Pf, Pf_plus, Pf_minus = estimate_Pf(g, g_mean, gs, N, N_MC, ALPHA)

        # Update best model
        delta = (Pf_plus - Pf_minus)/Pf
        print(f'delta = {delta*100:.2f}%, avg. loss = {avg_loss:.4f}')
        
        if avg_loss < best_loss and Pf_plus > 1e-4:
            print(f'New best model found at fold {it}:')
            print(f'    delta = {delta*100:.2f}%, avg. loss = {avg_loss:.4f}')
            print(f'    layer_sizes: {layer_sizes}, act fun: {activation_fn}\n')
            best_loss = avg_loss
            best_val_losses = model_validation_losses
            best_train_losses = model_training_losses
            best_model = model
            best_likelihood = likelihood
            best_train_x = train_x
            best_val_x = val_x
            best_train_g = train_g
            best_val_g = val_g
            best_x_max, best_x_min = x_max, x_min
            best_fold = it

        avg_losses.append(avg_loss)
        print(f'Fold: {it}, Avg. Loss: {avg_loss}\n')
    
    if len(avg_losses) == 0:
        return 1e2
    
    # Average validation loss over all folds
    avg_avg_loss = sum(avg_losses) / n_splits
    print(f'Average Loss across {n_splits} folds: {avg_avg_loss}\n')
    
    return avg_avg_loss, best_loss, best_model, best_likelihood, best_train_losses, best_val_losses, \
        best_train_x, best_val_x, best_train_g, best_val_g, best_x_max, best_x_min, best_fold
