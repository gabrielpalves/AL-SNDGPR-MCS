import os.path
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints.constraints import GreaterThan
from core.surrogate.SNDGPR.model import GPRegressionModel
from core.hyper_params_opt.optimization_variables import optimization_variables


def train_model(Data, Params, opt=False):
    train_x, train_g, val_x, val_g = Data.train_x, Data.train_g, Data.val_x, Data.val_g
    if torch.cuda.is_available():
        train_x, train_g, val_x, val_g = train_x.cuda(), train_g.cuda(), val_x.cuda(), val_g.cuda()
    
    spectral_normalization = Params.surrogate.spectral_normalization
    
    training_iterations = Params.surrogate.training_iterations
    lr = Params.surrogate.learning_rate
    if opt:
        spectral_normalization = True
        training_iterations = Params.optimization.training_iterations_ego
        lr = Params.optimization.learning_rate_ego

    layer_sizes, act_fun = Data.layer_sizes, Data.act_fun

    EXAMPLE = Params.config.example

    # Initialize the models and likelihood
    likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-4))
    model = GPRegressionModel(train_x=train_x, train_y=train_g,
                              likelihood=likelihood,
                              layer_sizes=layer_sizes,
                              activation_fn=act_fun,
                              spectral_normalization=spectral_normalization)
    
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    folder_path = os.path.join("examples", EXAMPLE, "data", "best_models", "temp")
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    model_path = os.path.join(folder_path, f'best_model_and_likelihood.pth')

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = Adam([
        {'params': model.feature_extractor.parameters()},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=lr)
    
    # Initialize Cosine Annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=64)

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # To track loss values
    training_losses = []
    validation_losses = []

    # Training loop with validation
    def train():
        best_loss, best_val_loss, best_train_loss = 1e8, 1e8, 1e8
        patience = int(training_iterations * 0.1)
        wait = 0
        best_epoch = 0

        for epoch in range(training_iterations):
            model.train()
            likelihood.train()

            # Zero backprop gradients
            optimizer.zero_grad()

            # Forward pass and calculate loss on training set
            output = model(train_x)
            loss = -mll(output, train_g)
            loss.backward()
            optimizer.step()
            
            # Update learning rate using scheduler
            scheduler.step()

            # Validation step
            model.eval()
            likelihood.eval()
            with torch.no_grad():
                val_output = model(val_x)
                val_loss = -mll(val_output, val_g).item()

            # Save the best model based on validation and training loss
            training_loss = loss.item()
            # considered_loss = val_loss*0.5 + training_loss*0.5
            considered_loss = val_loss*1.0
            if considered_loss < best_loss:
                best_loss = considered_loss*1.0
                best_val_loss = val_loss
                best_train_loss = training_loss
                best_epoch = epoch
                # torch.save(model.state_dict(), 'best_model.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'likelihood_state_dict': likelihood.state_dict(),
                }, model_path)
                wait = 0  # Reset patience counter when improvement is found
            else:
                wait += 1  # Increment patience counter if no improvement

            # Early stopping
            if wait > patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

            # Track losses for plotting
            training_losses.append(loss.item())
            validation_losses.append(val_loss)

            # print(f'Epoch {epoch + 1} - Training Loss: {loss.item()} - Validation Loss: {val_loss}')

        print(f'Best Loss: {best_loss} at epoch {best_epoch}. Training loss: {best_train_loss} and val. loss: {best_val_loss}')
        return best_loss
    best_loss = train()

    # Load the best model state
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])

    # Set model and likelihood to eval mode for further evaluation
    model.eval()
    likelihood.eval()
    
    Data.model, Data.likelihood = model, likelihood
    Data.train_losses, Data.val_losses = training_losses, validation_losses

    # Plot training and validation loss
    # plt.figure(figsize=(10, 5))
    # plt.plot(training_losses, label='Training Loss')
    # plt.plot(validation_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss Over Epochs')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return best_loss, Data
