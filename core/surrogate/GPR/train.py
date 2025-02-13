import torch
from torch.optim import Adam
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from core.surrogate.GPR.model import GPRegressionModel
import os.path


def train_model(Data, Params, opt=True):
    train_x, train_g, val_x, val_g = Data.train_x, Data.train_g, Data.val_x, Data.val_g
    
    training_iterations = Params.surrogate.training
    lr = Params.surrogate.learning_rate
    if opt:
        # check if it used for optimization of hyperparameters
        # or if it is the surrogate used to calculate the Pf
        training_iterations = Params.optimization.training_iterations_ego
        lr = Params.optimization.learning_rate_ego
    
    EXAMPLE = Params.config.example

    # Initialize the models and likelihood
    likelihood = GaussianLikelihood()
    model = GPRegressionModel(train_x=train_x, train_y=train_g, likelihood=likelihood)
    
    folder_path = os.path.join("examples", EXAMPLE, "data", "best_models", "temp")
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    model_path = os.path.join(folder_path, f'best_model_and_likelihood_GPR.pth')

    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     likelihood = likelihood.cuda()

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = Adam([
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # To track loss values
    training_losses = []
    validation_losses = []

    # Training loop with validation
    def train():
        best_loss, best_val_loss, best_train_loss = 1e8, 1e8, 1e8
        patience = int(training_iterations * 0.025)
        wait = 0

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

            # Validation step
            model.eval()
            likelihood.eval()
            with torch.no_grad():
                val_output = model(val_x)
                val_loss = -mll(val_output, val_g).item()

            # Save the best model based on validation and training loss
            training_loss = loss.item()
            final_loss = val_loss*1.0
            if final_loss < best_loss:
                best_loss = final_loss
                best_val_loss = val_loss
                best_train_loss = training_loss
                best_epoch = epoch
                # torch.save(model.state_dict(), 'best_model_GPR.pth')
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
        return best_val_loss, best_train_loss
    best_loss = train()

    # Load the best model state
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])

    # Set model and likelihood to eval mode for further evaluation
    model.eval()
    likelihood.eval()

    # Plot training and validation loss
    # plt.figure(figsize=(4, 2))
    # plt.plot(training_losses, label='Training Loss')
    # plt.plot(validation_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss Over Epochs')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return model, likelihood, best_loss, training_losses, validation_losses
