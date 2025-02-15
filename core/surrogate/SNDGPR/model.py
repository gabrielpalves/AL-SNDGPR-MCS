import torch
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, ELU, Tanh, Sigmoid, GELU, init
from torch.nn.utils import spectral_norm
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.utils.grid import ScaleToBounds


class DNN(Sequential):
    def __init__(self, data_dim, layer_sizes, activation_fn=ReLU, spectral_normalization=True):
        super(DNN, self).__init__()
        
        # Define the layer configurations
        layer_size = [data_dim] + layer_sizes
        
        # Add layers dynamically
        for i in range(len(layer_size) - 1):
            in_size = layer_size[i]
            out_size = layer_size[i + 1]
            
            # Add linear layer with or without spectral normalization
            linear_layer = torch.nn.Linear(in_size, out_size)
            if spectral_normalization:
                linear_layer = spectral_norm(linear_layer)
            
            self.add_module(f'linear{i + 1}', linear_layer)
            
            # Add the chosen activation function if it's not the last layer
            if i < len(layer_size) - 2:
                self.add_module(f'activation{i + 1}', activation_fn())


        # Initialize weights using Kaiming initialization
        # Map activation functions to their corresponding Kaiming nonlinearity
        activation_to_nonlinearity = {
            ReLU: 'relu',
            LeakyReLU: 'relu',
            ELU: 'relu',
            Tanh: 'tanh',
            Sigmoid: 'sigmoid',
            GELU: 'relu'      # Approximation
        }

        # Use the activation function to determine nonlinearity
        nonlinearity = activation_to_nonlinearity.get(activation_fn, 'relu')  # Default to 'relu'
        
        for m in self.modules():
            if isinstance(m, Linear):
                init.kaiming_normal_(m.weight, mode='fan_out',
                                     nonlinearity=nonlinearity)
                # If your Linear layer has biases, initialize them to zero
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class GPRegressionModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, layer_sizes, activation_fn, spectral_normalization):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        data_dim = train_x.size(-1)  # train_x is a 2D tensor [N, data_dim]

        self.spectral_normalization = spectral_normalization
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(MaternKernel(nu=1.5))
        self.feature_extractor = DNN(data_dim=data_dim,
                                     layer_sizes=layer_sizes,
                                     activation_fn=activation_fn,
                                     spectral_normalization=spectral_normalization)
        if not spectral_normalization:
            self.scale_to_bounds = ScaleToBounds(-1., 1.)

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        if not self.spectral_normalization:
            projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return MultivariateNormal(mean_x, covar_x)
