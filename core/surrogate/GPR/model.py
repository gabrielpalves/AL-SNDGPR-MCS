from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.utils.grid import ScaleToBounds


class GPRegressionModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
        self.scale_to_bounds = ScaleToBounds(-1., 1.)
        
        # Store train_y for later use
        self.train_outputs = train_y

    def forward(self, x):
        projected_x = self.scale_to_bounds(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return MultivariateNormal(mean_x, covar_x)
