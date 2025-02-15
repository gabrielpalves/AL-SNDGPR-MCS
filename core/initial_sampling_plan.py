from scipy.stats import qmc
from scipy.stats import uniform, norm, lognorm, gumbel_r, \
    gamma, expon, rayleigh
import numpy as np
import torch


def LHS(variable_specs, num_points, seed):
    """Latin Hypercube sampling plan"""
    # Number of variables and sampling points
    size = int(num_points)

    num_variables = len(variable_specs)

    # Generate LHS samples
    sampler = qmc.LatinHypercube(d=num_variables,
                                 optimization="random-cd",
                                 seed=seed)
    lhs_sample = sampler.random(n=size)

    # Initialize an array for the transformed samples
    transformed_sample = np.zeros_like(lhs_sample)

    # Example transformations for each variable
    for i, spec in enumerate(variable_specs):
        distribution = "uniform"
        if 'distribution' in spec.keys():
            distribution = spec['distribution'].lower()
        
        parameters = [0, 1]
        if 'moments' in spec.keys():
            parameters = spec['moments']
        
        bounds = []
        if 'bounds' in spec.keys():
            bounds = spec['bounds']

        if distribution == 'normal' or distribution == 'gaussian':
            # Normal (Gaussian) distribution
            # Parameters: mean (mu), standard deviation (sigma)
            mean, cv = parameters
            sigma = cv * mean

            if bounds:
                Fbounds = norm.cdf(bounds, loc=mean, scale=sigma)
                Fa, Fb = Fbounds[0], Fbounds[1]
                lhs_sample[:, i] = lhs_sample[:, i]*(Fb-Fa) + Fa

            transformed_sample[:, i] = norm.ppf(lhs_sample[:, i],
                                                loc=mean, scale=sigma)

        elif distribution == 'uniform':
            # Uniform distribution
            # Parameters: lower bound (a), upper bound (b)
            a, b = parameters
            transformed_sample[:, i] = uniform.ppf(lhs_sample[:, i],
                                                   loc=a, scale=b-a)

        elif distribution == 'lognormal':
            # Lognormal distribution
            # Lognormal distribution parameters are derived from
            # the underlying normal distribution's mu and sigma
            mean, cv = parameters
            sigma = np.sqrt(np.log(cv**2 + 1))
            mu = np.exp(np.log(mean) - 0.5 * sigma**2)
            transformed_sample[:, i] = lognorm.ppf(lhs_sample[:, i],
                                                   s=sigma, scale=mu)

        elif distribution == 'gumbel':
            # Gumbel (Extreme Value Type I) distribution
            # Parameters: location (mu), scale (beta)
            mean, cv = parameters
            std = cv * mean
            beta = std * np.sqrt(6) / np.pi
            mu = mean - beta * 0.5772156649
            transformed_sample[:, i] = gumbel_r.ppf(lhs_sample[:, i],
                                                    loc=mu, scale=beta)

        elif distribution == 'gamma':
            # Gamma distribution
            # Parameters: shape (k), scale (theta)
            mean, cv = parameters
            std = cv * mean
            alfa = np.square(mean/std)  # k
            beta = mean/np.square(std)  # theta or lambda

            transformed_sample[:, i] = gamma.ppf(lhs_sample[:, i],
                                                 a=alfa, scale=1/beta)

        elif distribution == 'exponential':
            # Exponential distribution
            # Parameter: scale (1/lambda), where lambda is the rate
            scale = parameters
            transformed_sample[:, i] = expon.ppf(lhs_sample[:, i], scale=scale)

        elif distribution == 'rayleigh':
            # Shifted Rayleigh distribution
            loc, scale = parameters

            # Generating the sample and then shifting
            transformed_sample[:, i] = rayleigh.ppf(lhs_sample[:, i],
                                    loc=loc, scale=scale)

    return torch.Tensor(transformed_sample)
