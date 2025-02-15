import torch
import numpy as np
from scipy.stats import norm, rayleigh


def sampling_plan(variable_specs, size):
    # Number of variables and sampling points
    size = int(size)

    num_variables = len(variable_specs)
    # Generate sampling points
    points = np.empty((size, num_variables))

    for i, spec in enumerate(variable_specs):
        distribution = spec['distribution'].lower()
        parameters = spec['moments']
        bounds = []
        if 'bounds' in spec.keys():
            bounds = spec['bounds']

        if distribution == 'normal' or distribution == 'gaussian':
            # Normal (Gaussian) distribution
            # Parameters: mean (mu), standard deviation (sigma)
            mean, cv = parameters
            sigma = cv * mean

            points[:, i] = np.random.uniform(low=0.0, high=1.0, size=size)

            if bounds:
                Fbounds = norm.cdf(bounds, loc=mean, scale=sigma)
                Fa, Fb = Fbounds[0], Fbounds[1]
                points[:, i] = points[:, i]*(Fb-Fa) + Fa

            points[:, i] = norm.ppf(points[:, i],
                                    loc=mean, scale=sigma)

        elif distribution == 'uniform':
            # Uniform distribution
            # Parameters: lower bound (a), upper bound (b)
            a, b = parameters
            points[:, i] = np.random.uniform(low=a, high=b, size=size)

        elif distribution == 'lognormal':
            # Lognormal distribution
            # Parameters for the underlying normal distribution: mean (mu),
            # standard deviation (sigma)
            mean, cv = parameters
            std = mean*cv
            variance = std**2
            mu = np.log(mean**2 / np.sqrt(variance + mean**2))
            sigma = np.sqrt(np.log(variance / mean**2 + 1))
            points[:, i] = np.random.lognormal(mean=mu, sigma=sigma, size=size)

        elif distribution == 'gumbel':
            # Gumbel (Extreme Value Type I) distribution
            # Parameters: location (mu), scale (beta)
            mean, cv = parameters
            std = cv * mean
            beta = std * np.sqrt(6) / np.pi
            mu = mean - beta * 0.5772156649
            points[:, i] = np.random.gumbel(loc=mu, scale=beta, size=size)

        elif distribution == 'gamma':
            # Gamma distribution
            # Parameters: shape (k), scale (theta)
            mean, cv = parameters
            std = cv * mean
            k = np.square(mean/std)
            theta = np.square(std)/mean
            points[:, i] = np.random.gamma(shape=k, scale=theta, size=size)

        elif distribution == 'exponential':
            # Exponential distribution
            # Parameter: scale (1/lambda), where lambda is the rate
            scale = parameters
            points[:, i] = np.random.exponential(scale=scale, size=size)

        elif distribution == 'rayleigh':
            # Shifted Rayleigh distribution
            loc, scale = parameters
            
            points[:, i] = np.random.uniform(low=0.0, high=1.0, size=size)
            points[:, i] = rayleigh.ppf(points[:, i],
                                    loc=loc, scale=scale)

    return torch.Tensor(points)