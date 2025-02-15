# AL-SNDGPR-MCS
Spectrally normalized deep neural network coupled with Gaussian process regression and active learning

## Main file
Use AL-SNDGPR.MCS.ipynb file to run the code of this repository.

Choose an example or implement yours based on some example in the "_examples_" folder.

## Configs

#### Config
The variable config defines the configurations of the active learning procedure and also your example. The _seed_ field is for reproducibility of results.

#### ReliabilityParams
There is only the Monte Carlo Simulation (MCS) method implemented for now. The field _method_ chooses the reliability assessment method, implemented inside the _reliability_ folder.
_n_ is the size of the MC sample and _alpha_ defines the confidence interval.

#### SurrogateParams
Define which surrogate will be used. There are two options, set by the _model_ field: "SNDGPR" or "GPR". The field spectral_normalization only works for the SNDGPR.

#### OptimizationParams
Define the type of hyperparameters optimization (check hyper_params_opt folder) bounds and the type of optimization: "grid_search", or "bay_opt". The former tries all possibilities inside bounds_opt and the latter is the Bayesian optimization framework.
