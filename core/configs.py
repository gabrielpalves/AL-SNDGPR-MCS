from dataclasses import dataclass
from typing import List, Optional, Callable
import torch

# Active Learning Pipeline Configuration
@dataclass
class Config:
    example: str
    sampling_plan_strategy: str
    learning_function: str
    convergence_function: str
    n_initial: int
    n_infill: int
    seed: Optional[int] = 42

# Reliability and Prediction Parameters
@dataclass
class ReliabilityParams:
    method: str
    n: int
    alpha: float

# Surrogate and its Hyperparameters
@dataclass
class SurrogateParams:
    model: str
    training_iterations: int
    learning_rate: float
    validation_split: int
    spectral_normalization: Optional[bool] = False

# Optimization Parameters
@dataclass
class OptimizationParams:
    opt_type: str
    bounds_opt: list
    opt_inside_AL: bool = False
    n_initial_ego: Optional[int]
    n_infill_ego: Optional[int]
    dim_ego: Optional[int]
    training_iterations_ego: Optional[int]
    learning_rate_ego: Optional[float]

# Sensitivity analysis
@dataclass
class SensitivityParams:
    sensitivity_type: str = 'sobol_indices'

#########
# Data generated in the pipeline
@dataclass
class RuntimeData:
    x: Optional[torch.Tensor] = None
    g: Optional[torch.Tensor] = None
    x_candidate: Optional[torch.Tensor] = None
    f_opt: Optional[torch.Tensor] = None
    x_opt: Optional[torch.Tensor] = None
    model: Optional[torch.nn.Module] = None
    likelihood: Optional[object] = None
    layer_sizes: Optional[List[int]] = None
    act_fun: Optional[Callable] = torch.nn.ReLU
    train_losses: Optional[List[float]] = None
    val_losses: Optional[List[float]] = None
    train_x: Optional[torch.Tensor] = None
    val_x: Optional[torch.Tensor] = None
    train_g: Optional[torch.Tensor] = None
    val_g: Optional[torch.Tensor] = None
    x_max: Optional[torch.Tensor] = None
    x_min: Optional[torch.Tensor] = None

# Top-level dataclass that nests the others
@dataclass
class Params:
    config: Config
    reliability: ReliabilityParams
    surrogate: SurrogateParams
    optimization: OptimizationParams
    sensitivity: SensitivityParams
