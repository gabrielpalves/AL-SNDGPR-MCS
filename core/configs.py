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
    spectral_normalization: Optional[bool] = True

# Optimization Parameters
@dataclass
class OptimizationParams:
    bounds_opt: list
    opt_type: str
    opt_inside_AL: Optional[bool] = True
    n_initial_ego: Optional[int] = 15
    n_infill_ego: Optional[int] = 10
    dim_ego: Optional[int] = 3
    training_iterations_ego: Optional[int] = 10000
    learning_rate_ego: Optional[float] = 0.005

# Sensitivity analysis
@dataclass
class SensitivityParams:
    n: int

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
