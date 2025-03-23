from core.configs import Config, SurrogateParams, \
    ReliabilityParams, SensitivityParams, OptimizationParams, Params

config = Config(
    example='example1',
    seed=42,
    sampling_plan_strategy='LHS',
    learning_function='U',
    convergence_function='stop_Pf',
    n_initial=81,
    n_infill=419
)

reliability = ReliabilityParams(
    method='MCS',
    n=5e5,
    alpha=0.05
)

sensitivity = SensitivityParams(
    n=5e4,
)

surrogate = SurrogateParams(
    model='SNDGPR',
    training_iterations=1000,
    learning_rate=0.01,
    validation_split=5,
    spectral_normalization=True
)

optimization = OptimizationParams(
    opt_type='grid_search',
    bounds_opt=[[1, 3], [1, 3]], #, [0, 4]],  # L, r, act_fun
    opt_inside_AL=True,
    n_initial_ego=5,
    n_infill_ego=2,
    dim_ego=3,
    training_iterations_ego=10000,
    learning_rate_ego=0.005,
)

params = Params(
    config=config,
    reliability=reliability,
    surrogate=surrogate,
    optimization=optimization,
    sensitivity=sensitivity,
)
