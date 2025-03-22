from core.configs import Config, SurrogateParams, \
    ReliabilityParams, SensitivityParams, OptimizationParams, Params

config = Config(
    example='guyed_NLG_no_wind',
    seed=42,
    sampling_plan_strategy='LHS',
    learning_function='U',
    convergence_function='stop_Pf',
    n_initial=60,
    n_infill=1940
)

reliability = ReliabilityParams(
    method='MCS',
    n=1e7,
    alpha=0.05
)

sensitivity = SensitivityParams(
    n=1e5,
    type='sobol_indices'
)

surrogate = SurrogateParams(
    model='SNDGPR',
    training_iterations=2500,
    learning_rate=0.01,
    validation_split=5,
    spectral_normalization=True
)

optimization = OptimizationParams(
    opt_type='grid_search',
    bounds_opt=[[1, 5], [1, 10]],#, [0, 4]],  # L, r, act_fun
    opt_inside_AL=True,
    n_initial_ego=20,
    n_infill_ego=10,
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
