import importlib

# Functions to dynamically load modules and functions
def load_core_modules(Params):
    sampling_plan_module = importlib.import_module("core.initial_sampling_plan")
    initial_sampling_plan = getattr(sampling_plan_module, Params.config.sampling_plan_strategy)
    
    learning_function_module = importlib.import_module("core.learning_function")
    learning_function = getattr(learning_function_module, Params.config.learning_function)
    
    convergence_function_module = importlib.import_module("core.convergence_function")
    convergence_function = getattr(convergence_function_module, Params.config.convergence_function)
    
    return initial_sampling_plan, learning_function, convergence_function


def load_surrogate_modules(Params):
    predict_module = importlib.import_module(f"core.surrogate.{Params.surrogate.model}.prediction")
    predict = getattr(predict_module, "predict")
    
    train_module = importlib.import_module(f"core.surrogate.{Params.surrogate.model}.train")
    train = getattr(train_module, "train_model")
    
    return predict, train


def load_optimization_modules(Params):
    opt_module = importlib.import_module(
        f"core.hyper_params_opt.{Params.optimization.opt_type}.optimize"
        )
    opt_type = getattr(opt_module, "optimize")
    
    return opt_type


def load_reliability_modules(Params):
    estimate_pf_module = importlib.import_module(
        f"core.reliability.{Params.reliability.method}.estimate_pf")
    estimate_Pf = getattr(estimate_pf_module, "estimate_Pf")
    
    sampling_plan_module = importlib.import_module(
        f"core.reliability.{Params.reliability.method}.sampling_plan")
    sampling_plan = getattr(sampling_plan_module, "sampling_plan")
    
    return estimate_Pf, sampling_plan


def load_sensitivity_modules(Params):
    sensitivity_module = importlib.import_module(
        f"core.sensitivity.{Params.sensitivity.type}"
        )
    sensitivity_analysis = getattr(sensitivity_module, "sensitivity_analysis")
    
    return sensitivity_analysis


def load_example_modules(example):
    random_variables_module = importlib.import_module(f"examples.{example}.random_variables")
    RVs = getattr(random_variables_module, "RVs")
    
    limit_state_module = importlib.import_module(f"examples.{example}.limit_state_function")
    limit_state_function = getattr(limit_state_module, "limit_state_function")
    
    configs = importlib.import_module(f"examples.{example}.configs")
    params = getattr(configs, "params")
    params.config.example = example
    
    return RVs, limit_state_function, params
