import importlib

# Functions to dynamically load modules and functions
def load_core_modules(initial_plan_name, learning_func_name, convergence_func_name):
    sampling_plan_module = importlib.import_module("core.initial_sampling_plan")
    initial_sampling_plan = getattr(sampling_plan_module, initial_plan_name)
    learning_function_module = importlib.import_module("core.learning_function")
    learning_function = getattr(learning_function_module, learning_func_name)
    convergence_function_module = importlib.import_module("core.convergence_function")
    convergence_function = getattr(convergence_function_module, convergence_func_name)
    return initial_sampling_plan, learning_function, convergence_function

def load_example_modules(example):
    random_variables_module = importlib.import_module(f"examples.{example}.random_variables")
    RVs = getattr(random_variables_module, "RVs")
    limit_state_module = importlib.import_module(f"examples.{example}.limit_state_function")
    limit_state_function = getattr(limit_state_module, "limit_state_function")
    return RVs, limit_state_function
