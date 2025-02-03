import os.path
import importlib
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from datetime import datetime
from scipy.special import ndtri
from .bay_opt import optimization_variables


def min_max_normalization(x_max, x_min, x_candidate):
    return (x_candidate - x_min) / (x_max - x_min)


def save_x_added(x_added, it, EXAMPLE):
    numpy_array = x_added.numpy()
    # Define the folder path within the example directory
    folder_path = os.path.join(EXAMPLE, "data/x")
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    # Save the .mat file inside the example's data folder
    file_name = f'x{it}.mat'
    full_path = os.path.join(folder_path, file_name)

    sio.savemat(full_path, {'x': numpy_array})


def evaluate_g(x_added, it, limit_state_function, EXAMPLE):
    file_name = f'g{it-1}.mat'
    full_path = os.path.join(folder_path, file_name)
    check_file = os.path.isfile(full_path)
    
    # Define the folder path within the example directory
    folder_path = os.path.join(EXAMPLE, "data/g")
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    if check_file:
        mat_contents = sio.loadmat(full_path)
        g_added = mat_contents['g']
        g_added = torch.Tensor(g_added)
        g_added = g_added.view(-1)
    else:
        # Add the selected sample to the experimental design
        g_added = limit_state_function(x_added)
        g_added = torch.Tensor(g_added)
        numpy_array = g_added.numpy()

        # Save the .mat file inside the example's data folder
        sio.savemat(full_path, {'g': numpy_array})
    return g_added


def save_bests(best_model, best_likelihood, best_training_losses, best_validation_losses,
               x, train_x, val_x, train_g, val_g, x_EGO, f_EGO, x_max, x_min,
               it, BOUNDS_BAY_OPT, SPECTRAL_NORMALIZATION, EXAMPLE):
    
    
    # Define the folder path for saving files
    folder_path = os.path.join(EXAMPLE, "data/best_model")
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    # Save the model state using torch
    torch_model_likelihood_path = os.path.join(folder_path, f'best_model_and_likelihood_{it}.pth')
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'likelihood_state_dict': best_likelihood.state_dict(),
    }, torch_model_likelihood_path)

    layer_sizes, act_fun = optimization_variables(BOUNDS_BAY_OPT, x_EGO[torch.argmin(f_EGO), :], x, SPECTRAL_NORMALIZATION)

    folder_path = os.path.join(EXAMPLE, "data/variables")
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    # Save the .mat file inside the example's data folder
    file_name = 'variables_best_' + act_fun.__name__ + '_SN_' + str(SPECTRAL_NORMALIZATION) + str(it) + '.mat'
    full_path = os.path.join(folder_path, file_name)

    sio.savemat(full_path, {
        'best_training_losses': np.array(best_training_losses),
        'best_validation_losses': np.array(best_validation_losses),
        'x_EGO': x_EGO.numpy(),
        'f_EGO': f_EGO.numpy(),
        'layer_sizes': np.array(layer_sizes),
        'train_x': np.array(train_x),
        'val_x': np.array(val_x),
        'train_g': np.array(train_g),
        'val_g': np.array(val_g),
        'x_max': np.array(x_max),
        'x_min': np.array(x_min),
        })


def pickle_save(Results, History, Params, example):
    """
    Saves the Results, History, and Params objects as pickle files inside the example's data directory.

    Args:
        Results: The results object to save.
        History: The history object to save.
        Params: The parameters object to save.
        example (str): The name of the example directory (e.g., 'collapse_simulation').
    """
    # Define the folder path
    folder_path = os.path.join(example, "data")
    os.makedirs(folder_path, exist_ok=True)  # Create the directory if it doesn't exist

    # Define full file paths
    results_path = os.path.join(folder_path, f'Results.pkl')
    history_path = os.path.join(folder_path, f'History.pkl')
    params_path = os.path.join(folder_path, f'Params.pkl')

    # Save the Results object
    with open(results_path, 'wb') as Results_file:
        pickle.dump(Results, Results_file)

    # Save the History object
    with open(history_path, 'wb') as History_file:
        pickle.dump(History, History_file)

    # Save the Params object
    with open(params_path, 'wb') as Params_file:
        pickle.dump(Params, Params_file)


def pickle_load(example):
    """
    Loads the Results, History, and Params objects from pickle files inside the example's data directory.

    Args:
        example (str): The name of the example directory (e.g., 'collapse_simulation').

    Returns:
        tuple: A tuple containing (Results, History, Params) loaded from the pickle files.
    """
    # Define the folder path
    folder_path = os.path.join(example, "data")

    # Define full file paths
    results_path = os.path.join(folder_path, 'Results.pkl')
    history_path = os.path.join(folder_path, 'History.pkl')
    params_path = os.path.join(folder_path, 'Params.pkl')

    # Load the Results object
    with open(results_path, 'rb') as Results_file:
        Results = pickle.load(Results_file)

    # Load the History object
    with open(history_path, 'rb') as History_file:
        History = pickle.load(History_file)

    # Load the Params object
    with open(params_path, 'rb') as Params_file:
        Params = pickle.load(Params_file)

    return Results, History, Params


def load_core_modules(initial_plan_name, learning_func_name, convergence_func_name):
    """
    Dynamically imports core functions based on user input.

    Args:
        initial_plan_name (str): The name of the initial sampling plan function/class (e.g., 'LHS').
        learning_func_name (str): The name of the learning function (e.g., 'U').
        convergence_func_name (str): The name of the convergence function (e.g., 'stop_Pf').

    Returns:
        tuple: (initial_sampling_plan, learning_function, convergence_function), where:
            - initial_sampling_plan is the imported sampling plan function or class.
            - learning_function is the imported learning function.
            - convergence_function is the imported convergence function.
    """
    try:
        # Import initial sampling plan
        sampling_plan_module = importlib.import_module("core.initial_sampling_plan")
        initial_sampling_plan = getattr(sampling_plan_module, initial_plan_name)

        # Import learning function
        learning_function_module = importlib.import_module("core.learning_function")
        learning_function = getattr(learning_function_module, learning_func_name)
        evaluate_lf = getattr(learning_function_module, "evaluate_lf")

        # Import convergence function
        convergence_function_module = importlib.import_module("core.convergence_function")
        convergence_function = getattr(convergence_function_module, convergence_func_name)

        return initial_sampling_plan, learning_function, evaluate_lf, convergence_function

    except ModuleNotFoundError as e:
        print(f"Module not found: {e}")
        raise e
    except AttributeError as e:
        print(f"Attribute not found in module: {e}")
        raise e


def load_example_modules(example):
    """
    Dynamically imports the 'RVs' and 'limit_state_function' modules for a given example.

    Args:
        example (str): The name of the example directory (e.g., 'collapse_simulation').

    Returns:
        tuple: (RVs, limit_state_function), where:
            - RVs is the dictionary of random variables.
            - limit_state_function is the function to evaluate the limit state.
    """
    try:
        # Import the random variables module and retrieve RVs
        random_variables_module = importlib.import_module(f"examples.{example}.random_variables")
        RVs = getattr(random_variables_module, "RVs")

        # Import the limit state function module and retrieve limit_state_function
        limit_state_module = importlib.import_module(f"examples.{example}.limit_state_function")
        limit_state_function = getattr(limit_state_module, "limit_state_function")

        return RVs, limit_state_function

    except ModuleNotFoundError as e:
        print(f"Module not found: {e}")
        raise e
    except AttributeError as e:
        print(f"Attribute not found in module: {e}")
        raise e


def print_sample_info(strategy, x):
    print(f'{strategy} sample -- \
mean: {torch.mean(x):4.2f}, \
std: {torch.std(x):4.2f}, \
max: {torch.max(x):4.2f}, \
min: {torch.min(x):4.2f}')


def sample_info(variable_specs, strategies, samples):
    """Print info about the sample(s)

    Args:
        variable_specs (list): list containing dictionaries
        with info about the variables
        strategies (list): list containing strings
        which contain the strategy(ies) used to generate the sample
        samples (list): contains the sample(s)
    """

    for i in range(len(variable_specs)):
        rv = variable_specs[i]
        name = rv['name']
        distribution = rv['distribution']
        moments = rv['moments']
        mean = moments[0]
        cov = moments[1]

        print(f'RV: {name} -- mean: {mean:4.2f}, std: {mean*cov:4.2f}, \
distribution: {distribution}')

        for j in range(len(strategies)):
            x = samples[j]
            print_sample_info(strategies[j], x[:, i])
        print('')


def plot_sample_histogram(strat, x, name, dist, mean, cov, color):
    plt.hist(x, bins=1000, color=color)

    plt.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.ylabel('Frequency')
    plt.xlabel('Value')
    plt.title(f'RV: {name} ({strat}) - {dist} distribution \
with mean {mean} and std {mean*cov}')

    fig = plt.gcf()
    fig.set_size_inches(16, 9)


def print_info(N, N_INFILL, it, Pf, Pf_plus, Pf_minus):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S %d-%m-%Y")
    print(f'Points added: {N+it} of {N+N_INFILL}')
    print(f'Current time: {current_time}')
    print(f'Pf: {Pf*100:.3f}% - Pf interval: [{Pf_minus*100:.3f}, {Pf_plus*100:.3f}]%')
    print(f'delta: {(Pf_plus - Pf_minus)*100/Pf:.3f}%')
    
    
def plot_losses(tloss, vloss, it):
    # Plot training and validation loss
    # plt.figure(figsize=(4, 2))
    # plt.plot(training_losses, label='Training Loss')
    # plt.plot(validation_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss Over Epochs')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    plt.figure(figsize=(16, 10), dpi=300)
    font_size = 32
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams["font.family"] = "Times New Roman"
    tloss, vloss = np.array(tloss), np.array(vloss)
    plt.plot(tloss, label='Training Loss', color='blue', linewidth=3)
    plt.plot(vloss, label='Validation Loss', color='red', linestyle="--", linewidth=3)
    
    # Axes and grid lines
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
    plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
    
    # Calculate the combined loss and find the minimum point
    combined_loss = 0.5 * tloss + vloss + 0.5
    min_index_comb = np.argmin(combined_loss)
    min_index = np.argmin(vloss)
    
    # Add vertical line at the minimum combined loss
    plt.axvline(min_index_comb, color='purple', linestyle=':', label='Min. avg. loss')
    plt.axvline(min_index, color='green', linestyle='-.', label='Min. val. loss')
    
    # Titles and labels
    plt.title("Loss during training", fontsize=font_size + 4)
    plt.xlabel("Epochs", fontsize=font_size)
    plt.ylabel("Loss", fontsize=font_size)

    # Legend and grid
    plt.legend(fontsize=font_size)
    plt.grid()
    plt.show()
    fig = plt.gcf()
    fig.savefig('Loss' + str(it) + '.png', dpi=300)


def results_print(Results, History, Params):
    model_evals = Results["Model_Evaluations"]
    N = History["N_Init"]
    N_added = History["N_Samples"][-1]
    Pf = Results["Pf"]
    Pf_CI = Results["Pf_CI"]
    Beta = Results["Beta"]
    CoV = Results["CoV"]
    print(f'Model evaluations: {model_evals} ({N} + {N_added})')
    print(f"Probability of failure: {Pf}")
    print(f"Confidence interval: [{Pf_CI[0]:.5f}, {Pf_CI[1]:.5f}] \
(alpha = {Params['alpha']})")
    print(f'Beta: {Beta:.3f}')
    print(f'CoV: {CoV:4f}')


def results_plot(Results, History, Params, file_name):
    plt.rcParams.update({'font.size': 28})
    # Plot Pf history
    x = np.array(History["N_Samples"]) + Params['N']
    y = np.array(History["Pf"])
    plt.plot(x, y, color=(0, 0.4470, 0.7410),
             linestyle='solid', linewidth=2, label=r'$P_f$')

    # # Plot Pf minus history
    ym = np.array(History["Pf_Lower"])
    # plt.plot(x, ym, 'k', linestyle='solid', linewidth=1.5)

    # # Plot Pf plus history
    yp = np.array(History["Pf_Upper"])
    # plt.plot(x, yp, 'k', linestyle='solid', linewidth=1.5)

    # Fill between Pf minus and Pf plus
    gray = 0.9
    plt.fill_between(x, ym, yp, alpha=0.3,
                     facecolor=(gray, gray, gray),
                     edgecolor=(0, 0, 0),
                     label=r'$P_f^+, P_f^-$')

    # Plot CI
    # CI_lower = Results["Pf_CI"][0]
    # CI_upper = Results["Pf_CI"][-1]
    # y = array([CI_lower, CI_upper])
    plt.errorbar(x[-1], y[-1],
                 yerr=Results["CoV"]*ndtri(1-Params['alpha']/2)*y[-1],
                 color=(0, 0, 0), linewidth=2, capsize=6,
                 label='CI MCS')

    # Plot IEC targets
    T = 150
    upper_IEC_target = 0.92*1/T
    y = np.array([upper_IEC_target, upper_IEC_target])
    plt.plot(np.array((x[0], x[-1])), y, 'k',
             linestyle='dashed', linewidth=2,
             label='IEC 60826 Target (1/T)')

    lower_IEC_target = 0.92*1/(2*T)
    y = np.array([lower_IEC_target, lower_IEC_target])
    plt.plot(np.array((x[0], x[-1])), y, 'k',
             linestyle='dashdot', linewidth=2,
             label='IEC 60826 Target (1/2T)')

    # limits and ticks
    plt.yticks(np.array([0, 0.005, 0.01, 0.015, 0.02]))
    plt.ylim([0, 0.02])
    plt.xticks(x)
    plt.xlim([x[0], x[-1]])
    plt.grid(color='gray', linestyle='-', linewidth=0.5)

    # Labels and legend
    plt.ylabel('Probability of failure')
    plt.xlabel('Model evaluations')
    plt.legend()

    # Set figure size and save
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    fig.savefig('Pf_convergence_' + file_name + '.png', dpi=300)
