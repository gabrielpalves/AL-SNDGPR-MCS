import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtri

# Functions related to plotting and visualization
def results_plot(history, params):
    x = np.array(history["N_Samples"]) + params['N']
    y = np.array(history["Pf"])
    plt.plot(x, y, label='Probability of Failure', color='blue', linewidth=2)
    plt.fill_between(x, history["Pf_Lower"], history["Pf_Upper"], color='gray', alpha=0.3)
    plt.xlabel('Model Evaluations')
    plt.ylabel('Probability of Failure')
    plt.title('Pf Convergence')
    plt.legend()
    plt.grid()
    plt.show()

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

def plot_losses(training_losses, validation_losses):
    plt.figure(figsize=(16, 10), dpi=300)
    plt.plot(training_losses, label='Training Loss', color='blue', linewidth=3)
    plt.plot(validation_losses, label='Validation Loss', color='red', linestyle="--", linewidth=3)
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
    plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
    plt.title("Loss during training")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

def print_info(N, N_INFILL, it, Pf, Pf_plus, Pf_minus):
    from datetime import datetime
    current_time = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
    print(f'Points added: {N + it} of {N + N_INFILL}')
    print(f'Current time: {current_time}')
    print(f'Pf: {Pf * 100:.3f}% - Pf interval: [{Pf_minus * 100:.3f}, {Pf_plus * 100:.3f}]%')
    print(f'delta: {(Pf_plus - Pf_minus) * 100 / Pf:.3f}%')
