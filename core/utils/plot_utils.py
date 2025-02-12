import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtri
from datetime import datetime

# Functions related to plotting and visualization
def results_plot(history, params):
    x = np.array(history["N_Samples"]) + params.config.n_initial
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
(alpha = {Params.reliability.alpha})")
    print(f'Beta: {Beta:.3f}')
    print(f'CoV: {CoV:4f}')

def plot_losses(Data, it):
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

def print_info(Params, it, Pf, Pf_plus, Pf_minus):
    N = Params.config.n_initial
    N_INFILL = Params.config.n_infill
    current_time = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
    print(f'Points added: {N + it} of {N + N_INFILL}')
    print(f'Current time: {current_time}')
    print(f'Pf: {Pf * 100:.3f}% - Pf interval: [{Pf_minus * 100:.3f}, {Pf_plus * 100:.3f}]%')
    print(f'delta: {(Pf_plus - Pf_minus) * 100 / Pf:.3f}%')
