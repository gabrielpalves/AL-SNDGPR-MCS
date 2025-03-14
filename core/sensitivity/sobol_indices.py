from SALib.sample import saltelli
from SALib.analyze import sobol

def sensitivity_analysis(RVs, Data, Params, predict, evaluate_lf):
    problem = {
        'num_vars': len(RVs),
        'names': [],
        'bounds': []
    }

    for random_variable in RVs:
        problem['names'].append(random_variable['name'])
        problem['bounds'].append([0, 1])  # assuming min-max scaling

    # Generate Sobol sequence samples using Saltelli’s method
    n = Params.sensitivity.n
    param_values = saltelli.sample(problem, n)

    # Evaluate the model
    preds = predict(Data, param_values)
    Y, _, _ = evaluate_lf(preds)

    # Compute Sobol' indices
    Si = sobol.analyze(problem, Y)

    # Display results
    print("First-order Sobol' indices:", Si['S1'])
    print("Total-order Sobol' indices:", Si['ST'])
    
    return Si
