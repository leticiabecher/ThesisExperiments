"""
========================================================================
PEC Experiments — Section 5.1 (Twin Benchmark for Parameter Calibration)
------------------------------------------------------------------------
Context:
    This executable reproduces the computational workflow reported in
    Chapter 5, Section 5.1 of the thesis, and is included as supplementary
    material in Appendix A.

------------------------------------------------------------------------
This executable performs:
    (1) Artificial-observation generation
    (2) Heuristic search
    (3) DFO-LS optimization
    (4) Structured JSON output and PDF reporting

Execution modes:
    - Light mode: numerical smoke test.
    - Full mode: experiments reported in the thesis.

Author:
    Letícia Becher Yamashita
========================================================================
"""

# ===============================================================
# 1. IMPORTS
# ===============================================================

from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
import numpy as np
import PEC_aux_functions as PEC
import dfols

# ===============================================================
# 2. EXECUTION CONFIGURATION
# ===============================================================
# Output file to save results
output_pdf="PEC_Chapter5_1_Report.pdf"
output_json = "PEC_5_benchmark_1.jsonl"

PEC.set_model_mode("2param") # For experiments calibrating Vmax and Lambda
print(f"[MODEL_MODE = {PEC.MODEL_MODE}]")

# Light mode ON for quick testing (set False for full experiment)
PEC.set_light_mode(True)

# Model
system  = PEC.system
# Heuristic search
N_H       = PEC.N_H
tmaxH_aux = PEC.tmaxH_aux
# Optimization
N         = PEC.N
tmax_aux  = PEC.tmax_aux
# Utilities
get_bounds              = PEC.get_bounds
generateObservations    = PEC.generateObservations
systemPredictedSolution = PEC.systemPredictedSolution
get_n_workers           = PEC.get_n_workers

# Observational parameters, used to generate artificial observations
ObsParams = np.array([1.0, 0.05])

# Other experiment presets (uncomment manually when needed) ----------------
# output_json = "PEC_5_benchmark_2.jsonl"; ObsParams = np.array([1.4,0.05])
# output_json = "PEC_5_benchmark_3.jsonl"; ObsParams = np.array([1.4,0.3])
# output_json = "PEC_5_benchmark_4.jsonl"; ObsParams = np.array([2.0,0.05])
# output_json = "PEC_5_benchmark_5.jsonl"; ObsParams = np.array([2.0,0.3])

# ===============================================================
# 3. ARTIFICIAL OBSERVATIONS
# ===============================================================

heuristic_observations = PEC.generateArtificialObservations(ObsParams, N_H)
y0 = np.array([np.mean(heuristic_observations[0,:]),np.mean(heuristic_observations[1,:])])

print("Reached part 1: Artificial observations generated.")

# ===============================================================
# 4. HEURISTIC SEARCH
# ===============================================================

def misfitH(params):
    """
    Light version of the misfit function associated with (Vmax, lambda), used for heuristics. 
    Not vectorized. Returns a large value if integration fails.
    """   
    ode_solution = systemPredictedSolution(system, params, tmaxH_aux, N_H, y0)
    if ode_solution is None:
        return 1e12   # prevents crash during heuristic
    residualH = np.linalg.norm((ode_solution - heuristic_observations), axis=0)
    return 0.5*np.linalg.norm(residualH)

def grid_misfitH(args):
    """Wrapper used for parallel execution of misfitH."""    
    i, j, Vmax_values, lambda_values = args
    params = np.array([Vmax_values[i], lambda_values[j]])
    return (i, j, misfitH(params))

# Parameter grid
Boundsinf, Boundsup, boundsParams = get_bounds() # Updated bounds
Vmax_values = np.linspace(Boundsinf[0], Boundsup[0], PEC.grid_size_a)
lambda_values = np.linspace(Boundsinf[1], Boundsup[1], PEC.grid_size_b)
fGrid = -1*np.ones((len(Vmax_values), len(lambda_values)))

indices = itertools.product(range(len(Vmax_values)), range(len(lambda_values)))
args = [(i, j, Vmax_values, lambda_values) for i, j in indices]

# Parallel evaluation
with ProcessPoolExecutor(max_workers=get_n_workers()) as executor:
    futures = [executor.submit(grid_misfitH, arg) for arg in args]
    results = [f.result() for f in as_completed(futures)]

# Fill misfit grid
for i, j, misfit in results:
    fGrid[i, j] = misfit

IndicesH = np.argwhere(fGrid >= 0)
f_list = fGrid[IndicesH[:, 0], IndicesH[:, 1]]
param_list = np.column_stack((Vmax_values[IndicesH[:, 0]], lambda_values[IndicesH[:, 1]]))

MisfitHeuristic = [param_list.tolist(),f_list.tolist()]

print("Reached part 2: Grid heuristic finished.")

min_index = np.argmin(f_list)
InitialGuess = param_list[min_index]
f_InitialGuess = f_list[min_index]

MisfitMedian = np.mean(f_list)
fCeil = MisfitMedian
        
# Save heuristic results
results_dict = {
    "ObsParams": ObsParams.tolist(),
    "y0": y0.tolist(),
    "MisfitHeuristic": MisfitHeuristic,
    "fCeil": fCeil,
    "InitialGuess": InitialGuess.tolist()
}

with open(output_json, "a") as f: 
    f.write(json.dumps(results_dict) + "\n")

# ===============================================================
# 6. DFO-LS OPTIMIZATION
# ===============================================================

Obs = generateObservations(N,tmax_aux)
y0 = np.array([np.mean(Obs[0,:]),np.mean(Obs[1,:])])

# Reload heuristic data from JSON.
# Useful when partitioning the executable or parallelizing multiple runs.
with open(output_json, 'r') as file:
    line = next(file)
    data = json.loads(line)
    fCeil = data["fCeil"]
    InitialGuess = np.array(data["InitialGuess"])

# DFOLS evaluation history:
# Each entry has the form [[Vmax, lambda], misfit], allowing
# reconstruction of the convergence path for analysis and plotting.
dfols_history = []
        
def res(parameters):
    """Residual function passed to DFOLS."""
    
    ode_solution = systemPredictedSolution(system, parameters, tmax_aux, N, y0)
    if ode_solution is None:
        return np.ones(N) * fCeil   # prevents crash
    
    residual = np.linalg.norm((ode_solution - Obs), axis=0)

    misfit = 0.5*np.linalg.norm(residual)
    dfols_history.append([parameters.tolist(), misfit.item()])

    return np.minimum(fCeil, residual)
            
try:
    output1 = dfols.solve(res, x0=InitialGuess, bounds=boundsParams, scaling_within_bounds=True)    
    dfols_solution = output1.x  # Returns the optimal parameter vector

    print("Reached part 3: DFO-LS optimization finished.")

except Exception as e:
    print("Error in DFOLS.")
    dfols_solution = np.array([-1.,-1.])  # Default value in case of failure
        
# Save DFOLS results
results_dict = {
    "DFOLSiterations": dfols_history,
    "DFOLSoutput": dfols_solution.tolist(),
    "Bounds": [Boundsinf.tolist(), Boundsup.tolist()]
}

with open(output_json, "a") as f:
    f.write(json.dumps(results_dict) + "\n")

print("Execution completed successfully.")

PEC.generate_experiment_report(output_json, output_pdf)