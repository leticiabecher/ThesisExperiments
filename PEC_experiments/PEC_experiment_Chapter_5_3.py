"""
===================================================================
PEC Experiments — Section 5.3 (Calibration of nitrate parameter w)
-------------------------------------------------------------------
Context:
    This executable reproduces the computational workflow reported in
    Chapter 5, Section 5.3 of the thesis, and is included as supplementary
    material in Appendix A.

------------------------------------------------------------------
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
"""

# ===============================================================
# 1. IMPORTS
# ===============================================================

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import numpy as np
import PEC_aux_functions as PEC
import dfols

# ===============================================================
# 2. EXECUTION CONFIGURATION
# ===============================================================
# Output file to save results
output_pdf = "PEC_Chapter5_3_Report.pdf"
output_json = "PEC_5_section_5_3.jsonl"

PEC.set_model_mode("Nitrate_param") # For experiments calibrating w
print(f"[MODEL_MODE = {PEC.MODEL_MODE}]")

# Light mode ON for quick testing (set False for full experiment)
PEC.set_light_mode(True)

# Model
system = PEC.system
# Heuristic search
N_H = PEC.N_H
tmaxH_aux = PEC.tmaxH_aux
# Optimization
N = PEC.N
tmax_aux = PEC.tmax_aux
# Utilities
get_bounds              = PEC.get_bounds
generateObservations    = PEC.generateObservations
systemPredictedSolution = PEC.systemPredictedSolution
get_n_workers           = PEC.get_n_workers

# ===============================================================
# 3. "REAL" OBSERVATIONS
# ===============================================================

heuristic_observations = generateObservations(N_H,tmaxH_aux)
y0 = heuristic_observations[:,0]

print("Reached part 1: Artificial observations generated.")

# ===============================================================
# 4. HEURISTIC SEARCH
# ===============================================================

def misfitH(param):
    """
    Light version of the misfit function associated with w, used for heuristics. 
    Not vectorized. Returns a large value if integration fails.
    """   
    ode_solution = systemPredictedSolution(system, np.array([param]), tmaxH_aux, N_H, y0)
    if ode_solution is None:
        return 1e12   # prevents crash during heuristic
    residualH = np.linalg.norm((ode_solution - heuristic_observations), axis=0)
    return 0.5*np.linalg.norm(residualH)


# Parameter list
Boundsinf, Boundsup, boundsParams = get_bounds() # Updated bounds
w_values = np.linspace(Boundsinf[0], Boundsup[0], PEC.grid_size_b*3)

def list_misfitH(param):    
    return [param,misfitH(param)]

# Parallel evaluation
with ProcessPoolExecutor(max_workers=get_n_workers(), initializer=PEC.set_model_mode, 
                         initargs=("Nitrate_param",)) as executor:
    futures = [executor.submit(list_misfitH, p) for p in w_values]
    results = [f.result() for f in as_completed(futures)]

# Ordena pelo parâmetro (1ª coluna) para alinhar com param_list_a
results = np.array(results)
order = np.argsort(results, axis=0)[:,0]
f_list = results[order,1]

MisfitHeuristic = [w_values.tolist(),f_list.tolist()]

print("Reached part 2: Heuristic finished.")

min_index = np.argmin(f_list)
InitialGuess = w_values[min_index]
f_InitialGuess = f_list[min_index]

MisfitMedian = np.mean(f_list)
fCeil = MisfitMedian
        
# Save heuristic results
results_dict = {
    "y0": y0.tolist(),
    "MisfitHeuristic": MisfitHeuristic,
    "fCeil": fCeil,
    "InitialGuess": InitialGuess.item()
}

with open(output_json, "a") as f: 
    f.write(json.dumps(results_dict) + "\n")

# ===============================================================
# 6. DFO-LS OPTIMIZATION
# ===============================================================

Obs = generateObservations(N,tmax_aux)
y0 = Obs[:,0]

# Reload heuristic data from JSON.
# Useful when partitioning the executable or parallelizing multiple runs.
with open(output_json, 'r') as file:
    line = next(file)
    data = json.loads(line)
    fCeil = data["fCeil"]
    InitialGuess = np.array([data["InitialGuess"]])

# DFOLS evaluation history:
# Each entry has the form [w, misfit], allowing
# reconstruction of the convergence path for analysis and plotting.
dfols_history = []
        
def res(param):
    """Residual function passed to DFOLS."""
    
    ode_solution = systemPredictedSolution(system, np.array(param, dtype=float).ravel(), tmax_aux, N, y0)
    if ode_solution is None:
        return np.ones(N) * fCeil   # prevents crash
    
    residual = np.linalg.norm((ode_solution - Obs), axis=0)

    misfit = 0.5*np.linalg.norm(residual)
    dfols_history.append([param.item(), misfit.item()])

    return np.minimum(fCeil, residual)
            
try:
    output1 = dfols.solve(res, x0=InitialGuess, bounds=boundsParams, scaling_within_bounds=True)    
    dfols_solution = output1.x  # Returns the optimal parameter vector
    print("Reached part 3: DFO-LS optimization finished.")

except Exception as e:
    print("Error in DFOLS.")
    dfols_solution = np.array([-1.])  # Default value in case of failure
        
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