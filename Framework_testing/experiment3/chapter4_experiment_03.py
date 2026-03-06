"""
===============================================================
Chapter 4 — Experiment 03
Sin–cos ODE — Weighted residuals and single-parameter fit
---------------------------------------------------------------
    This executable is a version of the computational workflow reported in
    Chapter 5, Section 4.1 of the thesis, and is included as supplementary
    material in Appendix A.
------------------------------------------------------------------------
This executable performs:
    (1) Definition of a simple harmonic oscillator with analytical
        solution (sin, cos)
    (2) Construction of two residual components (res1, res2)
        and weighted misfit combining res1 and res2
    (3) Observation generation
    (4) Heuristic search
    (5) DFO-LS optimization
    (6) Structured JSON output and PDF reporting

Author:
    Letícia Becher Yamashita
===============================================================
"""
# ===============================================================
# 1. IMPORTS
# ===============================================================

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import json
import numpy as np
from scipy.integrate import solve_ivp
import dfols
from typing import List, Tuple

import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
from typing import Dict, Any

# ===============================================================
# 2. EXECUTION CONFIGURATION
# ===============================================================
# Output files
output_pdf = "Chapter4_Experiment03_Report.pdf"
output_json = "Chapter4_Experiment03.jsonl"

print(f"[Executing: Chapter 4 - Experiment 03]")

# ===============================================================
# 3. ADJUSTABLE SETTINGS
# ===============================================================
# Observational parameter
aObs = 1.0
ObsParams = np.array([aObs], dtype=float)

# Heuristic search grid for the parameter a
a_min = 0.001
a_max = 2.0 * np.pi - 0.001
grid_size = 11 # Number of heuristic evaluations

# Number of samples per cycle
N = 10

# Weighting factors
alpha = 0.5 
m1 = 5 # dimension of res1
m2 = 3 # dimension of res2

print(f"\n[Running Experiment 03, with N={N} samples, alpha={alpha}, m1={m1}, m2={m2}]")
                                 
# ===============================================================
# 3. GENERAL EXPERIMENT SETTINGS
# ===============================================================
# Bounds and initial guess for a
Boundsinf = np.array([a_min], dtype=float)
Boundsup = np.array([a_max], dtype=float)
boundsParams = (Boundsinf, Boundsup)
InitialGuess = np.array([np.pi], dtype=float)

# Parameter list for heuristic evaluation
a_values = np.linspace(a_min, a_max, grid_size)

# Initial time (cannot be zero, otherwise the system converges
# to the trivial solution in this construction).
tini = 1.0e-3

# Approximate length of one oscillation cycle in physical time t
cyclesize = 2.0 * np.pi
ncycles = 200.0

# Total integration time used in the original notebook
tmax = ncycles*cyclesize

# Interval for the full integration
tspan = (tini, tini + tmax)

# ===============================================================
# 4. ODE SYSTEM, NUMERICAL INTEGRATION & OBSERVATION GENERATION
# ===============================================================
def system(t: float, y: np.ndarray, parameters: np.ndarray) -> List[float]:
    """
    Harmonic oscillator system:

        x' = a v
        v' = -a x

    Parameters
    ----------
    t : float
        Time.
    y : ndarray, shape (2,)
        State [x, v].
    parameters : ndarray, shape (1,)
        Parameter array containing a.

    Returns
    -------
    [dx/dt, dv/dt]
    """
    x, v = y
    a = float(parameters[0])
    dxdt = a * v
    dvdt = -a * x
    return [dxdt, dvdt]

def system_exact_solution(t, parameter_a: np.ndarray) -> np.ndarray:
    """
    Analytical solution used in the notebook:

        x(t) = sin(a t)
        v(t) = cos(a t)

    Works for scalar or numpy-array t.

    Parameters
    ----------
    t : float or ndarray
        Time(s).
    parameters : ndarray, shape (1,)
        Parameter array containing a.

    Returns
    -------
    y : ndarray, shape (2, len(t))
        State [x, v].
    """
    t_arr = np.asarray(t, dtype=float)
    x = np.sin(parameter_a * t_arr)
    v = np.cos(parameter_a * t_arr)
    return np.vstack([x, v])

def integrate_ode(system, parameter, tmax, NN, y0):
    """Integrates the ODE system over tmax cycles using adaptive tolerance."""
    tspan = (0,tmax)
    AuxTol = 1e-12
    tstore = np.linspace(tmax-cyclesize, tmax, NN+1)

    while AuxTol < 1e-3:
        try:
            sol = solve_ivp(
                system, tspan, y0, args=(parameter,),
                method="BDF", t_eval=tstore,
                rtol=AuxTol, atol=AuxTol
            )
            if sol.success and np.all(np.isfinite(sol.y)):
                return sol
            
        except Exception as e:
            print("Integration exception:", repr(e), "tol=", AuxTol, "param=", parameter)
        
        AuxTol *= 100.0
    
    print("Integration failure.")
    return None

def system_predicted_solution(system, parameters, tmax, NN,y0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrates the system and returns the solution in DFO-LS-compatible form.
    """
    sol = integrate_ode(system, parameters, tmax, NN, y0)    
    return sol.y

def generate_observations(Obsparams: np.ndarray, NN: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates observations from the analytical solution.
    Returns array shape (2, NN+1).
    """
    tmax = ncycles * cyclesize
    tstore_ini = (ncycles - 1) * cyclesize
    t_obs = np.linspace(tstore_ini, tmax, NN + 1)

    y_obs = np.zeros((2, t_obs.size))
    for i, s in enumerate(t_obs):
        y_obs[:, i] = system_exact_solution(s, Obsparams).ravel()
    return t_obs, y_obs

# -----------------------------------------------------------
# 4.1 Generate observations from exact solution
# -----------------------------------------------------------
t_obs, y_obs = generate_observations(ObsParams, N)

# Initial condition for integration
y0 = system_exact_solution(0, ObsParams).ravel()

# ===============================================================
# 5. Misfit definition
# ===============================================================
def residual_vector(param_a: float) -> np.ndarray:
    """
    Residual at each time point is the Euclidean norm between model
    and observations: r_i = || y_model_i - y_obs_i ||_2.
    """
    param = np.array([param_a], dtype=float)
    y0 = system_exact_solution(0, param).ravel()

    sol = integrate_ode(system, param, tmax, N, y0)

    model_vals = sol.y 
    residuals = np.linalg.norm(model_vals - y_obs, axis=0)
    return residuals

def misfit(param_a: float) -> float:
    """
    Scalar misfit for given 'a':

        J(a) = 0.5 * ||r||_2

    where r is the residual vector.
    """
    r = residual_vector(param_a)
    return 0.5 * np.linalg.norm(r)

# ===============================================================
# 6. HEURISTIC SEARCH
# ===============================================================
# Parallel evaluation
def list_misfit(param):    
    return [param,misfit(param)]

def get_n_workers():
    total = os.cpu_count()
    # use 75% of CPU cores, minimum 1, maximum total-1
    workers = max(1, min(total - 1, int(total * 0.75)))
    return workers

with ProcessPoolExecutor(max_workers=get_n_workers()) as executor:
    futures = [executor.submit(list_misfit, p) for p in a_values]
    results = [f.result() for f in as_completed(futures)]

results = np.array(results)
order = np.argsort(results, axis=0)[:,0]
f_list = results[order,1]

MisfitHeuristic = [a_values.tolist(),f_list.tolist()]

print("Reached part 1: Heuristic finished.")

min_index = np.argmin(f_list)
InitialGuess = a_values[min_index]
f_InitialGuess = f_list[min_index]

# Save heuristic results
results_dict = {
    "ObsParams": ObsParams.tolist(),
    "y0": y0.tolist(),
    "MisfitHeuristic": MisfitHeuristic,
    "InitialGuess": InitialGuess.item(),
    "alpha": alpha,
    "m1": m1,
    "m2": m2,
    "N": N
}

with open(output_json, "a") as f: 
    f.write(json.dumps(results_dict) + "\n")

# ===============================================================
# 7. DFO-LS OPTIMIZATION
# ===============================================================
# DFOLS evaluation history:
# Each entry has the form [a, misfit], allowing
# reconstruction of the convergence path for analysis and plotting.
dfols_history = []

def res(param_array):
    """Residual function passed to DFOLS."""
    param = param_array[0]
    ode_solution = system_predicted_solution(system, np.array(param, dtype=float).ravel(), tmax, N, y0)
    if ode_solution is None:
        return np.ones(N) * 1e12   # prevents crash
    
    residual = np.linalg.norm((ode_solution - y_obs), axis=0)

    misfit = 0.5*np.linalg.norm(residual)
    dfols_history.append([param.item(), misfit.item()])

    return residual
            
try:
    output1 = dfols.solve(res, x0=np.array([InitialGuess]), bounds=boundsParams, scaling_within_bounds=True, maxfun=20)    
    dfols_solution = output1.x  # Returns the optimal parameter vector
    print("Reached part 2: DFO-LS optimization finished.")

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

print("Execution completed successfully. Generating report.")

# ===============================================================
# REPORT GENERATION
# ===============================================================
# -----------------------------
# Helpers
# -----------------------------
def _fig_to_image(fig: plt.Figure) -> np.ndarray:
    """Convert Matplotlib figure to an image array (for PDF embedding)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    return img

def _safe_np(a, dtype=float):
    if a is None:
        return None
    return np.array(a, dtype=dtype)

def _merge_jsonl_objects(jsonl_file: str) -> Dict[str, Any]:
    """Merge all dict lines in a JSONL file into one dict (last write wins)."""
    with open(jsonl_file, "r") as f:
        raw_lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if not raw_lines:
        raise ValueError(f"Empty JSONL file: {jsonl_file}")

    merged: Dict[str, Any] = {}
    for ln in raw_lines:
        obj = json.loads(ln)
        if isinstance(obj, dict):
            merged.update(obj)
    return merged

# REPORT GENERATION
def generate_experiment_report(
    json_file: str,
    output_pdf: str
) -> None:
    """
    Generates a 2-page A4 PDF summarizing the experiment.

    Experiment 03 (1 parameter):
      - MisfitHeuristic: [[a_grid...], [J_grid...]]
      - InitialGuess: float
      - DFOLSiterations: [[a, misfit], ...]
      - DFOLSoutput: [a_opt]
      - Bounds: [[a_min], [a_max]]

    Optional time-series:
      - t_obs, y_obs (2,n)
      - t_model, y_model_opt (2,n)
    """

    data = _merge_jsonl_objects(json_file)

    # PDF font options (text copyable)
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    obs_params = _safe_np(data.get("ObsParams", None), float)
    if obs_params is not None:
        obs_params = obs_params.ravel()
    
    title = "Chapter 4 - ODE Calibration - Experiment 03"
    
    # ---- Parse heuristic
    a_grid = _safe_np(data["MisfitHeuristic"][0], float)
    J_grid = _safe_np(data["MisfitHeuristic"][1], float)
    initial_guess = float(data.get("InitialGuess", np.nan))

    # ---- Parse DFOLS
    hist = _safe_np(data["DFOLSiterations"], float)
    a_path = hist[:, 0]
    J_path = hist[:, 1]
    it = np.arange(len(a_path))

    a_opt = float(np.array(data["DFOLSoutput"], dtype=float).ravel()[0])

    bnd = data["Bounds"]
    bounds_txt = f"[{float(bnd[0][0]):.3g}, {float(bnd[1][0]):.3g}]"

    # ---- Optional time-series data
    t_obs = _safe_np(data.get("t_obs", None), float)
    y_obs = _safe_np(data.get("y_obs", None), float)      # (2, n)
    # t_model = _safe_np(data.get("t_model", None), float)
    y_model_opt = _safe_np(data.get("y_model_opt", None), float)

    # ------------------ Figures (legacy)
    def fig_misfit_curve() -> plt.Figure:
        fig = plt.figure(figsize=(5.2, 4.0))
        ax = plt.gca()
        ax.plot(a_grid, J_grid, marker="o", markersize=3, linestyle="-", label="heuristic")
        if np.isfinite(initial_guess):
            idx = int(np.argmin(np.abs(a_grid - initial_guess)))
            ax.scatter([a_grid[idx]], [J_grid[idx]], s=45, zorder=3, label="initial guess")
        ax.set_xlabel("Parameter a")
        ax.set_ylabel("Misfit J(a)")
        ax.set_title("Heuristic misfit curve")
        ax.grid(alpha=0.4)
        ax.legend(loc="best")
        return fig

    def fig_dfols_convergence_a() -> plt.Figure:
        fig = plt.figure(figsize=(5.2, 4.0))
        ax = plt.gca()
        ax.plot(it, a_path, marker="o", linestyle="-", label="a (DFO-LS path)")
        if a_opt is not None:
            ax.axhline(a_opt, linestyle="--", label=f"a* = {a_opt:.6g}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Parameter a")
        ax.set_title("DFO-LS convergence (parameter a)")
        ax.grid(alpha=0.4)
        ax.legend(loc="best")
        return fig

    def fig_dfols_misfit() -> plt.Figure:
        fig = plt.figure(figsize=(5.2, 4.0))
        ax = plt.gca()
        ax.plot(it, J_path, marker="o", linestyle="-")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Misfit J")
        ax.set_title("DFO-LS misfit decrease")
        ax.grid(alpha=0.4)
        return fig

    def fig_timeseries() -> plt.Figure:
        fig = plt.figure(figsize=(10.5, 6.0))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        x_obs, yobs2 = y_obs[0, :], y_obs[1, :]
        ax1.plot(t_obs, x_obs, marker="o", linestyle="-", label="obs x")
        ax2.plot(t_obs, yobs2, marker="o", linestyle="-", label="obs y")
        ax1.set_xlabel("t")
        ax2.set_xlabel("t")

        x_opt, y_opt2 = y_model_opt[0, :], y_model_opt[1, :]
        ax1.plot(t_obs, x_opt, marker="o", linestyle="-", label="model x (a*)")
        ax2.plot(t_obs, y_opt2, marker="o", linestyle="-", label="model y (a*)")

        ax1.set_title("Time-series comparison")
        ax1.set_ylabel("x")
        ax2.set_ylabel("y")
        ax1.grid(alpha=0.35)
        ax2.grid(alpha=0.35)
        ax1.legend(loc="best")
        ax2.legend(loc="best")
        fig.tight_layout()
        return fig

    IMG1 = _fig_to_image(fig_misfit_curve())
    IMG2 = _fig_to_image(fig_dfols_convergence_a())
    IMG3 = _fig_to_image(fig_dfols_misfit())
    IMG4 = _fig_to_image(fig_timeseries())

    with PdfPages(output_pdf) as pdf:
        # Page 1
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        summary = f"{title}\nData source: {json_file}"
        summary += f"\nObservational parameter: a_Obs = {float(obs_params[0]):.6g}"
        if np.isfinite(initial_guess):
            summary += f"\nHeuristic initial guess: a0 = {initial_guess:.6g}"
        if a_opt is not None:
            summary += f"\nDFO-LS optimum: a* = {a_opt:.10f}"
        if bounds_txt:
            summary += f"\nBounds: {bounds_txt}"

        fig.text(0.05, 0.94, summary, va="top", family="monospace", fontsize=11)

        ax1 = fig.add_axes([0.07, 0.50, 0.40, 0.38])
        ax1.imshow(IMG1); ax1.axis("off")
        ax2 = fig.add_axes([0.53, 0.50, 0.40, 0.38])
        ax2.imshow(IMG2); ax2.axis("off")
        ax3 = fig.add_axes([0.18, 0.10, 0.64, 0.32])
        ax3.imshow(IMG3); ax3.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        fig.text(0.05, 0.95, "Model vs observations (x(t), y(t))",
                    va="top", family="monospace", fontsize=11)
        ax4 = fig.add_axes([0.07, 0.08, 0.86, 0.84])
        ax4.imshow(IMG4); ax4.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

    return

# ---------------------------------------------------------------
#  EXTRA DATA FOR FULL REPORT (time-series)
# ---------------------------------------------------------------
a_opt = float(np.array(dfols_solution, dtype=float).ravel()[0])

sol_opt = integrate_ode(system, np.array([a_opt], dtype=float), tmax, N, y0)

extra_report = {
    "t_obs": t_obs.tolist(),
    "y_obs": y_obs.tolist(),

    "t_model": sol_opt.t.tolist(),
    "y_model_opt": sol_opt.y.tolist()
}

with open(output_json, "a") as f:
    f.write(json.dumps(extra_report) + "\n")

generate_experiment_report(output_json, output_pdf)
print(f"[Report] PDF saved -> {output_pdf}")