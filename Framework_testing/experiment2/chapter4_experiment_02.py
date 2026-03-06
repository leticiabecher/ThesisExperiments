"""
===============================================================
Chapter 4 - Experiment 02
2D ODE - Two-Parameter Estimation (a, b)
---------------------------------------------------------------
    This executable is a version of the computational workflow reported in
    Chapter 5, Section 4.1 of the thesis, and is included as supplementary
    material in Appendix A.
------------------------------------------------------------------------
This executable performs:
    (1) Observation generation
    (2) Heuristic search
    (3) DFO-LS optimization
    (4) Structured JSON output and PDF reporting

Author:
    Letícia Becher Yamashita
===============================================================
"""
# ===============================================================
# 1. IMPORTS
# ===============================================================
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
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

# Optional (contour interpolation)
try:
    from scipy.interpolate import griddata  # type: ignore
except Exception:  # scipy might be unavailable in some environments
    griddata = None

# ===============================================================
# 2. EXECUTION CONFIGURATION
# ===============================================================

# Output files
output_pdf = "Chapter4_Experiment02_Report.pdf"
output_json = "Chapter4_Experiment02.jsonl"

print(f"[Executing: Chapter 4 - Experiment 02]")

# ===============================================================
# 3. ADJUSTABLE SETTINGS
# ===============================================================
# Observational parameters
aObs = 2.5
bObs = 3.5
ObsParameters = np.array([aObs, bObs])

# Number of samples per cycle
N = 10

# Heuristic grid size
grid_size_a = 3
grid_size_b = 3

print(f"\n[Running Experiment 02, with N={N} samples]")

# ===============================================================
# 3. GENERAL EXPERIMENT SETTINGS
# ===============================================================
# Time / cycles
cyclesize = 2.0
ncycles = 200 # Default total number of cycles for spin-up
b_fixed = 1e12 # fixed parameter
tmax = ncycles * cyclesize #last integration time

# Bounds for (a, b) around the observational parameters
a_span = 3.0
b_span = 3.0
bounds_inf = np.array([aObs - a_span, bObs - b_span], dtype=float)
bounds_up = np.array([aObs + a_span, bObs + b_span], dtype=float)
boundsParams = (bounds_inf, bounds_up)

# Heuristic search grid
a_values = np.linspace(bounds_inf[0], bounds_up[0], grid_size_a)
b_values = np.linspace(bounds_inf[1], bounds_up[1], grid_size_b)

# ===============================================================
# 4. ODE SYSTEM, NUMERICAL INTEGRATION & OBSERVATION GENERATION
# ===============================================================
def system(s: float, y: np.ndarray, parameters: np.ndarray) -> List[float]:
    """
    Two-dimensional ODE system for Experiment 02.

    Parameters
    ----------
    s : float
        Adimensional time variable.
    y : ndarray, shape (2,)
        Current state [x, v].
    parameters : ndarray, shape (2,)
        Model parameters [a, b].

    Returns
    -------
    [dx/ds, dv/ds]
    """
    x, v = y
    a = float(parameters[0])
    b = float(parameters[1])

    t = np.pi * s
    u = 1.0 / (t + 1e-4)

    dxdt = -v - (u**2) / (np.exp(a) + u) + np.log(np.exp(b) + u)
    dvdt = x - (u**2) / (np.exp(b) + u) - np.log(np.exp(a) + u)

    dxds = np.pi * dxdt
    dvds = np.pi * dvdt

    return [dxds, dvds]

def system_exact_solution(s: float, parameters: np.ndarray) -> np.ndarray:
    """
    Analytical solution used in the notebook for Experiment 02.

    Parameters
    ----------
    s : float
        Adimensional time.
    parameters : ndarray, shape (2,)
        Parameters [a, b].

    Returns
    -------
    y : ndarray, shape (2,)
        State [x(s), y(s)].
    """
    a = float(parameters[0])
    b = float(parameters[1])

    t = np.pi * s
    u = 1.0 / (t + 1e-4)

    x = np.log(np.exp(a) + u) + np.cos(t)
    y = np.log(np.exp(b) + u) + np.sin(t)

    return np.array([x, y], dtype=np.float64)

def integrate_ode(system, parameters, tmax, NN, y0):
    """Integrates the ODE system over tmax cycles using adaptive tolerance."""
    tspan = (0,tmax)
    AuxTol = 1e-12
    tstore = np.linspace(tmax-cyclesize, tmax, NN+1)

    while AuxTol < 1e-3:
        try:
            sol = solve_ivp(
                system, tspan, y0, args=(parameters,),
                method="BDF", t_eval=tstore,
                rtol=AuxTol, atol=AuxTol
            )
            if sol.success and np.all(np.isfinite(sol.y)):
                return sol
            
        except Exception as e:
            print("Integration exception:", repr(e), "tol=", AuxTol, "params=", parameters)
        
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
        y_obs[:, i] = system_exact_solution(s, Obsparams)

    return t_obs, y_obs

# -----------------------------------------------------------
# 4.1 Generate observations from exact solution
# -----------------------------------------------------------
t_obs, y_obs = generate_observations(ObsParameters, N)

# Initial condition for integration
y0 = system_exact_solution(0, ObsParameters)

# ===============================================================
# 5. Misfit definition
# ===============================================================

def residual_vector(params: np.ndarray) -> np.ndarray:
    """
    Residual vector for given '[a,b]' parameters.

    Residual at each time point is the Euclidean norm between model
    and observations: r_i = || y_model_i - y_obs_i ||_2.

    Parameters
    ----------
    a_value : float
        Candidate value of parameter 'a'.
    b_fixed : float
        Candidate value of parameter 'b'.
    t_obs : ndarray
        Observation times (s).
    y_obs : ndarray, shape (2, len(t_obs))
        Observed states.
    ncycles, cyclesize
        Integration settings.

    Returns
    -------
    residuals : ndarray, shape (len(t_obs),)
        Residual vector.
    """
    y0 = system_exact_solution(0, params)

    sol = integrate_ode(system, params, tmax, N, y0)

    model_vals = sol.y 
    residuals = np.linalg.norm(model_vals - y_obs, axis=0)
    return residuals

def misfit(params: float) -> float:
    """
    Scalar misfit for given 'a' and fixed 'b':

        J(a) = 0.5 * ||r||_2

    where r is the residual vector.
    """
    r = residual_vector(params)
    return 0.5 * np.linalg.norm(r)

# ===============================================================
# 6. HEURISTIC SEARCH
# ===============================================================
# Parallel evaluation
def grid_misfit(args):
    """Wrapper used for parallel execution of misfitH."""    
    i, j, a_values, b_values = args
    params = np.array([a_values[i], b_values[j]])
    return (i, j, misfit(params))

# Parameter grid
fGrid = -1*np.ones((len(a_values), len(b_values)))

indices = itertools.product(range(len(a_values)), range(len(b_values)))
args = [(i, j, a_values, b_values) for i, j in indices]

# Parallel evaluation
def get_n_workers():
    total = os.cpu_count()
    # use 75% of CPU cores, minimum 1, maximum total-1
    workers = max(1, min(total - 1, int(total * 0.75)))
    return workers

with ProcessPoolExecutor(max_workers=get_n_workers()) as executor:
    futures = [executor.submit(grid_misfit, arg) for arg in args]
    results = [f.result() for f in as_completed(futures)]

# Fill misfit grid
for i, j, misfit in results:
    fGrid[i, j] = misfit

IndicesH = np.argwhere(fGrid >= 0)
f_list = fGrid[IndicesH[:, 0], IndicesH[:, 1]]
param_list = np.column_stack((a_values[IndicesH[:, 0]], b_values[IndicesH[:, 1]]))

MisfitHeuristic = [param_list.tolist(),f_list.tolist()]

print("Reached part 1: Heuristic finished.")

min_index = np.argmin(f_list)
InitialGuess = param_list[min_index]
f_InitialGuess = f_list[min_index]

# Save heuristic results
results_dict = {
    "y0": y0.tolist(),
    "MisfitHeuristic": MisfitHeuristic,
    "InitialGuess": InitialGuess.tolist()
}

with open(output_json, "a") as f: 
    f.write(json.dumps(results_dict) + "\n")

# ===============================================================
# 7. DFO-LS OPTIMIZATION
# ===============================================================

# DFOLS evaluation history:
# Each entry has the form [[a,b], misfit], allowing
# reconstruction of the convergence path for analysis and plotting.
dfols_history = []

def res(params):
    """Residual function passed to DFOLS."""
    ode_solution = system_predicted_solution(system, np.array(params, dtype=float).ravel(), tmax, N, y0)
    if ode_solution is None:
        return np.ones(N) * 1e12   # prevents crash
    
    residual = np.linalg.norm((ode_solution - y_obs), axis=0)

    misfit = 0.5*np.linalg.norm(residual)
    dfols_history.append([params.tolist(), misfit.item()])

    return residual

try:
    output1 = dfols.solve(res, x0=np.array(InitialGuess), bounds=boundsParams, scaling_within_bounds=True, maxfun=20)    
    dfols_solution = output1.x  # Returns the optimal parameter vector
    print("Reached part 2: DFO-LS optimization finished.")

except Exception as e:
    print(f"Error in DFOLS.")
    dfols_solution = np.array([-1.])  # Default value in case of failure

# Save DFOLS results
results_dict = {
    "DFOLSiterations": dfols_history,
    "DFOLSoutput": dfols_solution.tolist(),
    "Bounds": [bounds_inf.tolist(), bounds_up.tolist()]
}

with open(output_json, "a") as f:
    f.write(json.dumps(results_dict) + "\n")

print("Execution completed successfully.")

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

    Experiment 02 (2 parameters):
      - MisfitHeuristic: [ [[a,b],...], [misfit,...] ]
      - InitialGuess: [a0, b0]
      - DFOLSiterations: [ [[a,b], misfit], ... ]   (ragged)
      - DFOLSoutput: [a_opt, b_opt]
      - Bounds: [[a_min,b_min], [a_max,b_max]]
      - (optional) ObsParams: [a_obs, b_obs]   # if saved by experiment_02

    Optional time-series (both):
      - t_obs, y_obs (2,n)
      - t_model, y_model_opt (2,n)
    """

    data = _merge_jsonl_objects(json_file)

    # PDF font options (text copyable)
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    # Observational parameters (if saved in JSONL)
    obs_params = _safe_np(data.get("ObsParams", None), float)
    if obs_params is not None:
        obs_params = obs_params.ravel()
    
    title = "Chapter 4 - ODE Calibration - Experiment 02"

    param_list = _safe_np(data["MisfitHeuristic"][0], float)  # (m,2)
    f_list = _safe_np(data["MisfitHeuristic"][1], float)      # (m,)

    initial_guess = _safe_np(data.get("InitialGuess", None), float)
    initial_guess = initial_guess.ravel()

    hist_raw = data["DFOLSiterations"]

    params_path = np.array([row[0] for row in hist_raw], dtype=float)  # (n,2)
    J_path = np.array([row[1] for row in hist_raw], dtype=float)       # (n,)
    it = np.arange(len(J_path))

    a_path = params_path[:, 0]
    b_path = params_path[:, 1]

    # ---- DFOLS output
    a_opt = None
    b_opt = None

    out = np.array(data["DFOLSoutput"], dtype=float).ravel()
    a_opt = float(out[0])
    b_opt = float(out[1])

    # ---- Bounds
    bounds_txt = ""
    bounds_arr = None

    bnd = data["Bounds"]
    lo = np.array(bnd[0], dtype=float).ravel()
    hi = np.array(bnd[1], dtype=float).ravel()
    bounds_arr = np.vstack([lo, hi])  # (2,2)
    bounds_txt = f"a∈[{lo[0]:.3g},{hi[0]:.3g}], b∈[{lo[1]:.3g},{hi[1]:.3g}]"

    # ---- Optional time-series data
    t_obs = _safe_np(data.get("t_obs", None), float)
    y_obs = _safe_np(data.get("y_obs", None), float)
    t_model = _safe_np(data.get("t_model", None), float)
    y_model_opt = _safe_np(data.get("y_model_opt", None), float)

    # ------------------ Figures (Exp02)
    def fig_heuristic_2d() -> plt.Figure:
        """Contour-style plot from scattered (a,b)->misfit samples (like the 2D template)."""
        fig = plt.figure(figsize=(5.2, 5.0))
        ax = plt.gca()

        # Grid limits
        if bounds_arr is not None:
            a_lo, b_lo = float(bounds_arr[0, 0]), float(bounds_arr[0, 1])
            a_hi, b_hi = float(bounds_arr[1, 0]), float(bounds_arr[1, 1])
        else:
            a_lo, b_lo = float(np.min(param_list[:, 0])), float(np.min(param_list[:, 1]))
            a_hi, b_hi = float(np.max(param_list[:, 0])), float(np.max(param_list[:, 1]))

        # Interpolate -> contourf (fallback to scatter if griddata is unavailable/fails)
        if griddata is not None:
            try:
                gx, gy = np.mgrid[a_lo:a_hi:200j, b_lo:b_hi:200j]
                gz = griddata(param_list, f_list, (gx, gy), method="linear")
                cf = ax.contourf(gx, gy, gz, levels=20, cmap="viridis")
                fig.colorbar(cf, ax=ax, label="Misfit")
            except Exception:
                sc = ax.scatter(param_list[:, 0], param_list[:, 1], c=f_list, s=18, cmap="viridis")
                fig.colorbar(sc, ax=ax, label="Misfit")
        else:
            sc = ax.scatter(param_list[:, 0], param_list[:, 1], c=f_list, s=18, cmap="viridis")
            fig.colorbar(sc, ax=ax, label="Misfit")

        # Sample points overlay (subtle)
        ax.scatter(param_list[:, 0], param_list[:, 1], s=6, alpha=0.35, color="red")

        # True/observational parameters lines (if available)
        if obs_params is not None and obs_params.size >= 2:
            ax.axvline(x=float(obs_params[0]), color="lightgreen")
            ax.axhline(y=float(obs_params[1]), color="lightgreen")

        ax.set_title("Heuristic search")
        ax.set_xlabel("Parameter a")
        ax.set_ylabel("Parameter b")
        ax.grid(alpha=0.2)
        return fig

    def fig_dfols_path_2d() -> plt.Figure:
        """DFO-LS path in (a,b) colored by iteration (like the template)."""
        fig = plt.figure(figsize=(5.2, 5.0))
        ax = plt.gca()

        sc = ax.scatter(a_path, b_path, c=it, s=80, alpha=0.8, edgecolor="white")
        cbar = fig.colorbar(sc, ax=ax, label="Iteration")
        ticks = np.unique(np.linspace(0, len(it) - 1, 6, dtype=int)) if len(it) > 1 else [0]
        cbar.set_ticks(ticks)

        # Initial (prefer InitialGuess if provided)
        if initial_guess is not None and initial_guess.size >= 2:
            a0, b0 = float(initial_guess[0]), float(initial_guess[1])
        else:
            a0, b0 = float(a_path[0]), float(b_path[0])

        # Optimized (prefer DFOLSoutput if provided)
        if a_opt is not None and b_opt is not None:
            a1, b1 = float(a_opt), float(b_opt)
        else:
            a1, b1 = float(a_path[-1]), float(b_path[-1])

        ax.scatter(a0, b0, s=150, edgecolor="black", label="Initial guess")
        ax.scatter(a1, b1, s=150, edgecolor="white", label="Optimized")

        ax.set_title("DFO-LS search")
        ax.set_xlabel("Parameter a")
        ax.set_ylabel("Parameter b")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        return fig

    def fig_fit_x() -> plt.Figure:
        fig = plt.figure(figsize=(8.0, 4.5))
        ax = plt.gca()

        ax.plot(t_obs, np.maximum(y_obs[0, :], 0), label="Observations")
        ax.plot(t_model, np.maximum(y_model_opt[0, :], 0), linestyle="dashed", label="Model output")

        ax.set_title("Fit: x(t)")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
        ax.legend(loc="best", frameon=True, framealpha=0.85)
        fig.tight_layout()
        return fig

    def fig_fit_y() -> plt.Figure:
        fig = plt.figure(figsize=(8.0, 4.5))
        ax = plt.gca()

        ax.plot(t_obs, np.maximum(y_obs[1, :], 0), label="Observations")
        ax.plot(t_model, np.maximum(y_model_opt[1, :], 0), linestyle="dashed", label="Model output")

        ax.set_title("Fit: y(t)")
        ax.set_xlabel("t")
        ax.set_ylabel("y")
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
        ax.legend(loc="best", frameon=True, framealpha=0.85)
        fig.tight_layout()
        return fig

    # Produce images for PDF embedding
    IMG1 = _fig_to_image(fig_heuristic_2d())
    IMG2 = _fig_to_image(fig_dfols_path_2d())
    IMG3 = _fig_to_image(fig_fit_x())
    IMG4 = _fig_to_image(fig_fit_y())

    with PdfPages(output_pdf) as pdf:
        # --------------------------------------------------------
        # PAGE 1 — TEXT + FIGURE 1 + FIGURE 2 (side by side)
        # --------------------------------------------------------
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        text = f"{title}\nData source: {json_file}\n\nOptimized parameters:"
        if a_opt is not None and b_opt is not None:
            text += f"\n  a* = {a_opt:.10f}\n  b* = {b_opt:.10f}"
        else:
            text += "\n  (missing DFOLSoutput in JSONL)"

        if obs_params is not None and obs_params.size >= 2:
            text += f"\n\nObservational parameters:" \
                    f"\n  a_obs = {float(obs_params[0]):.10f}" \
                    f"\n  b_obs = {float(obs_params[1]):.10f}"

        if initial_guess is not None and initial_guess.size >= 2:
            text += f"\n\nInitial guess:\n  a0 = {float(initial_guess[0]):.10f}\n  b0 = {float(initial_guess[1]):.10f}"

        if bounds_txt:
            text += f"\n\nBounds:\n  {bounds_txt}"

        fig.text(0.05, 0.92, text, va="top", family="monospace", fontsize=12)

        ax1 = fig.add_axes([0.05, 0.15, 0.44, 0.44])
        ax1.imshow(IMG1)
        ax1.axis("off")

        ax2 = fig.add_axes([0.51, 0.15, 0.44, 0.44])
        ax2.imshow(IMG2)
        ax2.axis("off")

        pdf.savefig(fig)
        plt.close(fig)

        # --------------------------------------------------------
        # PAGE 2 — FIGURE 3 (top) + FIGURE 4 (bottom)
        # --------------------------------------------------------
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        ax3 = fig.add_axes([0.10, 0.52, 0.80, 0.40])
        ax3.imshow(IMG3)
        ax3.axis("off")

        ax4 = fig.add_axes([0.10, 0.05, 0.80, 0.40])
        ax4.imshow(IMG4)
        ax4.axis("off")

        pdf.savefig(fig)
        plt.close(fig)
    return
# ---------------------------------------------------------------
#  EXTRA DATA FOR FULL REPORT (time-series)
# ---------------------------------------------------------------
opt_params = np.array(dfols_solution, dtype=float)

sol_opt = integrate_ode(system, opt_params, tmax, N, y0)

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