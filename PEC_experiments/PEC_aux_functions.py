"""
========================================================================
Paranaguá Estuarine Complex (PEC) — Auxiliary Functions and Model Core
------------------------------------------------------------------------
Context:
    Supplementary material supporting the implementation of the
    conceptual nutrient–phytoplankton (NP) model and the calibration
    workflow described in the thesis:

        Yamashita, L. B. (2025). A Data-Constrained Framework for Marine
        Biogeochemistry Modeling with Applications to the Paranaguá
        Estuarine Complex. PhD Thesis, Universidade Federal do Paraná.

    This module centralizes:
        (i)  data loading and preprocessing utilities;
        (ii) construction of cyclic environmental forcing series;
        (iii) PEC two-box NP model ODE right-hand side definitions;
        (iv) numerical integration wrappers with robustness safeguards;
        (v) helper routines used by computational experiments and reports.

Purpose:
    Provide a single, reusable implementation layer for the experiment
    scripts presented in Appendix A (Implementation and Computational
    Experiments).

Notes:
    - The dataset is expected as a JSON Lines file named "dataset.jsonl"
      in the working directory.
    - The annual cycle is represented as 360 days (12 months × 30 days),
      consistent with the numerical experiments reported in the thesis.

Author:
    Letícia Becher Yamashita
========================================================================
"""

# ===============================================================
# 1. IMPORTS AND GLOBAL CONSTANTS
# ===============================================================

import numpy as np
import scipy.integrate as scipyInt
import json
import os

# Dataset
data_json = "dataset.jsonl"

# Annual cycle (360 days), month (30 days)
cyclesize = 360
monthsize = 30

# Fixed parameters
KN = 0.1
VolBox=1.7*5.7*105.4*1e+6

MODEL_MODE = "2param" # For experiments calibrating Vmax and Lambda
# MODEL_MODE = "1param" # For experiments calibrating Lambda
# MODEL_MODE = "Nitrate_param" # For experiments calibrating w

# ===============================================================
# 2. DATA LOADING
# ===============================================================

DATA = {}
with open(data_json, 'r') as file:
    for line in file:
        data = json.loads(line)
        for key, value in data.items():
            DATA[key] = np.array(value)

# ===============================================================
# 3. UTILITY FUNCTIONS
# ===============================================================

def data_smoother(arr):
    """Smooths time series vectors using a moving average."""
    new_arr = arr.copy()
    auxSize = len(arr)

    new_arr[0] = (arr[0] + arr[-1] + arr[-2]) / 3
    new_arr[1] = (arr[1] + arr[0] + arr[-1]) / 3
    
    for i in range(2,auxSize):
        new_arr[i] = (arr[i] + arr[i-1] + arr[i-2]) / 3

    return new_arr

def make_cyclic_series(arr,smooth):
    """
    Converts a monthly series starting in June into an annual cyclic series
    starting in January. Used to make observational data compatible.
    """
    aux0 = np.array([arr[i] for i in range(6,12)])
    aux0[:,0] -= 6
    aux1 = np.array([arr[i] for i in range(6)])
    aux1[:,0] += 6
    
    out = np.concatenate((aux0, aux1), axis=0)
    if smooth==True:
        out = data_smoother(out)
    return np.concatenate((out, [[out[0,0] + 12., out[0,1]]]), axis=0)

def weighted_mean(values, weights):
    """Weighted mean used across several functions."""
    values = np.array(values)
    return np.sum(values * weights) / np.sum(weights)

def _prepare_xy(xy):
    """
    Receives an array Nx2 (col0 = day of year, col1 = value).
    Returns (unique_x, averaged_y) sorted, with averages for repeated days.
    Used for interpolation.
    """
    x = xy[:, 0].astype(float)
    y = xy[:, 1].astype(float)

    x_unique, inv = np.unique(x, return_inverse=True)
    y_sum = np.bincount(inv, weights=y)
    y_count = np.bincount(inv)
    y_mean = y_sum / y_count

    return x_unique, y_mean

def make_linear_interpolator(xy):
    """Creates a simple linear interpolator."""
    x_u, y_m = _prepare_xy(xy)

    def f(t):
        return np.interp(t, x_u, y_m, left=y_m[0], right=y_m[-1])

    return f

# ===============================================================
# 4. ORGANIZATION OF MODEL INPUT DATA SERIES
# ===============================================================

PV = DATA["WeightsVolume"]

def build_series(name_base,smooth=False):
    """Builds cyclic series for I, II, III stations."""
    return (
        make_cyclic_series(DATA[name_base + "I"],smooth),
        make_cyclic_series(DATA[name_base + "II"],smooth),
        make_cyclic_series(DATA[name_base + "III"],smooth)
    )

TempSI, TempSII, TempSIII = build_series("TempS")
SalSI,  SalSII,  SalSIII  = build_series("SalS")
SalBI,  SalBII,  SalBIII  = build_series("SalB")
ChlorBI, ChlorBII, ChlorBIII = build_series("ChlorophyllB",smooth=True)
NitBI,  NitBII,  NitBIII  = build_series("NitrateB",smooth=True)
ChlorSI, ChlorSII, ChlorSIII = build_series("ChlorophyllS",smooth=True)
NitSI,  NitSII,  NitSIII  = build_series("NitrateS",smooth=True)

# ===============================================================
# 5. MODEL ENVIRONMENTAL VARIABLES
# ===============================================================

def Temp(t):
    """Weighted-average temperature of the upper box."""
    tt = np.remainder(t, cyclesize)/monthsize
    vals = [
        np.interp(tt, TempSI[:,0], TempSI[:,1]),
        np.interp(tt, TempSII[:,0], TempSII[:,1]),
        np.interp(tt, TempSIII[:,0], TempSIII[:,1])
    ]        
    return weighted_mean(vals, PV)

def Sal(t):
    """Weighted-average salinity of the upper box."""
    tt = np.remainder(t, cyclesize)/monthsize
    vals = [
        np.interp(tt, SalSI[:,0], SalSI[:,1]),
        np.interp(tt, SalSII[:,0], SalSII[:,1]),
        np.interp(tt, SalSIII[:,0], SalSIII[:,1])
    ]
    return weighted_mean(vals, PV)

def SalLower(t):
    """Weighted-average salinity of the lower box."""
    tt = np.remainder(t, cyclesize)/monthsize
    vals = [
        np.interp(tt, SalBI[:,0], SalBI[:,1]),
        np.interp(tt, SalBII[:,0], SalBII[:,1]),
        np.interp(tt, SalBIII[:,0], SalBIII[:,1])
    ]
    return weighted_mean(vals, PV)

# ===============================================================
# Hydrodynamic Fluxes
# ===============================================================
# Multiplied by 200 to enforce ~3-day estuarine residence time
# using mean Qebm divided by the total volume of both boxes.
# Additional constants convert seconds to days.
sampleQriver = 200*(12*60*60)*4*np.array(DATA["SampleQriver"])
sampleNriver = np.array(DATA["SampleNriver"])

def Qriver(t):
    """Riverine and pluvial inflow flux."""
    tt = np.remainder(t, cyclesize)/monthsize
    return np.interp(tt, np.linspace(0, 12, 13), sampleQriver)

def Nriver(t):
    """Nitrate concentration associated with the riverine inflow."""
    tt = np.remainder(t, cyclesize)/monthsize
    return np.interp(tt, np.linspace(0, 12, 13), sampleNriver)

def Qocean(t):
    """Oceanic inflow flux computed relative to Qriver and salinity gradients."""
    Su = Sal(t)
    Sl = SalLower(t)
    denom = Sl - Su
    if abs(denom) < 1e-6:
        denom = 1e-6
    return Qriver(t) * Su / denom

def Qebm(t):
    """Outflow flux from the model domain."""
    return Qriver(t) + Qocean(t)

# ===============================================================
# Biological Factors
# ===============================================================
def alpha(t):
    """Reduction in phytoplankton growth rate due to 
    temperature fluctuations.""" 
    return max(0,min(1.,0.6+0.08*(30.-Temp(t)-10.)))

def beta(t):
    """Reduction in phytoplankton growth rate due to 
    salinity fluctuations."""
    return max(0,min(1.,1.-0.07*(20.-Sal(t))))

# ===============================================================
# 6. ODE SYSTEM, NUMERICAL INTEGRATION & OBSERVATION GENERATION
# ===============================================================

# ===============================================================
# Observed Chlorophyll and Nitrate Profiles
# ===============================================================

def ChlorophyllInf(t):
    """Weighted-average chlorophyll-A concentration in the lower box."""
    tt = np.remainder(t, cyclesize)/monthsize
    vals = [
        np.interp(tt, ChlorBI[:,0], ChlorBI[:,1]),
        np.interp(tt, ChlorBII[:,0], ChlorBII[:,1]),
        np.interp(tt, ChlorBIII[:,0], ChlorBIII[:,1])
    ]
    return weighted_mean(vals, PV)

def ChlorophyllUp(t):
    """Weighted-average chlorophyll-A concentration in the upper box."""
    tt = np.remainder(t, cyclesize)/monthsize
    vals = [
        np.interp(tt, ChlorSI[:,0], ChlorSI[:,1]),
        np.interp(tt, ChlorSII[:,0], ChlorSII[:,1]),
        np.interp(tt, ChlorSIII[:,0], ChlorSIII[:,1])
    ]
    return weighted_mean(vals, PV)
    
def NitrateInf(t):
    """Weighted-average nitrate concentration in the lower box."""
    tt = np.remainder(t, cyclesize)/monthsize
    vals = [
        np.interp(tt, NitBI[:,0], NitBI[:,1]),
        np.interp(tt, NitBII[:,0], NitBII[:,1]),
        np.interp(tt, NitBIII[:,0], NitBIII[:,1])
    ]
    return weighted_mean(vals, PV)

def NitrateUp(t):
    """Weighted-average nitrate concentration in the upper box."""
    tt = np.remainder(t, cyclesize)/monthsize
    vals = [
        np.interp(tt, NitSI[:,0], NitSI[:,1]),
        np.interp(tt, NitSII[:,0], NitSII[:,1]),
        np.interp(tt, NitSIII[:,0], NitSIII[:,1])
    ]
    return weighted_mean(vals, PV)

# ===============================================================
# ODE SYSTEM (N, P)
# ===============================================================

def system_2param(t,y,param):
    """
    Model ODE system:
        y[0] = Nitrate concentration in upper box.
        y[1] = Phytoplankton concentration in upper box.
    """
    Vmax, Lambda = param
    w = 0.588

    N = max(y[0], 0.0)
    P = max(y[1], 0.0)
    
    # dN/dt - nitrate concentration variation
    dN = w*Nriver(t)*Qriver(t)/VolBox+NitrateInf(t)*Qocean(t)/VolBox+0.7*Lambda*P-N*(P*alpha(t)*beta(t)*Vmax/(N+KN)+Qebm(t)/VolBox)

    # dP/dt - phytoplankton concentration variation
    dP = P*(N*alpha(t)*beta(t)*Vmax/(N+KN)-Lambda-Qebm(t)/VolBox) + ChlorophyllInf(t)*Qocean(t)/VolBox
    
    return [dN,dP]

def system_1param(t,y,param):
    """
    Model ODE system:
        y[0] = Nitrate concentration in upper box.
        y[1] = Phytoplankton concentration in upper box.
    """
    Vmax = Vmax2(t)
    Lambda = param[0]
    w = 0.588

    N = max(y[0], 0.0)
    P = max(y[1], 0.0)
    
    # dN/dt - nitrate concentration variation
    dN = w*Nriver(t)*Qriver(t)/VolBox+NitrateInf(t)*Qocean(t)/VolBox+0.7*Lambda*P-N*(P*alpha(t)*beta(t)*Vmax/(N+KN)+Qebm(t)/VolBox)

    # dP/dt - phytoplankton concentration variation
    dP = P*(N*alpha(t)*beta(t)*Vmax/(N+KN)-Lambda-Qebm(t)/VolBox) + ChlorophyllInf(t)*Qocean(t)/VolBox
    
    return [dN,dP]

def system_w_param(t,y,param):
    """
    Model ODE system:
        y[0] = Nitrate concentration in upper box.
        y[1] = Phytoplankton concentration in upper box.
    """
    Vmax = Vmax2(t)
    Lambda = 0.05
    w = param[0]

    N = max(y[0], 0.0)
    P = max(y[1], 0.0)
    
    # dN/dt - nitrate concentration variation
    dN = w*Nriver(t)*Qriver(t)/VolBox+NitrateInf(t)*Qocean(t)/VolBox+0.7*Lambda*P-N*(P*alpha(t)*beta(t)*Vmax/(N+KN)+Qebm(t)/VolBox)

    # dP/dt - phytoplankton concentration variation
    dP = P*(N*alpha(t)*beta(t)*Vmax/(N+KN)-Lambda-Qebm(t)/VolBox) + ChlorophyllInf(t)*Qocean(t)/VolBox
    
    return [dN,dP]

def system(t,y,param):
    return system_2param(t,y,param)

# ===============================================================
# Numerical Integration with Adaptive Tolerance
# ===============================================================
lightmode = False
Tol_ini = 1e-8
def integrate_ode(system, parameters, tmax, NN,y0):
    """Integrates the ODE system over tmax cycles using adaptive tolerance."""
    Tend = tmax*cyclesize
    tspan = (0,Tend)
    tstore = np.linspace(Tend-cyclesize, Tend, NN+1)
    
    scale = np.maximum(1.0, np.abs(y0))
    
    rTol = Tol_ini
    maxTol = 1e-1

    while rTol <= maxTol:
        try:
            atol_vec = rTol * scale

            sol = scipyInt.solve_ivp(
                system, tspan, y0, args=(parameters,),
                method="BDF", t_eval=tstore,
                rtol=rTol, atol=atol_vec
            )
            if sol.success and np.all(np.isfinite(sol.y)):
                return sol
            # else: # for debugging
                # print("solve_ivp failed:", sol.message, "tol=", rTol, "param=", parameters)
        except Exception as e:
            print("Integration exception:", repr(e), "tol=", rTol, "param=", parameters)
        
        rTol *= 10
    
    print("Integration failure.")
    return None

# 6.4 SystemPredictedSolution -------------
def systemPredictedSolution(system, parameters, tmax, NN,y0):
    """
    Integrates the system and returns the solution in DFO-LS-compatible form.
    """
    sol = integrate_ode(system, parameters, tmax, NN, y0)    
    if (lightmode==False) and (sol is None):
        # Returns a safe value compatible with the misfit residual function.
        print(f'Sol is none, N={NN}, Parameters = {parameters}') 
        return None
    return sol.y

# ===============================================================
# Observations and Artificial Observations
# ===============================================================
def Chlorophyll(NN,tmax):
    """Upper-box chlorophyll-A observations for one full cycle."""
    Tend = tmax*cyclesize
    tSamples = np.linspace(Tend-cyclesize, Tend, NN+1) # Ultimo ciclo, NN+1 pontos
    vec = np.zeros(NN+1)
    for i, t in enumerate(tSamples):
        vec[i] = ChlorophyllUp(t)
    return vec

def Nitrate(NN,tmax):
    """Upper-box nitrate observations for one full cycle."""
    Tend = tmax*cyclesize
    tSamples = np.linspace(Tend-cyclesize, Tend, NN+1) # Ultimo ciclo, NN+1 pontos
    vec = np.zeros(NN+1)
    for i, t in enumerate(tSamples):
        vec[i] = NitrateUp(t)
    return vec

def generateObservations(NN,tmax):
    """
    Observations generated from weighted "real" observational data.
    Returns array shape (2, NN+1).
    """
    nitrate = Nitrate(NN,tmax)
    chl = Chlorophyll(NN,tmax)
    return np.maximum(0, np.array([nitrate,chl]))

tmax_artificial=1000. # integration time for generating artificial observations
def generateArtificialObservations(ObsParams,NN):
    """
    Integrates the model using user-selected parameters to produce
    artificial observations.
    """
    global tmax_artificial
    # Initial condition based on the first observation
    y0 = generateObservations(2,1)[:,0]
    Obs = systemPredictedSolution(system, ObsParams, tmax_artificial, NN,y0)
    return Obs

# ===============================================================
# 7. GENERAL PARAMETERS FOR PEC EXPERIMENTS
#    (Heuristic, Integration, DFOLS)
# ===============================================================

# Heuristic parameters
N_H = 12            # number of samples per cycle in the heuristic
tmaxH_aux = 51.0    # number of cycles for spin-up in heuristic stage

# Grid parameters for section 5.1 experiments
grid_size_a = 11    # number of Vmax values evaluated in grid search
grid_size_b = 11    # number of lambda values evaluated in grid search

# Processing capacity for parallel execution
def get_n_workers():
    total = os.cpu_count()
    # use 75% of CPU cores, minimum 1, maximum total-1
    workers = max(1, min(total - 1, int(total * 0.75)))
    return workers

# Full integration parameters for DFOLS
N = 120             # number of observation points per cycle
tmax_aux = 201.0    # cycles needed to reach quasi-periodic regime

# Parameter bounds for DFOLS optimization (Vmax, lambda)
Boundsinf = np.array([])
Boundsup  = np.array([])

def _update_model_mode():
    global system, Boundsinf, Boundsup, boundsParams

    if MODEL_MODE == "2param":
        system_new = system_2param
        Boundsinf_new = np.array([0.1, 0.01])
        Boundsup_new  = np.array([6.0, 0.8])

    elif MODEL_MODE == "1param":
        system_new = system_1param
        Boundsinf_new = np.array([0.01])
        Boundsup_new  = np.array([0.8])

    elif MODEL_MODE == "Nitrate_param":
        system_new = system_w_param
        Boundsinf_new = 1.47*np.array([0.4])
        Boundsup_new  = 1.47*np.array([0.8])
    
    Boundsinf = Boundsinf_new
    Boundsup  = Boundsup_new
    system = system_new

    boundsParams = (Boundsinf, Boundsup)

    return

def set_model_mode(mode: str):
    global MODEL_MODE
    MODEL_MODE = mode
    _update_model_mode()
    return

def get_bounds():
    global Boundsinf, Boundsup, boundsParams
    return Boundsinf, Boundsup, boundsParams
# ====================================================================
# Light Mode for quick tests
# ====================================================================
def set_light_mode(active: bool):
    """
    Activates/deactivates Light Mode for quick testing,
    drastically reducing simulation cost.
    But also losing any accuracy.

    Usage:
        from PEC_aux_functions import set_light_mode
        set_light_mode(True)

    When active, the entire executable automatically uses:
        - small tmax
        - small sample counts
        - reduced grid size
        - modified non-stiff ODE model for integration
    """
    global lightmode, Tol_ini, system, tmax_artificial, tmaxH_aux, tmax_aux, N, N_H, grid_size_a, grid_size_b

    if active:
        # Smaller integration horizons
        tmaxH_aux = 5
        tmax_aux  = 5
        tmax_artificial = 1.0
        # Reduced sampling
        N = 12
        # Smaller heuristic search grids
        grid_size_a = 3
        grid_size_b = 3

        # Suppress prints from integration errors
        lightmode = True

        # Increase tolerance for integration
        Tol_ini = 1e-3
        
        # Lightweight ODE system (only for testing correctness)
        def system_light(t, y, param):
            if MODEL_MODE == "2param":
                Vmax, Lambda = param
                w = 0.588
            elif MODEL_MODE == "1param":
                Vmax = Vmax2(t)
                Lambda = param[0]
                w = 0.588
            elif MODEL_MODE == "Nitrate_param":
                Vmax = Vmax2(t)
                Lambda = 0.05
                w = param[0]

            N = max(y[0], 0.0)
            P = max(y[1], 0.0)
            
            dN = w*Nriver(t)*Qriver(t)/VolBox+NitrateInf(t)*Qocean(t)/VolBox+0.7*Lambda*P-N*(P*alpha(t)*beta(t)*Vmax/(N+KN)+Qebm(t)/VolBox)
            dP = P*(N*alpha(t)*beta(t)*Vmax/(N+KN)-Lambda-Qebm(t)/VolBox) + ChlorophyllInf(t)*Qocean(t)/VolBox
            
            return [max(0,dN),max(0,dP)] # artificially enforced positivity for the ODE — not scientifically valid
        
        # Replace system inside this module
        system = system_light

        print(f"[Light Mode = {active}] \n Note: This run is executing in Light Mode, which is intended only for checking that the executable runs without numerical errors. To perform the full original experiment with scientifically valid integration times and grid sizes, please set Light Mode = False before running. \n")
    return

# ===============================================================
# 8. MORTALITY CALIBRATION AND WEIGHTED Vmax
# ===============================================================
"""
This section includes:
 - Transformation of species time series (Skeletonema costatum,
   Asterionellopsis glacialis) into equivalent time coordinates.
 - Construction of interpolating functions.
 - Computation of an effective Vmax (Vmax2) weighted by the relative
   abundance of SCostatum and AGlacialis.

Used in the mortality calibration experiments and in the alternative
formulation described in the thesis chapter 5, sections 5.2 and 5.3.
"""

def adjust_species_cycle(arr):
    """
    Adjusts the time column of each species using:
        new_time = 103 + 14 * original_day.
    """
    new_time = 103.0 + 14.0 * arr[:, 0]
    arr[:, 0] = np.remainder(new_time, cyclesize)
    return arr

SCostatum = adjust_species_cycle(DATA["SCostatum"])
AGlacialis = adjust_species_cycle(DATA["AGlacialis"])

interp_SC = make_linear_interpolator(SCostatum)
interp_AG = make_linear_interpolator(AGlacialis)

VmaxSC = 2.5
VmaxAG = 3.2
def Vmax2(t):
    """Weighted-average Vmax based on species proportions."""
    SC = interp_SC(t)
    AG = interp_AG(t)
    sumP = SC+AG
    weight = SC/sumP
    return weight*VmaxSC + (1-weight)*VmaxAG

# ===============================================================
# 9. REPORT GENERATION — 2-PAGE A4 REPORT WITH SCALED FIGURES
# ===============================================================

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import numpy as np
import json
import io
from scipy.interpolate import griddata


def generate_experiment_report(json_file, output_pdf="PEC_Report.pdf"):
    """
    Generates a PDF report for:
      - Section 5.1 (2D heuristic: Vmax, lambda)
      - Section 5.2 (1D heuristic: lambda)
      - Section 5.3 (1D heuristic: w)
    
    Page 1:
        - Text summary
        - Figures 1 and 2
    Page 2:
        - Figures 3a and 3b
    """

    # -----------------------------------------------------------
    # LOAD JSON
    # -----------------------------------------------------------
    with open(json_file, 'r') as file:
        data = json.loads(next(file))
        Heuristic = data["MisfitHeuristic"]
        y0 = np.array(data["y0"])
        if MODEL_MODE == "2param":
            ObsParams = np.array(data["ObsParams"])

        data = json.loads(next(file))
        dfols_iterations = data["DFOLSiterations"]
        Bounds           = np.array(data["Bounds"])
        OptParams        = np.array(data["DFOLSoutput"])

    # -----------------------------------------------------------
    # GENERATE OBSERVATIONS
    # -----------------------------------------------------------
    N = 12
    OptObs = systemPredictedSolution(system, OptParams, tmax_aux, N, y0)
    Obs_t = np.linspace(tmax_aux*cyclesize - cyclesize, tmax_aux*cyclesize, N+1)

    if MODEL_MODE == "2param":
        Obs = generateArtificialObservations(ObsParams, N)
    else:
        Obs = generateObservations(N, cyclesize)

    # Prepare heuristic plot
    param_list = Heuristic[0]
    Misfit_list = Heuristic[1]

    dfolsIter = np.array([item[0] for item in dfols_iterations])
    iterations = np.arange(len(dfolsIter))

    # ============================================================
    # FIGURE GENERATION UTILITIES (1D and 2D)
    # ============================================================

    def plot_heuristic_2d():
        """
        Contour-style plot for 2D heuristic search (used in Section 5.1).
        """
        mpl.rcParams.update({'figure.figsize': (4.5, 4.5)})
        fig = plt.figure()

        grid_x, grid_y = np.mgrid[Bounds[0,0]:Bounds[1,0]:100j,
                              Bounds[0,1]:Bounds[1,1]:100j]
        grid_z = griddata(param_list, Misfit_list, (grid_x, grid_y), method='linear')
        plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap='viridis')
        plt.colorbar()
        Vmax_list, lambda_list = zip(*param_list)
        plt.scatter(Vmax_list, lambda_list, c='red', s=1, alpha=0.5)
        plt.axvline(x=ObsParams[0], color='lightgreen')
        plt.axhline(y=ObsParams[1], color='lightgreen')
        plt.title("Heuristic search")
        plt.xlabel("Parameter Vmax")
        plt.ylabel("Parameter lambda")
        return fig
    
    def plot_heuristic_1d():
        """
        Line plot for 1D heuristic search (used in Sections 5.2 and 5.3).
        """
        fig = plt.figure(figsize=(4.5, 4.5))
        ax = plt.gca()
        ax.plot(param_list, Misfit_list, linewidth=1.5)

        if MODEL_MODE == "1param":
            xlabel = "Parameter lambda"
        elif MODEL_MODE == "Nitrate_param":
            xlabel = "Parameter w"

        ax.set_title("Heuristic search")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Misfit")
        
        return fig

    def fig_dfols():
        mpl.rcParams.update({'figure.figsize': (4.5, 4.5)})
        fig, ax = plt.subplots()  
        
        if MODEL_MODE == "2param":
            x = dfolsIter[:, 0]
            y = dfolsIter[:, 1]
            ax.set_xlabel("Parameter Vmax")
            ax.set_ylabel("Parameter lambda")
        else:
            x = dfolsIter
            y = 0*x
            ax.set_xlabel("Parameter lambda")

        scatter = ax.scatter(x, y, c=iterations, s=80, alpha=0.8, edgecolor='white')
        cbar = plt.colorbar(scatter, label='Iteration')
        ticks = np.unique(np.linspace(0, len(iterations)-1, 6, dtype=int))
        cbar.set_ticks(ticks)

        ax.scatter(x[0],  y[0],  color='lime',  s=150,
                   edgecolor='black', label='Initial guess')
        ax.scatter(x[-1], y[-1], color='black', s=150,
                   edgecolor='white', label='Optimized')

        ax.set_title("DFO-LS search")
        ax.legend(loc='upper right')
        ax.grid(alpha=0.2)
        return fig

    def fig_fitting_N():
        mpl.rcParams.update({'figure.figsize': (8, 4.5)})
        fig = plt.figure()
        ax = plt.gca()

        xAxis = Obs_t - Obs_t[0]
        ax.plot(xAxis, np.maximum(Obs[0], 0),
                'red', label=['Observations'], zorder=1)

        ax.plot(xAxis, np.maximum(OptObs[0], 0),
                'green', linestyle='dashed', label=['Model output'], zorder=2)

        indices = np.linspace(0, len(Obs_t)-1, 13, dtype=int)
        ax.scatter(xAxis[indices],
                   np.maximum(OptObs[0][indices], 0),
                   s=8, c='black', zorder=3)

        ax.legend(loc='upper center', frameon=True, framealpha=0.85)
        ax.set_xlabel("t (days)")
        ax.set_ylabel("Nitrate Concentration (mmol/m³)")
        ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.6)
        ax.margins(x=0.03)
        return fig

    def fig_fitting_P():
        mpl.rcParams.update({'figure.figsize': (8, 4.5)})
        fig = plt.figure()
        ax = plt.gca()

        xAxis = Obs_t - Obs_t[0]
        ax.plot(xAxis, np.maximum(Obs[1], 0),
                'red', label=['Observations'], zorder=1)

        ax.plot(xAxis, np.maximum(OptObs[1],0),
                'green', linestyle='dashed',
                label=['Model output'], zorder=2)

        indices = np.linspace(0, len(Obs_t)-1, 13, dtype=int)
        ax.scatter(xAxis[indices],
                   np.maximum(OptObs[1][indices], 0),
                   s=8, c='black', zorder=3)

        ax.legend(loc='upper left', frameon=True, framealpha=0.85)
        ax.set_xlabel("t (days)")
        ax.set_ylabel("Chlorophyll-A Concentration (mmol/m³)")
        ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.6)
        ax.margins(x=0.03)
        return fig

    # ============================================================
    # Helper: convert a fig into an image array to embed into PDF
    # ============================================================

    def fig_to_img(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img = plt.imread(buf)
        plt.close(fig)
        return img

    # =============================================
    # Produce images for PDF embedding
    # =============================================
    if MODEL_MODE == "2param":
        IMG1 = fig_to_img(plot_heuristic_2d())
    else:
        IMG1 = fig_to_img(plot_heuristic_1d())

    IMG2 = fig_to_img(fig_dfols())
    IMG3 = fig_to_img(fig_fitting_N())
    IMG4 = fig_to_img(fig_fitting_P())

    # ============================================================
    # BUILD 2-PAGE PDF
    # ============================================================

    with PdfPages(output_pdf, json_file) as pdf:

        # --------------------------------------------------------
        # PAGE 1 — TEXT + FIGURE 1 + FIGURE 2 (side by side)
        # --------------------------------------------------------
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        # Text block at top
        text = f"""
        PEC Experiments — Calibration Report
        Data from {json_file}.

        Optimized parameters:
        """

        if MODEL_MODE == "2param":
            text += f"""
            Vmax   = {OptParams[0]:.6f}
            Lambda = {OptParams[1]:.6f}

        Observational parameters:
            Vmax   = {ObsParams[0]:.6f}
            Lambda = {ObsParams[1]:.6f}
        """
            
        elif MODEL_MODE == "1param":
            text += f"""
            Lambda = {OptParams[0]:.6f}
        """

        elif MODEL_MODE == "Nitrate_param":
            text += f"""
            w = {OptParams[0]:.6f}
        """


        fig.text(0.05, 0.9, text, va='top', family='monospace', fontsize=12)

        # --- embed two images side by side ---
        # Keep proportions: width = 0.44 each

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

        # Top figure
        ax3 = fig.add_axes([0.10, 0.52, 0.80, 0.40])
        ax3.imshow(IMG3)
        ax3.axis("off")

        # Bottom figure
        ax4 = fig.add_axes([0.10, 0.05, 0.80, 0.40])
        ax4.imshow(IMG4)
        ax4.axis("off")

        pdf.savefig(fig)
        plt.close(fig)

    print(f"[Report] PDF saved → {output_pdf}")
