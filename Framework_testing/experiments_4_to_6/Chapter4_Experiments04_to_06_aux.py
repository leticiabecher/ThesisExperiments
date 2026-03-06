"""
========================================================================
Chapter 4: Experiments 4 to 6 — Auxiliary Functions and Model Core
------------------------------------------------------------------------
Context:
    Supplementary material supporting the experiment scripts and
    workflow described in the thesis:

        Yamashita, L. B. (2025). A Data-Constrained Framework for Marine
        Biogeochemistry Modeling with Applications to the Paranaguá
        Estuarine Complex. PhD Thesis, Universidade Federal do Paraná.

    This module centralizes:
        (i)  data loading and preprocessing utilities;
        (ii) construction of cyclic forcing series;
        (iii) ODE system definition for the problem at hand;
        (iv) numerical integration wrappers with robustness safeguards;
        (v) helper routines used by computational experiments and reports.

Purpose:
    Provide a single, reusable implementation layer for the experiment
    scripts presented in Appendix A (Implementation and Computational
    Experiments).

Notes:
    - The dataset is expected as a JSON Lines file named "dataset.jsonl"
      in the working directory.

Author:
    Letícia Becher Yamashita
========================================================================
"""

# ===============================================================
# 1. IMPORTS AND GLOBAL CONSTANTS
# ===============================================================

import numpy as np
import scipy.integrate as scipyInt
import os

lightmode = False
# Initital time for integration
Tini = 0.
# Spinup time for obtaining an approximate steady state as the 
# initial condition for integration
tmax1 = 40000.
tspan1 = (Tini,Tini+tmax1)

# Number of sample points for the heuristic search step
N_H = 10
# Time interval where integration is performed for the heuristic search step
tspanH = (Tini,Tini+200.)
# Time interval where integration is saved for the heuristic search step
tstoreH = np.linspace(tspanH[1]-100., tspanH[1], N_H+1)

# Interpolation parameters
# Percentual width of the horizontal and vertical transition lane 
# in relation to the dimensions of the quadrants 
DxB = 0.25
DyB = 0.25

# Time interval where integration is performed for optimization
# tmax = 2000. +100.
tmax = 200. +100.
tspan = (Tini,Tini+tmax)
# Time interval where integration is saved for optimization
tstoreini = tmax-100.

# Initial tolerance for the numerical integration
Tol_ini = 1e-12

# ===============================================================
# 2. ODE SYSTEM, NUMERICAL INTEGRATION & OBSERVATION GENERATION
# ===============================================================
def system(t,y,parameters):
    """Main ODE system for the problem at hand"""
    x, v = y
    a = parameters[0]
    b = parameters[1]
    dxdt = -x+v*(a+x**2)
    dvdt = b-v*(a+x**2)
    return [dxdt,dvdt]

def fixedPoint(parameters:np.ndarray):
    """Fixed point for the ODE system"""
    a = parameters[0]
    b = parameters[1]
    return np.array([b, b/(a+b**2)])

def integrate(system, tspan, y0, parameters, tstore):
    """ODE system integration."""
    
    AuxTol = Tol_ini
    max_tol = 1e-3
    
    for _ in range(10):
        try:
            sol = scipyInt.solve_ivp(
                system, tspan, y0, args=(parameters,),
                method='BDF', t_eval=tstore,
                rtol=AuxTol, atol=AuxTol
            )

            if sol.success and np.all(np.isfinite(sol.y)):
                return sol
        except:
            pass
        
        AuxTol *= 10
        if AuxTol > max_tol:
            break
    
    print(f"Integration fail: parameters {parameters} lead to NaN/inf.")
    return None

def systemPredictedSolution(system, tspan, y0, parameters, tstore,N):
    """"
    Integration for generating the points compared to the observation 
    in the misfit function.
    """
    
    sol = integrate(system, tspan, y0, parameters, tstore)
    
    if sol is None:
        # Returns a safe value consistent with the residual function, for not causing 
        # a crash while optimizing. This is a high value that will eventually be altered 
        # by the misfit function.
        print(f'Sol is none, N={N}, [a,b] = {parameters}')
        return 5000*np.ones_like(tstore) 
    return sol.y

def get_y0(i,list_parameters,list_y0):
    """ Obtaining the initial point for the integration of the ODE system as 
    approximately a point from the limit cycle."""
    aObs = list_parameters[i,0]
    bObs = list_parameters[i,1]

    # Observational parameters, which we want to recover via calibration
    ObsParameters = np.array([aObs,bObs])

    y01 = fixedPoint(ObsParameters) + np.array([0.06, 0.5])
    tstore1 = np.linspace(Tini+tmax1-1., Tini+tmax1, 3)
            
    sol = integrate(system, tspan1, y01, ObsParameters, tstore1)
    y0 = sol.y[:, -1]  

    list_y0[i,:] = y0
    return y0

def quadrant_bounds(SampledObs):
    """Quadrants definition: setting external  and internal boundaries
    in relation to each axis."""
    xLowerBound = np.min(SampledObs[0])
    xUpperBound = np.max(SampledObs[0])
    
    xMeanBound = (xLowerBound+xUpperBound)/2
    
    deltaxBound = ((xUpperBound-xLowerBound)/2)*DxB

    yLowerBound = np.min(SampledObs[1])
    yUpperBound = np.max(SampledObs[1])
    
    yMeanBound = (yLowerBound+yUpperBound)/2*DxB
    
    deltayBound = ((yUpperBound-yLowerBound)/2)*DyB

    return xMeanBound,yMeanBound,deltayBound,deltaxBound,xUpperBound,xLowerBound,yUpperBound,yLowerBound

def determine_quadrant(x,y,xMeanBound,yMeanBound,deltayBound,deltaxBound):
    """ Determines the quadrant containig the closest observation point."""
    nQ1 = 0.
    nQ2 = 0.
    nQ3 = 0.
    nQ4 = 0.

    xs1 = False
    xs2 = False
    xs3 = False
    xs4 = False

    ys1 = False
    ys2 = False
    ys3 = False
    
    if x<xMeanBound-deltaxBound:
        xs1 = True
    elif x<xMeanBound:
        xs2 = True
    elif x<xMeanBound+deltaxBound:
        xs3 = True
    else:
        xs4 = True

    if y<yMeanBound-deltayBound:
        ys1 = True
    elif y<yMeanBound:
        ys2 = True
    elif y<yMeanBound+deltayBound:
        ys3 = True

    if xs1 == True:
            #Region Q3:
        if ys1 == True:
            nQ3 += 1.
            #Region S1y:
        elif ys2 == True:
            nQ3 +=  1.
            nQ2 += 1.-(yMeanBound-y)/deltayBound
            #Region S2y
        elif ys3 == True:
            nQ3 +=  1.-(y-yMeanBound)/deltayBound
            nQ2 += 1.
            #Region Q2
        else:
            nQ2 += 1.
    elif xs2 == True:
            #Region S1x:
        if ys1 == True:
            nQ3 +=  1.
            nQ4 += 1.-(xMeanBound-x)/deltaxBound
            #Region S1xy:
        elif ys2 == True:
            nQ3 +=  1.
            auxnQ2 = 1.-(yMeanBound-y)/deltayBound
            nQ2 += auxnQ2
            auxnQ4 = 1.-(xMeanBound-x)/deltaxBound
            nQ4 += auxnQ4
            nQ1 += np.sqrt(auxnQ2**2+auxnQ4**2)
            #Region S2xy
        elif ys3 == True:
            auxnQ3 =  1.-(y-yMeanBound)/deltayBound
            nQ3 += auxnQ3
            nQ2 += 1.
            auxnQ1 = 1.-(xMeanBound-x)/deltaxBound
            nQ1 += auxnQ1
            nQ4 += np.sqrt(auxnQ3**2+auxnQ1**2)
            #Region S2x
        else:
            nQ2 += 1.
            nQ1 += 1.-(xMeanBound-x)/deltaxBound
    elif xs3 == True:
            #Region S4x:
        if ys1 == True:
            nQ3 +=  1.-(x-xMeanBound)/deltaxBound
            nQ4 += 1. 
            #Region S4xy:
        elif ys2 == True:
            nQ4 += 1.
            auxnQ3 =  1.-(x-xMeanBound)/deltaxBound
            nQ3 += auxnQ3
            auxnQ1 = 1.-(yMeanBound-y)/deltayBound
            nQ1 += auxnQ1
            nQ2 += np.sqrt(auxnQ3**2+auxnQ1**2)
            #Region S3xy
        elif ys3 == True:
            nQ1 += 1.
            auxnQ2 = 1.-(x-xMeanBound)/deltaxBound
            nQ2 += auxnQ2
            auxnQ4 = 1.-(y-yMeanBound)/deltayBound
            nQ4 += auxnQ4
            nQ3 +=  np.sqrt(auxnQ2**2+auxnQ4**2)
            #Region S3x
        else:
            nQ1 += 1.
            nQ2 += 1.-(x-xMeanBound)/deltaxBound
    elif xs4 == True:
            #Region Q4:
        if ys1 == True:
            nQ4 += 1. 
            #Region S4y:
        elif ys2 == True:
            nQ4 += 1.
            nQ1 += 1.-(yMeanBound-y)/deltayBound
            #Region S3y
        elif ys3 == True:
            nQ1 += 1.
            nQ4 += 1.-(y-yMeanBound)/deltayBound
            #Region Q1
        else:
            nQ1 += 1.
    
    return np.array([nQ1,nQ2,nQ3,nQ4])

def interpolation(SampledObs,xUpperBound,xLowerBound,yUpperBound,yLowerBound):
    """Linear interpolation of the observational limit cycle graph. 
    sol1: unsorted list containing the interpolation
    sol2: sampled points from the integration of the observational limit cycle (optional)"""
    # Step 1. Sorting the points
    # Removing any duplicate points from sol1
    sol1 = np.unique(SampledObs, axis=1)

    # Step 1.1 Sorting from the initial point to the rightmost intermediate point: (left to right)
    # Ascending reordering of sol1 with respect to x (first line):
    sol1 = sol1[:, np.argsort(sol1[0, :])]

    # Step 1.2 Choosing the leftmost point on the limit cycle graph.
    y1 = sol1[1, 0]
    IndicesRemove = np.where(sol1[1,:]>= y1)[0]
    Part1 = sol1[:, IndicesRemove]
    
    sol1 = np.delete(sol1, IndicesRemove, axis=1)

    # Step 1.3 Sorting from intermediate point 1 to intermediate point 2: (still from left to right)
    y2 = sol1[1,-1]
    x2 = Part1[0,-1]
    IndicesRemove = np.where((sol1[1,:]> y2) & (sol1[0,:]> x2))[0]
    Part2 = sol1[:, IndicesRemove]
    
    sol1 = np.delete(sol1, IndicesRemove, axis=1)

    # Step 1.4 Sorting from the intermediate point2 to the final point: (right to left)
    Part3 = sol1[:, ::-1]

    # The result so far is the array sol2, whose columns (x,y) are consecutive points of the limit cycle.
    sol2 = np.concatenate((Part1,Part2,Part3),axis=1)

    # Step 2. Piecewise linear interpolation
    aux_x = np.max(xUpperBound) -  np.min(xLowerBound)
    aux_y = np.max(yUpperBound) -  np.min(yLowerBound)
    auxDelta = 28.
    aux_diag = np.sqrt(aux_x**2+aux_y**2)

    # Sufficient proximity parameter
    Delta = aux_diag/auxDelta

    # List of indices of sol2, such that i and i+1 are distant.
    list = []

    for ii in range(len(sol2[0,:])-1):
        P1 = sol2[:, ii]
        P2 = sol2[:, ii+1]
        aux = np.linalg.norm(P2 - P1)
        if aux > Delta:
            list.append(ii)

    # Adding points to the array sol2.
    # Redefining the auxiliary array sol1 to contain the (unsorted) linear interpolation points.
    sol1 = sol2

    for ii in list:
        P1 = sol2[:, ii]
        P2 = sol2[:, ii+1]
        
        Vector = P2 - P1
        Dist = np.linalg.norm(Vector)
        n = int(np.ceil(Dist / Delta)) + 1

        tt = np.linspace(0, 1, n)
        pontos = (P1 + tt.reshape(-1, 1) * Vector).T
        
        sol1 = np.append(sol1, pontos[:,1:-1], axis=1)
    
    return sol1

def distance_to_cycle(X1,Sol1,NQ,xMeanBound,yMeanBound,deltayBound,deltaxBound):
        X = X1.flatten()
        auxSize = len(Sol1[0])
        Y2 = np.ones(auxSize)
        Y1 = np.array([X[0]*np.ones(auxSize),X[1]*np.ones(auxSize)])-Sol1
        
        for ii in range(auxSize):
            Y2[ii] = np.linalg.norm(Y1[:,ii])

        # Find the closest point in the limit cycle graph:
        Indice  = np.argmin(Y2)
        Y = Y2[Indice]

        # Checking the quadrant and region where the nearest point is located 
        # in order to assign weights to the residual function
        x = Sol1[0,Indice]
        y = Sol1[1,Indice]

        # Assigning weights to the residue function
        residual = determine_quadrant(x,y,xMeanBound,yMeanBound,deltayBound,deltaxBound)
    
        #---------------------------------------------------------------------
        # Multiplying the weights by the distance between the point and the limit cycle.
        residual *= Y
        #---------------------------------------------------------------------
        # Assigning weights based on the number of observations in each quadrant
        residual *= NQ

        return residual

# ===============================================================
# 3. UTILITY FUNCTIONS
# ===============================================================

def feasible_ab(a, b):
    # Checks if parameters a and b are in the feasible region.
    if 8.0 * a > 1.0:
        return False
    fr = 1.0 - 8.0 * a
    if fr < 0:
        return False
    low = np.sqrt(0.5 * (1.0 - 2.0 * a - np.sqrt(fr)))
    up  = np.sqrt(0.5 * (1.0 - 2.0 * a + np.sqrt(fr)))
    return (b >= low) and (b <= up)

def make_pack(MODEL_MODE: str, aObs: float, bObs: float):
    """
    Converts the solver input into parameters=[a,b].
    """
    if MODEL_MODE == "2param":
        def pack(params: np.ndarray) -> np.ndarray:
            return params

    elif MODEL_MODE == "a_param":
        def pack(param: np.ndarray) -> np.ndarray:
            return np.array([param, bObs], dtype=float)

    elif MODEL_MODE == "b_param":
        def pack(param: np.ndarray) -> np.ndarray:
            return np.array([aObs, param], dtype=float)

    return pack

# Number of heuristic evaluations
grid_size_a = 33
grid_size_b = 33
grid_size_ab = 6
def Grid_heuristic(misfitH,param_list_a, param_list_b,minimizers):
    """Auxiliary function for generating a parameter grid within the feasible region
    in the simultaneous calibration of a and b"""           
    # Generating the heuristic function at feasible points.
    # At non-feasible points, the heuristic function assumes the value -1 to facilitate the search for
    # local minimizers outside the boundary
    fGrid = -1*np.ones((len(param_list_a), len(param_list_b)))
    
    for i in range(len(param_list_a)):
        for j in range(len(param_list_b)):
            ab1 = np.array([param_list_a[i],param_list_b[j]])

            if not (8 * ab1[0] > 1) and not (ab1[1] < np.sqrt(0.5 * (1 - 2 * ab1[0] - np.sqrt(1 - 8 * ab1[0])))) and not (ab1[1] > np.sqrt(0.5 * (1 - 2 * ab1[0] + np.sqrt(1 - 8 * ab1[0])))):
                fGrid[i,j] = misfitH(ab1)

    if minimizers==True:
        # Checking if the parameters [a,b] are a local minimizer, excluding boundaries
        local_minimizers = []
        for i in range(1, len(param_list_a) - 1):
            for j in range(1, len(param_list_b) - 1):
                # Obtaining f evaluated at 8 neighbors of the point ab1
                fneighbours = (fGrid[i-1,j-1:j+1], fGrid[i,j-1:j+1], fGrid[i+1,j-1:j+1])
                # Checking if [a,b] is a local minimizer and not a boundary point
                if fGrid[i,j] == np.min(fneighbours):
                    local_minimizers.append([param_list_a[i],param_list_b[j]])
        Delta_ab = np.array([param_list_a[1]-param_list_a[0],param_list_b[1]-param_list_b[0]])

        local_minimizers = np.array(local_minimizers)
    else:
        local_minimizers = []
        Delta_ab = []
    # Keeping only parameters inside the feasible region
    Indices = np.argwhere(fGrid >= 0)
    f_list = fGrid[Indices[:, 0], Indices[:, 1]]
    param_list = np.column_stack((param_list_a[Indices[:, 0]], param_list_b[Indices[:, 1]]))
    return param_list,f_list,local_minimizers,Delta_ab

def make_misfitH(MODEL_MODE, y0=None, dist_to_cycle=None, aObs=None, bObs=None): 

    pack = make_pack(MODEL_MODE, aObs=aObs, bObs=bObs)

    def misfitH(params) -> float:
        parameters = pack(params)
        samples = systemPredictedSolution(system, tspanH, y0, parameters, tstoreH,N_H)        
        residual = np.zeros(4)
        for j in range(N_H):
            residual += dist_to_cycle(samples[:,j])**2
        residual = np.sqrt(residual)
        misfit = np.linalg.norm(residual)
        return misfit        
    return misfitH

def penalty(parameters: np.ndarray,fceil: float):        
    """Penalty function for the simultaneous calibration of a and b."""
    a = parameters[0]
    b = parameters[1]
    
    # Adaptive scaling factor (can be adjusted as needed)
    penalty_scale = 1.0  

    # Computing all violations independently:
    violation1 = max(0, 8*a - 1)  # Forces 8a ≤ 1
    # If violation1 > 0, violations 2 and 3 will not be well defined.
    if violation1>0. :
        violation2 = fceil
        violation3 = fceil
    else:
        violation2 = max(0, 0.5*(1 - 2*a - np.sqrt(1 - 8*a)) - b**2)  # b ≥ lower_bound
        violation3 = max(0, b**2 - 0.5*(1 - 2*a + np.sqrt(1 - 8*a)))  # b ≤ upper_bound

    penalty1 = penalty_scale * (violation1 + violation2 + violation3)

    return penalty1

def make_pack2(MODEL_MODE: str, aObs: float, bObs: float):
    """
    Converts the solver input into parameters=[a,b].
    """
    if MODEL_MODE == "2param":
        def pack2(params: np.ndarray) -> np.ndarray:
            return params

    elif MODEL_MODE == "a_param":
        def pack2(param: np.ndarray) -> np.ndarray:
            return np.array([param[0], bObs], dtype=float)

    elif MODEL_MODE == "b_param":
        def pack2(param: np.ndarray) -> np.ndarray:
            return np.array([aObs, param[0]], dtype=float)

    return pack2

def make_residual(MODEL_MODE, y0=None, tstore=None, N=None, dist_to_cycle=None, fCeil=None, aObs=None, bObs=None, dfols_history=None): 
    pack2 = make_pack2(MODEL_MODE, aObs=aObs, bObs=bObs)
    def res(params: np.ndarray) -> np.ndarray:
        parameters = pack2(params)
        samples = systemPredictedSolution(system, tspan, y0, parameters, tstore, N)        
        residual1 = np.zeros(4)
        for ii in range(N):
            residual1 += dist_to_cycle(samples[:,ii])**2

        residual = np.hstack([np.minimum(fCeil, residual1),np.array([penalty(parameters,fCeil)])])
        
        misfit = 0.5*np.linalg.norm(residual)
        dfols_history.append([parameters.tolist(), misfit])
        
        return residual
    
    return res

# ====================================================================
# Light Mode for quick tests
# ====================================================================
def set_light_mode(active: bool):
    """
    Activates/deactivates Light Mode for quick testing,
    drastically reducing simulation cost.
    But also losing any accuracy.

    Usage:
        from Chapter4_Experiments04_to_06_aux import set_light_mode
        set_light_mode(True)

    When active, the entire executable automatically uses:
        - small tmax
        - small sample counts
        - reduced grid size
        - modified non-stiff ODE model for integration
    """
    global lightmode, Tol_ini, tmax1, tmax, grid_size_a, grid_size_b, grid_size_ab

    if active:
        # Smaller integration horizons
        tmax1 = 100.0
        tmax  = 100.0
        # Reduced sampling
        # Smaller heuristic search grids
        grid_size_a = 5
        grid_size_b = 5
        grid_size_ab = 3

        # Suppress prints from integration errors
        lightmode = True

        # Increase tolerance for integration
        Tol_ini = 1e-3

        print(f"[Light Mode = {active}] \n Note: This run is executing in Light Mode, which is intended only for checking that the executable runs without numerical errors. To perform the full original experiment with scientifically valid integration times and grid sizes, please set Light Mode = False before running. \n")
    return

# ===============================================================
# REPORT GENERATION
# ===============================================================

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import numpy as np
import json
import io
from scipy.interpolate import griddata


def generate_experiment_report(json_file, output_pdf, MODEL_MODE):
    """
    Generates a PDF report for:
      - Section 5.4 (1D heuristic: a)
      - Section 5.5 (1D heuristic: b)
      - Section 5.6 (2D heuristic: a,b)
    
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
        file.readline()
        line = file.readline()
        data = json.loads(line)
        ObsParams = np.array(data["ObsParams"])
        y0 = np.array(data["y0"])
        N = np.array(data["N"])
        # Limit_cycle_t = np.array(data["Limit_cycle_t"])
        Limit_cycle_y = np.array(data["Limit_cycle_y"])   
        # tstoreN = np.array(data["tstoreN"])     
        tstore1 = np.array(data["tstore1"])
        # tstoreH = np.array(data["tstoreH"])
        # Linear_Interp = np.array(data["Linear_Interp"])
        MisfitHeuristic1 = data["MisfitHeuristic1"]
        MisfitHeuristic2 = data["MisfitHeuristic2"]
        fCeil_value = data["fCeil_value"]
        dfols_iterations = data["DFOLSiterations"]
        OptParams = np.array(data["DFOLSoutput"])
        Bounds = np.array(data["Bounds"])

    # ============================================================
    # FIGURE GENERATION UTILITIES (1D and 2D)
    # ============================================================
    def plot_heuristic_2d(param,misfit,auxTitle=''):
        """
        Contour-style plot for 2D heuristic search for experiment 6.
        """
        mpl.rcParams.update({'figure.figsize': (4.5, 4.5)})
        fig = plt.figure()

        grid_x, grid_y = np.mgrid[Bounds[0,0]:Bounds[1,0]:100j,
                              Bounds[0,1]:Bounds[1,1]:100j]
        param_list = np.array(param)
        Misfit_list = np.array(misfit)
        if auxTitle == "2":
            Misfit_list = np.minimum(Misfit_list, fCeil_value, None)
        grid_z = griddata(param_list, Misfit_list, (grid_x, grid_y), method='linear')
        plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap='viridis')
        plt.colorbar()
        Vmax_list, lambda_list = zip(*param_list)
        plt.scatter(Vmax_list, lambda_list, c='red', s=1, alpha=0.5)
        plt.axvline(x=ObsParams[0], color='lightgreen')
        plt.axhline(y=ObsParams[1], color='lightgreen')
        plt.title("Heuristic search "+auxTitle)
        plt.xlabel("Parameter a")
        plt.ylabel("Parameter b")
        return fig
    
    def plot_heuristic_1d(param,misfit,auxTitle=''):
        """
        Line plot for 1D heuristic search for experiments 4 and 5.
        """
        fig = plt.figure(figsize=(4.5, 4.5))
        ax = plt.gca()
        param_list = np.array(param)
        Misfit_list = np.array(misfit)
        if auxTitle == "2":
            Misfit_list = np.minimum(Misfit_list, fCeil_value, None)
        ax.plot(param_list, Misfit_list, linewidth=1.5)

        if MODEL_MODE == "a_param":
            xlabel = "Parameter a"
        elif MODEL_MODE == "b_param":
            xlabel = "Parameter b"

        ax.set_title("Heuristic search "+auxTitle)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Misfit")
        
        return fig

    def fig_dfols():
        mpl.rcParams.update({'figure.figsize': (4.5, 4.5)})
        fig, ax = plt.subplots()  
        x = [item[0][0] for item in dfols_iterations]
        y = [item[0][1] for item in dfols_iterations]
        iterations = range(len(x))
        ax.set_xlabel("Parameter a")
        ax.set_ylabel("Parameter b")

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

    def fig_fitting():
        mpl.rcParams.update({'figure.figsize': (8, 4.5)})
        fig = plt.figure()
        ax = plt.gca()
       
        # Sorting observational points
        Limit_cycle = interpolation(Limit_cycle_y,np.max(Limit_cycle_y[0]),np.min(Limit_cycle_y[0]),np.max(Limit_cycle_y[1]),np.min(Limit_cycle_y[0]))
        x = Limit_cycle[0]
        y = Limit_cycle[1]
        ax.plot(x[0:len(Limit_cycle_y[0])-1], y[0:len(Limit_cycle_y[0])-1],
                'red', label=['Observations'], zorder=1)
        
        tspan = (Tini,Tini+tmax)
        sol = integrate(system, tspan, y0, ObsParams, tstore1)
        Samples = sol.y
        ax.scatter(Samples[0], Samples[1],
                s=8, c='blue', label='Model output', zorder=2)

        ax.legend(loc='upper center', frameon=True, framealpha=0.85)
        ax.set_xlabel("Parameter a")
        ax.set_ylabel("Parameter b")
        ax.set_title("Fitting to observations")
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
        IMG1 = fig_to_img(plot_heuristic_2d(MisfitHeuristic1[0], MisfitHeuristic1[1],))
        IMG2 = fig_to_img(fig_dfols())
        IMG3 = fig_to_img(fig_fitting())
    else:
        IMG1 = fig_to_img(plot_heuristic_1d(MisfitHeuristic1[0], MisfitHeuristic1[1],"1"))
        IMG2 = fig_to_img(plot_heuristic_1d(MisfitHeuristic2[0], MisfitHeuristic2[1],"2"))        
        IMG3 = fig_to_img(fig_dfols())
        IMG4 = fig_to_img(fig_fitting())

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
        Chapter 4 Experiments — Calibration Report
        Data from {json_file}.

        Observational parameters: 

        a = {ObsParams[0]:.6f}
        b = {ObsParams[1]:.6f}

        
        Optimized parameters:
        """
        if MODEL_MODE == "2param":
            text += f"""
        a = {OptParams[0]:.6f}
        b = {OptParams[1]:.6f}
            """
        elif MODEL_MODE == "a_param":
            text += f"""
        a = {OptParams[0]:.6f}
        b = {ObsParams[1]:.6f}
            """
        elif MODEL_MODE == "b_param":
            text += f"""
        a = {ObsParams[0]:.6f}
        b = {OptParams[0]:.6f}
            """

        fig.text(0.05, 0.9, text, va='top', family='monospace', fontsize=12)

        # --- embed two images side by side ---
        # Keep proportions: width = 0.44 each

        ax1 = fig.add_axes([0.05, 0.15, 0.44, 0.44])
        ax1.imshow(IMG1)
        ax1.axis("off")

        if not MODEL_MODE == "2param":
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

        if MODEL_MODE == "2param":
            ax3 = fig.add_axes([0.05, 0.5, 0.44, 0.44])
            ax3.imshow(IMG2)
            ax3.axis("off")
        
            ax3 = fig.add_axes([0.10, 0.15, 0.7, 0.44])
            ax3.imshow(IMG3)
            ax3.axis("off")
        else:
            ax3 = fig.add_axes([0.05, 0.5, 0.44, 0.44])
            ax3.imshow(IMG3)
            ax3.axis("off")

            ax3 = fig.add_axes([0.10, 0.1, 0.7, 0.44])
            ax3.imshow(IMG4)
            ax3.axis("off")

        pdf.savefig(fig)
        plt.close(fig)

    print(f"[Report] PDF saved → {output_pdf}")
