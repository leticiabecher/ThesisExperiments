"""
========================================================================
Experiments 4 to 6 (Chapter 4) — Parameter Estimation and Calibration
------------------------------------------------------------------------
Context:
    This executable is a version of the computational workflow reported 
    in Chapter 4, Experiments 4 to 6 of the thesis, and is included as 
    supplementary material in Appendix A.
    Use this code for calibrating a list of parameters,
    over a list of observation numbers N.

------------------------------------------------------------------------
This executable performs:
    (1) Observation generation
    (2) Heuristic search
    (3) DFO-LS optimization
    (4) Structured JSON output and PDF reporting

Execution modes:
    - Light mode: numerical smoke test.
    - Full mode: experiments reported in the thesis.

Author:
    Letícia Becher Yamashita
===============================================================
"""

# ===============================================================
# 1. IMPORTS AND UTILS
# ===============================================================
from __future__ import print_function
import numpy as np
import dfols
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import Chapter4_Experiments04_to_06_aux as AUX

# ===============================================================
# 2. EXECUTION CONFIGURATION
# ===============================================================
# Light mode ON for quick testing (set False for full experiment)
AUX.set_light_mode(True)

# MODEL_MODE = "a_param" # For experiments calibrating a
MODEL_MODE = "b_param" # For experiments calibrating b
# MODEL_MODE = "2param" # For experiments calibrating a and b

if MODEL_MODE == "a_param":
    print(f"[Executing: Chapter 4 - Experiment 04]")
    # Output files to save results
    output_pdf = "Chapter4_experiment4_Report.pdf"
    output_json = "experiment4.jsonl"
    #Observational parameters for recovering a via calibration, where b is fixed
    # parameter_list = np.array([[0.001, 0.6], [0.02, 0.6], [0.04, 0.6], [0.06, 0.6], [0.08, 0.6], [0.1, 0.6], [0.12, 0.6]])
    parameter_list = np.array([[0.04, 0.6]])

elif MODEL_MODE == "b_param":
    print(f"[Executing: Chapter 4 - Experiment 05]")
    # Output files to save results
    output_pdf = "Chapter4_experiment5_Report.pdf"
    output_json = "experiment5.jsonl"
    #Observational parameters for recovering b via calibration, where a is fixed
    # parameter_list = np.array([[0.04, 0.3], [0.04, 0.4], [0.04, 0.6], [0.04, 0.7], [0.04, 0.9]])
    parameter_list = np.array([[0.04, 0.6]])

elif MODEL_MODE == "2param":
    print(f"[Executing: Chapter 4 - Experiment 06]")
    # Output files to save results
    output_pdf = "Chapter6_experiment6_Report.pdf"
    output_json = "experiment6.jsonl"
    #Observational parameters for recovering a and b via calibration
    # parameter_list = np.array([[0.02, 0.4], [0.02, 0.6], [0.02, 0.8], [0.06, 0.6], [0.12, 0.6]])
    parameter_list = np.array([[0.02, 0.6]])

# N: Number of sample points for the optimization step
# The list is used for performing the experiment using different numbers of samples
# list_N = np.array([20, 50, 100, 200, 400, 600])
list_N = np.array([10])
print(f"\n[Observational settings: \n N in {list_N} \n [aObs,bObs] in {parameter_list}")
size_parameter_list = len(parameter_list[:,0])

list_y0 = np.zeros_like(parameter_list)

results = {"list_N": list_N.tolist(), "ObsParams": parameter_list.tolist()}
with open(output_json, "a") as f:
        f.write(json.dumps(results) + "\n")
        
def Calibration_item(i,N):
    """
    Auxiliary function for executing parameter calibrations in parallel.
    i: Observational parameters index
    N: Number of observations
    """
    y0 = list_y0[i,:]
    aObs = parameter_list[i,0]
    bObs = parameter_list[i,1]

    # Observational parameters
    ObsParameters = np.array([aObs,bObs])

    # Time interval where integration is saved for obtaining the residual function sample points
    tstoreini = AUX.tmax - 20.
    tstoreN = np.linspace(AUX.Tini+tstoreini, AUX.Tini+AUX.tmax, N+1)
    # Minimum of sample points considered in calibration
    tstore1 = np.linspace(AUX.Tini, AUX.Tini+AUX.tmax, int(AUX.tmax)*2+1)
    # Observational samples generation, including the points used in the heuristic search  
    tstore = np.unique(np.concatenate((tstoreN, tstore1, AUX.tstoreH)))

    SolObs = AUX.integrate(AUX.system, AUX.tspan, y0, ObsParameters, tstore)
    SampledObs = SolObs.y
    t = SolObs.t

    SolObs_t = t
    SolObs_y = SampledObs

    #=========================================================
    # Misfit function definition
    #=========================================================
    xMeanBound,yMeanBound,deltayBound,deltaxBound,xUpperBound,xLowerBound,yUpperBound,yLowerBound = AUX.quadrant_bounds(SampledObs)
    def Determine_quadrant(X1):
        x = X1[0]
        y = X1[1]
        return AUX.determine_quadrant(x,y,xMeanBound,yMeanBound,deltayBound,deltaxBound)    

    # ---------------------------------------------------
    # Assigning weights to quadrants and regions, 
    # depending on the number of points in each of them 
    # ---------------------------------------------------
    # Option 1. Considering all the points saved during integration
    # ObsSample2 = SampledObs
    # Option 2. Considering only the N points sampled by the residual function
    indicesN = np.where(np.isin(t, tstoreN))[0]
    #tN = t[indicesN]
    ObsSample2 = SampledObs[:, indicesN]
    
    NQ = np.zeros(4)
    for ii in range(N+1):
        NQ += Determine_quadrant(ObsSample2[:, ii])
    NQ /= (N+1)
    
    # ---------------------------------------------------------
    # Piecewise linear interpolation of the limit cycle points, 
    # if they are too far from each other
    # --------------------------------------------------------
    sol1 = AUX.interpolation(SampledObs,xUpperBound,xLowerBound,yUpperBound,yLowerBound)

    def dist_to_cycle(X1):
        return AUX.distance_to_cycle(X1,sol1,NQ,xMeanBound,yMeanBound,deltayBound,deltaxBound)

    # Heuristic misfit function
    misfitH = AUX.make_misfitH(MODEL_MODE=MODEL_MODE, 
                     y0=y0, 
                     dist_to_cycle=dist_to_cycle,  
                     aObs=aObs, 
                     bObs=bObs
                     ) 
    
    # Parameter choices
    if MODEL_MODE == "a_param" or MODEL_MODE == "b_param":
        #---------------------------------------------------------
        # Heuristic search - Part 1
        #---------------------------------------------------------
        if MODEL_MODE == "a_param":
            InfBound = 0. 
            UpBound = -bObs**2+bObs*np.sqrt(2)
            param_list = np.linspace(InfBound, UpBound,AUX.grid_size_a)
            # Distance between two consecutive parameters:
            dH = (UpBound - InfBound)/(AUX.grid_size_a-1)
        
        elif MODEL_MODE == "b_param": 
            InfBound = np.sqrt(0.5*(1 - 2*aObs - np.sqrt(1 - 8*aObs))) 
            UpBound = np.sqrt(0.5*(1 - 2*aObs + np.sqrt(1 - 8*aObs)))
            param_list = np.linspace(InfBound, UpBound,AUX.grid_size_b)
            # Distance between two consecutive parameters:
            dH = (UpBound - InfBound)/(AUX.grid_size_b-1)
    

        param_list = np.array(param_list) 

        def list_misfitH(param):    
            return [param,misfitH(param)]

        # Parallel evaluation
        with ThreadPoolExecutor(os.cpu_count()) as executor:
            futures = [executor.submit(list_misfitH, p) for p in param_list]
            results = [f.result() for f in as_completed(futures)]

        results = np.array(results)
        order = np.argsort(results, axis=0)[:,0]
        Misfit_list = results[order,1]

        MisfitHeuristic1 = [param_list.tolist(), Misfit_list.tolist()]

        #---------------------------------------------------------
        # Heuristic search - Part 2
        #---------------------------------------------------------
        # Step 1. Obtaining all the local minimizers:
        min_Set = []
        for ii in range(1,len(Misfit_list)-1):
            auxf = Misfit_list[ii]
            if auxf<Misfit_list[ii-1] and auxf<Misfit_list[ii+1]:
                min_Set.append(ii)
        if min_Set ==[]:
            ii = np.argmin(Misfit_list)
            min_Set.append(ii)

        # An attempt to not exclude the extremities of the feasible region,
        # in case they contain a local minimizer
        if Misfit_list[0]<Misfit_list[1]:
            infBound_bool = True
        else:    
            infBound_bool = False    

        if Misfit_list[-1]<Misfit_list[-2]:
            upBound_bool = True
        else:    
            upBound_bool = False    

        # Step 2. Zoom in on the heuristic search around these local minimizers:
        # Number of points considered in this heuristic, for each local minimizer detected: 11    
        param_list2 = np.array([])
        for ii in min_Set:
            param_list2 = np.append(param_list2,np.linspace(max(InfBound, param_list[ii] - 2*dH), min(UpBound, param_list[ii] + 2*dH), 11))

        param_list2 = np.array(param_list2) 
        # Parallel evaluation
        with ThreadPoolExecutor(os.cpu_count()) as executor:
            futures = [executor.submit(list_misfitH, p) for p in param_list2]
            results = [f.result() for f in as_completed(futures)]
        results = np.array(results)
        order = np.argsort(results, axis=0)[:,0]
        Misfit_list2 = results[order,1]

        MisfitHeuristic2 = [param_list2.tolist(), Misfit_list2.tolist()]

        # Returns the best parameter found during the heuristic search.
        IndiceMin = np.argmin(Misfit_list2)
        InitialGuess = np.array([param_list2[IndiceMin]])

        # Limiting the misfit function, so that it doesn't grow excessively,
        # but still not hinder the search for the DFO-LS descent direction.
        # Other choices are possible, for example np.min(Misfit_list), with different effects on the optimization search. 
        MisfitMedian = np.median(Misfit_list2)

    elif MODEL_MODE == "2param":
    #---------------------------------------------------------
    # Heuristic search - Part 1
    #---------------------------------------------------------
    # Note: Step 2 was not considered in the calibration of 2 parameters simultaneously; this can be done similarly to what is described here, with 1 parameter.
        param_list_a = np.linspace(0., 0.14, AUX.grid_size_ab)
        param_list_b = np.linspace(0., 1., AUX.grid_size_ab)
        param_list,Misfit_list,local_minimizers, Delta_ab = AUX.Grid_heuristic(misfitH,param_list_a,param_list_b,minimizers=True)
    
        MisfitHeuristic1 = [param_list.tolist(),Misfit_list.tolist()]
        
        infBound_bool = False
        upBound_bool = False

        InfBound = np.array([0.,0.])
        UpBound = np.array([0.14,1.]) 
        # Distance between two consecutive parameters:
        dH = np.sqrt(UpBound - InfBound)/(33-1)

        MisfitMedian = np.median(Misfit_list)
        # Returns the best parameter found during the heuristic search.
        IndiceMin = np.argmin(Misfit_list)
        InitialGuess = np.array(param_list[IndiceMin], dtype=float)
        MisfitHeuristic2 = np.array([])
        param_list2 = np.array([])

    fCeil = MisfitMedian 
    if AUX.lightmode==True:
        print(f"\n Reached part 1: Heuristic finished for [aObs,bObs]={ObsParameters}, N={N}.")
    #----------------------------------------------------------------
    # Changes in the residual function
    #----------------------------------------------------------------
    # DFOLS evaluation history:
    # Each entry has the form [lambda, misfit], allowing
    # reconstruction of the convergence path for analysis and plotting.
    dfols_history = []
    res = AUX.make_residual(
        MODEL_MODE=MODEL_MODE,
        y0=y0,
        tstore=tstore,
        N=N,
        dist_to_cycle=dist_to_cycle,
        fCeil=fCeil,
        aObs=aObs,
        bObs=bObs,
        dfols_history=dfols_history
    )

    # Restricting the search area for a local minimum to a neighborhood of the heuristic minimum
    # Original boundaries: a in [0., 0.14], b in [0., 1.]
    # Considering an area corresponding to 0.2 of the original boundaries
    if MODEL_MODE=="a_param":
        if infBound_bool==True:
            Boundsinf = np.array([0.])
        else: 
            Boundsinf = np.array([max(0.,InitialGuess[0]-0.1*0.14)])
        
        if upBound_bool==True:
            Boundsup = np.array([0.14])
        else:
            Boundsup = np.array([min(0.14,InitialGuess[0]+0.1*0.14)])
    
    elif MODEL_MODE=="b_param":
        if infBound_bool==True:
            Boundsinf = np.array([0.])
        else:   
            Boundsinf = np.array([max(0.,InitialGuess[0]-0.1*1.)])
        
        if upBound_bool==True:
            Boundsup = np.array([1.])
        else:
            Boundsup = np.array([min(1.,InitialGuess[0]+0.1*1.)])

    elif MODEL_MODE=="2param":
        if infBound_bool==True:
            Boundsinf = np.array([0., 0.])
        else:   
            Boundsinf = np.array([max(0.,InitialGuess[0]-0.1*0.14), max(0.,InitialGuess[1]-0.1*1.)])
        
        if upBound_bool==True:
            Boundsup = np.array([0.14, 1.])
        else:
            Boundsup = np.array([min(0.14,InitialGuess[0]+0.1*0.14), min(1.,InitialGuess[1]+0.1*1.)])
    
    boundsteste = (Boundsinf,Boundsup)
    
    try:
        output1 = dfols.solve(res, x0=InitialGuess, bounds=boundsteste, scaling_within_bounds=True)     
        auxDfols = output1.x  # Optimized parameters
        if AUX.lightmode==True:
            print(f"\n Reached part 2: DFO-LS optimization finished for [aObs,bObs]={ObsParameters}, N={N}.")
    except Exception as e:
        print(f"Error while running DFOLS. \n,{e}")
        auxDfols = np.array([-1.])  # Standard value in case of failure
    
    results = {
        "ObsParams": ObsParameters.tolist(), 
        "y0": y0.tolist() if isinstance(y0, np.ndarray) else y0, 
        "N": N.item() if isinstance(N, np.integer) else N, 
        "Limit_cycle_t": SolObs_t.tolist() if isinstance(SolObs_t, np.ndarray) else SolObs_t, 
        "Limit_cycle_y": SolObs_y.tolist() if isinstance(SolObs_y, np.ndarray) else SolObs_y, 
        "tstoreN": tstoreN.tolist() if isinstance(tstoreN, np.ndarray) else tstoreN, 
        "tstore1": tstore1.tolist() if isinstance(tstore1, np.ndarray) else tstore1, 
        "tstoreH": AUX.tstoreH.tolist() if isinstance(AUX.tstoreH, np.ndarray) else AUX.tstoreH, 
        "Linear_Interp": sol1.tolist() if isinstance(sol1, np.ndarray) else sol1, 
        "MisfitHeuristic1": MisfitHeuristic1.tolist() if isinstance(MisfitHeuristic1, np.ndarray) else MisfitHeuristic1, 
        "MisfitHeuristic2": MisfitHeuristic2.tolist() if isinstance(MisfitHeuristic2, np.ndarray) else MisfitHeuristic2,
        "fCeil_value": fCeil.item() if isinstance(fCeil, np.floating) else fCeil, 
        "DFOLSiterations": dfols_history.tolist() if isinstance(dfols_history, np.ndarray) else dfols_history, 
        "DFOLSoutput": auxDfols.tolist() if isinstance(auxDfols, np.ndarray) else auxDfols,
        "Bounds": [Boundsinf.tolist(), Boundsup.tolist()],
        "InitialGuess": InitialGuess.tolist()
    }
    with open(output_json, "a") as f:
        f.write(json.dumps(results) + "\n")
    
    return results

#---------------------------------------------------------------
# Obtaining y0: the initial condition for integration as a point from the observational limit cycle
with ThreadPoolExecutor(os.cpu_count()) as executor:
    futures = [executor.submit(AUX.get_y0, i, parameter_list,list_y0) for i in range(size_parameter_list)]
    for future in as_completed(futures):
        future.result()
#---------------------------------------------------------------
# Parameter calibration
parallel_tasks = [(i, N) for i in range(size_parameter_list) for N in list_N]
for i, N in parallel_tasks:
    Calibration_item(i, N)
print("Execution completed successfully. Generating report.")
# ===============================================================
# REPORT GENERATION
# ===============================================================
AUX.generate_experiment_report(output_json, output_pdf, MODEL_MODE)