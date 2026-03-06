---------------------------------------------------------------------
Chapter4_experiment_01.py
---------------------------------------------------------------------
Chapter 4 — Experiment 01: theoretical example of parameter calibration

This script reproduces the full computational workflow for the
one-parameter calibration experiment.

Computational workflow:
1. Generation of observations from the exact analytical solution
   (using the “true” parameters)
2. Grid-based heuristic search for parameter a (with b fixed)
3. Optimization using the derivative-free DFO-LS algorithm (one parameter)
4. Structured JSONL output containing:
   - grid of tested values for a
   - misfit curve J(a)
   - DFO-LS iteration history
   - optimal value of a
5. Automatic PDF report generation including:
   - misfit curves for different values of N
   - DFO-LS convergence history
   - model–observation comparison in time for x(s) and y(s)

---------------------------------------------------------------------
Chapter4_experiment_02.py
---------------------------------------------------------------------
Chapter 4 — Experiment 02: simultaneous calibration of parameters (a, b)

This script performs the simultaneous calibration of two parameters in a
controlled theoretical setting.

Workflow:
1. Generation of perfect observations using the observational parameters
   a_obs, b_obs
2. Grid-based heuristic search in the (a, b) plane
3. Optimization using DFO-LS (two parameters)
4. Structured JSONL output containing:
   - list of evaluated (a, b) pairs
   - misfit values J(a, b) on the grid
   - DFO-LS iteration history
   - optimal parameters and associated misfit
5. Automatic PDF report generation including:
   - contour map of the misfit function over the (a, b) plane
   - DFO-LS convergence trajectory
   - model–observation comparison for x(s) and y(s) over the last cycle

---------------------------------------------------------------------
Chapter4_experiment_03.py
---------------------------------------------------------------------
Chapter 4 — Experiment 03: calibration of parameter a using weighted
residual combinations

This script studies parameter identifiability under different observation
designs.

Workflow:
1. Definition of observation designs through pairs (N, α), with N equally
   spaced points over the last cycle
2. Construction of residuals res1 and res2 and of the misfit function J(a; α)
3. Optimization with DFO-LS (one parameter) for each (N, α)
4. Structured JSONL output containing:
   - values of f_obs and ||res_obs|| for the observational parameter
   - DFO-LS iteration history
   - estimated optimal parameter for each (N, α)
5. Automatic PDF report generation including:
   - misfit curves J(a) for different values of α (fixed N)
   - dependence of the estimated optimal a on α and N
   - model–observation comparison for x(t) and v(t) over the last cycle

---------------------------------------------------------------------
Chapter4_experiments_03_to_06.py
---------------------------------------------------------------------
Chapter 4 — Experiments 04 to 06: Fitting model outputs to an 
observational limit cycle, when the analytical solution is unknown.

This script reproduces the full computational workflow for the
one-parameter and two-parameter calibration experiments.

Workflow:
1. Obtaining (an approximation of) an observational limit cycle
2. Construction of a light version of the misfit function and 
   heuristic search for an initial guess for optimization.
3. Update of the residual function and optimization with DFO-LS.
4. Structured JSONL output containing:
   - Observational setting
   - Optimization settings
   - DFO-LS iteration history
   - estimated optimal parameters
5. Automatic PDF report generation including:
   - heuristic search plot
   - DFO-LS search history plot
   - Fitting between the observations and the model output

=====================================================================
