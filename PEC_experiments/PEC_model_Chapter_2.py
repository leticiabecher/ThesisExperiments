"""
========================================================================
PEC Experiments — Chapter 2
------------------------------------------------------------------------
Context:
    Biogeochemical experiments (Chapter 2),
    included as supplementary material in Appendix A.

------------------------------------------------------------------------
This executable performs:
    (1) Explicit definition of a biogeochemical NP-model for 
        the Paranaguá Estuarine Complex dynamics
    (2) Model predictions obtained by numerical integration
    (3) Fitting to 'real' observations
    (4) An experiment on nutrient apport simulation
    (3) Structured JSONL output and PDF reporting (two figures)

Author:
    Letícia Becher Yamashita
========================================================================
"""

# ===============================================================
# 1. IMPORTS
# ===============================================================

import numpy as np
import PEC_aux_functions as PEC
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

# ===============================================================
# 2. OUTPUT FILES
# ===============================================================
output_pdf="PEC_Chapter2_Report.pdf"
output_json = "PEC_Chapter2.jsonl"

# ===============================================================
# 3. EXPERIMENT PARAMETERS
# ===============================================================

# Parameters and initial condition
ObsParams = np.array([1.4,0.05]) # Synthetic tracer observational parameters [Vmax, lambdaP]
y0 = np.array([80.0,50.0]) # Synthetic tracer data for initializing simulation [N, P]

# ===============================================================
# 4. MODEL SETUP AND NUMERICAL INTEGRATION
# ===============================================================

# First experiment (natural environmental conditions) --------------------------------------------------------------

system_NP = PEC.system_2param # NP-MODEL (ODE SYSTEM)

ModelOutput_aux = PEC.integrate(system_NP, ObsParams, 20, 12, y0) # Saves the last year after 20 years integration

ModelOutput = ModelOutput_aux.y
ModelOutput_t = ModelOutput_aux.t


# Second experiment (Diffuse anthropogenic supply of nutrients) ----------------------------------------------------

def system_NP2(t, y, param):
    old = PEC.Nriver
    try:
        PEC.Nriver = lambda tt: old(tt) + 10.0 # Increases the riverine nutrient supply in 10mmol/m³
        return PEC.system_2param(t, y, param)
    finally:
        PEC.Nriver = old

ModelOutput_aux2 = PEC.integrate(system_NP2, ObsParams, 20, 12, y0) # Saves the last year after 20 years integration

ModelOutput2 = ModelOutput_aux2.y
ModelOutput_t2 = ModelOutput_aux2.t

# ===============================================================
# 5. SAVING RESULTS AND GENERATING 2-PAGE PDF REPORT
# ===============================================================

# Saving results ------------------------------------------------

results_dict = {
    "Params": ObsParams.tolist(),
    "timeArray1": ModelOutput_t.tolist(),
    "ModelOutput1": ModelOutput.tolist(),
    "timeArray2": ModelOutput_t2.tolist(),
    "ModelOutput2": ModelOutput2.tolist()
}
with open(output_json, "a") as f: 
    f.write(json.dumps(results_dict) + "\n")


# PDF REPORT ----------------------------------------------------

Obs = PEC.generateObservations(12,1) # "Sampling 12+1 real observations along 1 year"
timeObs = np.linspace(0,360,13)

ObsN = Obs[0,:] # Nitrate observations
ObsP = Obs[1,:] # Phytoplankton observations

def fig_fitting_N(tstore, model_output):
    mpl.rcParams.update({'figure.figsize': (8, 4.5)})
    fig = plt.figure()
    ax = plt.gca()

    xAxis = tstore - 19*PEC.cyclesize

    ax.plot(timeObs, ObsN,
            'red', linestyle='-', label=['Observations'], zorder=1)
    
    ax.plot(xAxis, np.maximum(model_output[0], 0),
            'green', linestyle='dashed', label=['Model output'], zorder=2)

    ax.legend(loc='upper center', frameon=True, framealpha=0.85)
    ax.set_xlabel("t (days)")
    ax.set_ylabel("Nitrate Concentration (mmol/m³)")
    ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.6)
    ax.margins(x=0.03)
    return fig

def fig_fitting_P(tstore, model_output):
    mpl.rcParams.update({'figure.figsize': (8, 4.5)})
    fig = plt.figure()
    ax = plt.gca()

    xAxis = tstore - 19*PEC.cyclesize

    ax.plot(timeObs, ObsP,
            'red', linestyle='-', label=['Observations'], zorder=1)

    ax.plot(xAxis, np.maximum(model_output[1], 0),
            'green', linestyle='dashed', label=['Model output'], zorder=2)

    ax.legend(loc='upper left', frameon=True, framealpha=0.85)
    ax.set_xlabel("t (days)")
    ax.set_ylabel("Chlorophyll-A Concentration (mmol/m³)")
    ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.6)
    ax.margins(x=0.03)
    return fig

IMG1 = fig_fitting_N(ModelOutput_t, ModelOutput)
IMG2 = fig_fitting_P(ModelOutput_t, ModelOutput)

IMG3 = fig_fitting_N(ModelOutput_t2, ModelOutput2)
IMG4 = fig_fitting_P(ModelOutput_t2, ModelOutput2)


with PdfPages(output_pdf) as pdf:

    # ==========================
    # PAGE 1 — EXPERIMENT 1
    # ==========================
    
    page = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    ax_page = page.add_axes([0, 0, 1, 1])
    ax_page.axis("off")

    header = (
        "PEC Experiment 1 — Chapter 2 (1 year simulation under normal environmental conditions)\n"
    )
    page.text(0.05, 0.95, header, va="top", family="monospace", fontsize=11)

    def fig_to_img(fig):
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        img = plt.imread(buf)
        plt.close(fig)
        return img

    imgN = fig_to_img(IMG1)
    imgP = fig_to_img(IMG2)

    ax_top = page.add_axes([0.08, 0.52, 0.84, 0.33])
    ax_top.imshow(imgN)
    ax_top.axis("off")

    ax_bottom = page.add_axes([0.08, 0.14, 0.84, 0.33])
    ax_bottom.imshow(imgP)
    ax_bottom.axis("off")

    pdf.savefig(page)
    plt.close(page)

    # ==========================
    # PAGE 2 — EXPERIMENT 2
    # ==========================

    page2 = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    ax_page2 = page2.add_axes([0, 0, 1, 1])
    ax_page2.axis("off")

    header2 = (
        "PEC Experiment 2 — Chapter 2 (1 year simulation with +10 mmol/m³ river nutrient supply)\n"
    )
    page2.text(0.05, 0.95, header2, va="top", family="monospace", fontsize=11)

    imgN2 = fig_to_img(IMG3)
    imgP2 = fig_to_img(IMG4)

    ax_top2 = page2.add_axes([0.08, 0.52, 0.84, 0.33])
    ax_top2.imshow(imgN2)
    ax_top2.axis("off")

    ax_bottom2 = page2.add_axes([0.08, 0.14, 0.84, 0.33])
    ax_bottom2.imshow(imgP2)
    ax_bottom2.axis("off")

    pdf.savefig(page2)
    plt.close(page2)