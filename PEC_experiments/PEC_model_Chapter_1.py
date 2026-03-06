"""
========================================================================
PEC Experiment — Chapter 1
------------------------------------------------------------------------
Context:
    Short-time biochemical experiment (Chapter 1),
    included as supplementary material in Appendix A.

------------------------------------------------------------------------
This executable performs:
    (1) Explicit definition of a Nutrient–Phytoplankton (NP) model
    (2) Model predictions obtained by numerical integration
    (3) Structured JSONL output and PDF reporting (two figures)

Author:
    Letícia Becher Yamashita
========================================================================
"""

# ===============================================================
# 1. IMPORTS
# ===============================================================

import numpy as np
import scipy.integrate as scipyInt
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

# ===============================================================
# 2. OUTPUT FILES
# ===============================================================
output_pdf="PEC_Chapter1_Report.pdf"
output_json = "PEC_Chapter1.jsonl"

# ===============================================================
# 3. EXPERIMENT CONFIGURATION
# ===============================================================

Tend = 10.0 # Model integration time interval (10 days)
tspan = (0.0,Tend)
cyclesize = 360 # Annual cycle (360 days), month (30 days)
KN = 0.1 # Biochemical constant: half-saturation constant for nitrate uptake

# Parameters and initial condition
ObsParams = np.array([1.4,0.05]) # Synthetic tracer observational parameters [Vmax, lambdaP]
y0 = np.array([80.0,50.0]) # Synthetic tracer data for initializing simulation [N, P]

# ===============================================================
# 4. NP MODEL (ODE SYSTEM)
# ===============================================================
def system_NP(t,y,parameters):
    """
    Single-box Nutrient–Phytoplankton (NP) biochemical model.

    State:
        y[0] = N (nitrate concentration)
        y[1] = P (phytoplankton concentration)

    Parameters:
        parameters[0] = Vmax
        parameters[1] = lambdaP
    """
    Vmax = parameters[0]
    lambdaP = parameters[1]
    N = y[0]
    P = y[1]
    
    dN = 0.7*lambdaP*P-N*P*(Vmax/(N+KN)) # Nitrate concentration variation
    dP = P*(N*Vmax/(N+KN)-lambdaP) # Phytoplankton concentration variation
    
    return [dN,dP]

# ===============================================================
# 5. NUMERICAL INTEGRATION (SIMPLE, NON-ADAPTIVE)
# ===============================================================
ModelOutput_aux = scipyInt.solve_ivp(
    system_NP, tspan, y0, args=(ObsParams,),
    method='BDF', t_eval=tspan,
    rtol=1e-4, atol=1e-4*y0
)

ModelOutput = ModelOutput_aux.y
ModelOutput_t = ModelOutput_aux.t

# ===============================================================
# 8. SAVING RESULTS AND GENERATING 1-PAGE PDF REPORT
# ===============================================================
results_dict = {
    "Params": ObsParams.tolist(),
    "timeArray": ModelOutput_t.tolist(),
    "ModelOutput": ModelOutput.tolist(),
}
with open(output_json, "a") as f: 
    f.write(json.dumps(results_dict) + "\n")

# PDF REPORT
def fig_fitting_N(tspan, model_output):
    mpl.rcParams.update({'figure.figsize': (8, 4.5)})
    fig = plt.figure()
    ax = plt.gca()

    xAxis = tspan

    ax.plot(xAxis, np.maximum(model_output[0], 0),
            'green', linestyle='dashed', label=['Model output'], zorder=2)

    ax.legend(loc='upper center', frameon=True, framealpha=0.85)
    ax.set_xlabel("t (days)")
    ax.set_ylabel("Nitrate Concentration (mmol/m³)")
    ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.6)
    ax.margins(x=0.03)
    return fig

def fig_fitting_P(tspan, model_output):
    mpl.rcParams.update({'figure.figsize': (8, 4.5)})
    fig = plt.figure()
    ax = plt.gca()

    xAxis = tspan

    ax.plot(xAxis, np.maximum(model_output[1], 0),
            'green', linestyle='dashed', label=['Model output'], zorder=2)

    ax.legend(loc='upper left', frameon=True, framealpha=0.85)
    ax.set_xlabel("t (days)")
    ax.set_ylabel("Chlorophyll-A Concentration (mmol/m³)")
    ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.6)
    ax.margins(x=0.03)
    return fig

IMG1 = fig_fitting_N(tspan, ModelOutput)
IMG2 = fig_fitting_P(tspan, ModelOutput)


with PdfPages(output_pdf) as pdf:

    page = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    ax_page = page.add_axes([0, 0, 1, 1])
    ax_page.axis("off")

    header = (
        "PEC Experiment — Chapter 1 (Short-time NP biochemical model)\n"
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