#!/usr/bin/python
# -------------------------------------------------------------
# Author: Hongwei Cai, Huiyu Chu
# Date:   2025-06-06
# Description: Biocomputing with Brainoware
#              Procedure 2 - Reservoir computing hardware properties
#              5. Spatial information processing
#              This script reads the saved spatial-information data (spatial_information_plot.xlsx,
#              sheet "Fig 2e") and renders the evoked responses to the two complementary
#              stimulation patterns (P1 and P2) as raster/heatmap plots, in the same style as
#              the evoked-response raster of step 1 (psth_exp_..._raster.svg): electrodes on the
#              y-axis, time on the x-axis, colour = firing, RdYlBu_r colormap.
#
#              The two patterns drive distinct, complementary spatial activation maps, showing
#              that the organoid reservoir separates spatial input information.
# -------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## ------------------------------------------------------------------ ##
## read the spatial-information data (two stacked blocks: P1 then P2)
## ------------------------------------------------------------------ ##
here = os.path.dirname(os.path.abspath(__file__))
data = pd.read_excel(
    os.path.join(here, "spatial_information_plot.xlsx"),
    sheet_name="Fig 2e",
    header=None,
)

# P1 occupies rows 2-17, P2 occupies rows 21-36; column 0 holds the electrode labels
elec_labels = data.iloc[2:18, 0].tolist()  # Electrode 16 ... Electrode 1
P1 = (
    data.iloc[2:18, 1:]
    .apply(pd.to_numeric, errors="coerce")
    .to_numpy(dtype=float)
)
P2 = (
    data.iloc[21:37, 1:]
    .apply(pd.to_numeric, errors="coerce")
    .to_numpy(dtype=float)
)

## ------------------------------------------------------------------ ##
## two complementary raster/heatmap plots (step-1 raster style)
## ------------------------------------------------------------------ ##
fontsize = 8  # single font size shared by every text element in the figure
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = fontsize

vmax = max(
    np.nanmax(P1), np.nanmax(P2)
)  # shared colour scale for both patterns

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, pattern, name in zip(axes, [P1, P2], ["Pattern 1", "Pattern 2"]):
    sns.heatmap(
        pattern,
        cmap="RdYlBu_r",
        vmin=0,
        vmax=vmax,
        linewidths=0,
        yticklabels=elec_labels,
        xticklabels=5,
        cbar_kws={"label": "Firing rate (Hz)"},
        ax=ax,
    )
    ax.set_title(name, fontsize=fontsize)
    ax.set_xlabel("Time (bins)", fontsize=fontsize)
    ax.set_ylabel("Electrode", fontsize=fontsize)
    ax.tick_params(axis="both", which="both", length=0, labelsize=fontsize)
    plt.setp(ax.get_yticklabels(), rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.label.set_size(fontsize)

plt.tight_layout()
plt.savefig(os.path.join(here, "fig3e_spatial_raster.svg"), format="svg")
plt.show()
