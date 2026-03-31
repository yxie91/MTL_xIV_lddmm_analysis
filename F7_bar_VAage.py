import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple


FS_TITLE  = 30
FS_LABEL  = 25
FS_TICK   = 20
FS_LEGEND = 22

#Data from results of F7_BIOCARD_VAage.py
BIOCARD_control_mean=[1.04470515, 2.63108641, 0.02164953]
BIOCARD_control_std=[0.94341036, 1.20866284, 0.48099681]
BIOCARD_mcidat_mean=[2.01153786, 3.41857671, 0.67207094]
BIOCARD_mcidat_std=[1.58709796, 1.7159493,  0.91980122]
BIOCARD_age=[71.24395604395605,73.35641025641026]
con_colors = ["#FF9999", "#99CCFF", "#99FF99"]
mci_colors = ["#FF6666", "#6699FF", "#66FF66"]
ad_colors  = ["#CC0000", "#0000CC", "#009900"]
age_colors  = ["#CFCFCF", "#6F6F6F"]

labels = ['Amygdala', 'ERC/TEC', 'Hippocampus']
colors12= ["#FF9999", "#99CCFF", "#99FF99"]
colors15 = ["#FF3333", "#3366FF", "#33CC33"]
colors20 = ["#8B0000", "#00008B", "#006400"]

regions = ["Amygdala", "ERC/TEC", "Hippocampus"]
bar_w   = 0.3

fig, ax = plt.subplots(1, 1, figsize=(15, 15), sharey=True)


ax.set_title("BIOCARD", fontsize=FS_TITLE)#, fontweight="bold")
x_control = 0.0
x_mci     = 1.4
offsets = np.array([-bar_w, 0.0, bar_w])
# Regions (left y-axis)
ax.bar(x_control + offsets, BIOCARD_control_mean, width=bar_w, color=con_colors, edgecolor="black", linewidth=1.0, zorder=2)
ax.bar(x_mci     + offsets, BIOCARD_mcidat_mean,     width=bar_w, color=mci_colors, edgecolor="black", linewidth=1.0, zorder=2)
ax.set_ylim(0,9)
# Ages (right y-axis)
ax_r = ax.twinx()
ax_r.set_ylim(65, 85)
ax_r.set_ylabel("Age (years)", fontsize=FS_LABEL)

x_age = 2.8
age_offsets_adni = np.array([-bar_w/2, +bar_w/2])
ax_r.bar(x_age + age_offsets_adni[0], BIOCARD_age[0], width=bar_w, color=age_colors[0], edgecolor="black", linewidth=1.0, zorder=1)
ax_r.bar(x_age + age_offsets_adni[1], BIOCARD_age[1], width=bar_w, color=age_colors[1], edgecolor="black", linewidth=1.0, zorder=1)

ax.set_xticks([x_control, x_mci, x_age])
ax.set_xticklabels(["Control", "MCI/AD", "Age"], fontsize=FS_LABEL)#, fontweight="bold")
ax.set_ylabel("Mean Volume Atrophy Rate (%)", fontsize=FS_LABEL)
ax.tick_params(axis="y", labelsize=FS_TICK)

ax_r.tick_params(axis="y", labelsize=FS_TICK)
ax.grid(axis="y", linestyle="--", alpha=0.35)

region_legend_handles = [
    Patch(facecolor=mci_colors[0], edgecolor="black", label="Amygdala"),
    Patch(facecolor=mci_colors[1], edgecolor="black", label="ERC/TEC"),
    Patch(facecolor=mci_colors[2], edgecolor="black", label="Hippocampus"),
]
age_legend_handles_adni = [
    Patch(facecolor=age_colors[0], edgecolor="black", label="Age: Control"),
    Patch(facecolor=age_colors[1], edgecolor="black", label="Age: MCI/AD"),
]
leg = ax.legend(handles=region_legend_handles, loc="upper left", frameon=True, fontsize=FS_LEGEND)
ax.add_artist(leg)
plt.tight_layout()
plt.savefig(f"Figure/BIOCARD_VA_new.png", dpi=300, bbox_inches="tight")
plt.close()