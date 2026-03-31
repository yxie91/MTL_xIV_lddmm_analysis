import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple


FS_TITLE  = 30
FS_LABEL  = 25
FS_TICK   = 20
FS_LEGEND = 22

#Data from results of F7_BIOCARD_VAage.py
BIOCARD_control=[1.25928635, 2.90270638, 0.00564346]
BIOCARD_mcidat=[2.22382236, 4.03446444, 0.70217719]
BIOCARD_age=[71.24395604395605,73.35641025641026]
con_colors = ["#FF9999", "#99CCFF", "#99FF99"]
mci_colors = ["#FF6666", "#6699FF", "#66FF66"]
ad_colors  = ["#CC0000", "#0000CC", "#009900"]
age_colors  = ["#CFCFCF", "#6F6F6F"]

#Data from results of F7_BIOCARD_VA_315T.py
mean_12 = np.array([2.22382236, 4.03446444, 0.70217719])
mean_15 = np.array([1.552563,   1.82211934, 0.23179515])
mean_20 = np.array([2.11524214, 1.45105731, 0.66970512])
std_12 = np.array([1.45063352, 1.84776484, 0.97756546])
std_15 = np.array([3.99356104, 2.81651503, 2.06329408])
std_20 = np.array([2.61588323, 2.75746014, 1.20776241])
labels = ['Amygdala', 'ERC/TEC', 'Hippocampus']
colors12= ["#FF9999", "#99CCFF", "#99FF99"]
colors15 = ["#FF3333", "#3366FF", "#33CC33"]
colors20 = ["#8B0000", "#00008B", "#006400"]

regions = ["Amygdala", "ERC/TEC", "Hippocampus"]
bar_w   = 0.3

fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharey=True)


ax = axes[0]
ax.set_title("BIOCARD 3T", fontsize=FS_TITLE)#, fontweight="bold")
x_control = 0.0
x_mci     = 1.4
offsets = np.array([-bar_w, 0.0, bar_w])
# Regions (left y-axis)
ax.bar(x_control + offsets, BIOCARD_control, width=bar_w, color=con_colors, edgecolor="black", linewidth=1.0, zorder=2)
ax.bar(x_mci     + offsets, BIOCARD_mcidat,     width=bar_w, color=mci_colors, edgecolor="black", linewidth=1.0, zorder=2)
ax.set_ylim(0,8)
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
leg1 = ax.legend(handles=region_legend_handles, loc="upper left", frameon=True, fontsize=FS_LEGEND)
ax.add_artist(leg1)
ax_r.legend(handles=age_legend_handles_adni, loc="upper right", frameon=True, fontsize=FS_LEGEND)
x = np.arange(len(labels))

ax1 = axes[1]
ax1.set_title("BIOCARD all", fontsize=FS_TITLE)#, fontweight="bold")
ax1.bar(x - bar_w, mean_12, bar_w, color=colors12, edgecolor='black')#, yerr=std_12, capsize=5)
ax1.bar(x , mean_15, bar_w, color=colors15, edgecolor='black')#, yerr=std_15, capsize=5)
ax1.bar(x + bar_w, mean_20, bar_w, color=colors20, edgecolor='black')#, yerr=std_20, capsize=5)
ax1.set_ylim(0,8)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=FS_LABEL)
ax1.set_ylabel('Mean Volume Atrophy Rate (%)', fontsize=FS_LABEL)
ax1.tick_params(axis="y", labelsize=FS_TICK)

ax1.grid(axis='y', linestyle='--', alpha=0.7)

patches12 = tuple(Patch(facecolor=c, edgecolor='black', linewidth=1.5) for c in colors12)
patches15 = tuple(Patch(facecolor=c, edgecolor='black', linewidth=1.5) for c in colors15)
patches20 = tuple(Patch(facecolor=c, edgecolor='black', linewidth=1.5) for c in colors20)

ax1.legend(
    [(patches12), (patches15), (patches20)],
    ['1.2 mm (3T)', '1.5 mm (1.5T)', '2.0 mm (1.5T)'],
    handler_map={tuple: HandlerTuple(ndivide=None)},
    fontsize=FS_LEGEND,
    loc="upper right",
    title="Group: MCI/AD",
    title_fontsize=FS_LEGEND,
    frameon=True)

plt.tight_layout()
plt.savefig(f"Figure/BIOCARD_VA.png", dpi=300, bbox_inches="tight")
plt.close()