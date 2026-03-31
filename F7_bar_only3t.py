import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

FS_TITLE  = 30
FS_LABEL  = 25
FS_TICK   = 20
FS_LEGEND = 22

# --- Left panel (UNCHANGED) ---
BIOCARD_control_mean = [1.25928635, 2.90270638, 0.00564346]
BIOCARD_control_std  = [0.92177672, 1.46329804, 0.47438195]
BIOCARD_mcidat_mean  = [2.22382236, 4.03446444, 0.70217719]
BIOCARD_mcidat_std   = [1.45063352, 1.84776484, 0.97756546]

con_colors = ["#FF9999", "#99CCFF", "#99FF99"]
mci_colors = ["#FF6666", "#6699FF", "#66FF66"]

# --- Right panel: keep ONLY 1.2 mm (3T) ---
mean_12 = np.array([2.22382236, 4.03446444, 0.70217719])
std_12  = np.array([1.45063352, 1.84776484, 0.97756546])
labels  = ['Amygdala', 'ERC/TEC', 'Hippocampus']
colors12 = ["#FF6666", "#6699FF", "#66FF66"]
bar_w = 0.3
fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharey=True)

# ---------------- Left panel ----------------
ax = axes[0]
x_control = 0.0
x_mci     = 1.4
offsets = np.array([-bar_w, 0.0, bar_w])

ax.bar(
    x_control + offsets, BIOCARD_control_mean, width=bar_w,
    color=con_colors, edgecolor="black", linewidth=1.0,
    yerr=BIOCARD_control_std, capsize=5
)
ax.bar(
    x_mci + offsets, BIOCARD_mcidat_mean, width=bar_w,
    color=mci_colors, edgecolor="black", linewidth=1.0,
    yerr=BIOCARD_mcidat_std, capsize=5
)

ax.set_ylim(-2.5, 8)
ax.set_xticks([x_control, x_mci])
ax.set_xticklabels(["Control", "MCI/AD"], fontsize=FS_LABEL)
ax.set_ylabel("Mean Volume Atrophy Rate (%)", fontsize=FS_LABEL)
ax.tick_params(axis="y", labelsize=FS_TICK)
ax.grid(axis="y", linestyle="--", alpha=0.35)

region_legend_handles = [
    Patch(facecolor=mci_colors[0], edgecolor="black", label="Amygdala"),
    Patch(facecolor=mci_colors[1], edgecolor="black", label="ERC/TEC"),
    Patch(facecolor=mci_colors[2], edgecolor="black", label="Hippocampus"),
]
ax.legend(handles=region_legend_handles, loc="upper left", frameon=True, fontsize=FS_LEGEND)

# ---------------- Right panel (ONLY 1.2mm 3T) ----------------
ax1 = axes[1]
x = np.array([0.0, 0.8, 1.6])   # tighter spacing
bar_w = 0.45                   # wider bars


ax1.bar(
    x, mean_12, bar_w,
    color=colors12, edgecolor='black',
    yerr=std_12, capsize=5
)

ax1.set_ylim(-2.5, 8)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=FS_LABEL)
ax1.set_ylabel('Mean Volume Atrophy Rate (%)', fontsize=FS_LABEL)
ax1.tick_params(axis="y", labelsize=FS_TICK)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Combined 3-color legend block for 1.2 mm (3T)
patches_12 = tuple(
    Patch(facecolor=c, edgecolor='black', linewidth=1.5)
    for c in colors12
)

ax1.legend(
    [patches_12],
    ['1.2 mm (3T)'],
    handler_map={tuple: HandlerTuple(ndivide=None)},
    fontsize=FS_LEGEND,
    loc="upper right",
    title="Group: MCI/AD",
    title_fontsize=FS_LEGEND,
    frameon=True
)


plt.tight_layout()
plt.savefig("Figure/BIOCARD_VA_only3t.png", dpi=300, bbox_inches="tight")
plt.close()
