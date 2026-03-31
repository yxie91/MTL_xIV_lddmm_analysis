import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


nnUNet_mean=[0.872002306, 0.828395971, 0.937580033]
MRICloud_mean=[0.695547247, 0.415142611, 0.799236551]
Freesurfer_mean=[0.727585873, 0.282733593, 0.826226557]


nnUNet_std=[0.041002337, 0.058914288, 0.028888195]
MRICloud_std=[0.070098639, 0.092303535, 0.024301782]
Freesurfer_std=[0.080874291, 0.069521174, 0.03287051]

out_path="/cis/home/yxie91/paper2025/Figure/S1_segcompare.png"

# Data
regions = ["Amygdala", "ERC/TEC", "Hippocampus"]
bar_w = 0.25


out_path = "/cis/home/yxie91/paper2025/Figure/S1_segcompare.png"

# X positions
x = np.arange(len(regions))
offsets = np.array([-bar_w, 0.0, bar_w])

# Colors (distinct & clean)
colors = {
    "nnUNet": "#4C72B0",      # blue
    "MRICloud": "#DD8452",    # orange
    "FreeSurfer": "#55A868"   # green
}

# Plot
fig, ax = plt.subplots(figsize=(10, 7))

# Bars
ax.bar(x + offsets[0], nnUNet_mean, width=bar_w,
       color=colors["nnUNet"], edgecolor="black",
       yerr=nnUNet_std, capsize=5, label="nnU-Net")

ax.bar(x + offsets[1], MRICloud_mean, width=bar_w,
       color=colors["MRICloud"], edgecolor="black",
       yerr=MRICloud_std, capsize=5, label="MRICloud")

ax.bar(x + offsets[2], Freesurfer_mean, width=bar_w,
       color=colors["FreeSurfer"], edgecolor="black",
       yerr=Freesurfer_std, capsize=5, label="FreeSurfer")

# Labels & title
ax.set_xticks(x)
ax.set_xticklabels(regions, fontsize=14)
ax.set_ylabel("Dice Coefficient", fontsize=14)
ax.set_title("Segmentation Performance Comparison", fontsize=16)

# Y range
ax.set_ylim(0, 1.0)

# Grid
ax.grid(axis="y", linestyle="--", alpha=0.4)

# Legend
ax.legend(fontsize=12)

# Aesthetics
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()