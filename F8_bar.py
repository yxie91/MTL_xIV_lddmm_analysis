import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FS_TITLE  = 30
FS_LABEL  = 25
FS_TICK   = 20
FS_LEGEND = 22

#Data from results of F8_ADNI12GO_VA_2g.py
ADNI_control=[3.14942944, 3.5562738,  1.66763116]
ADNI_mci=[8.53427242, 7.61025027, 3.09047816]
ADNI_age=[74.27633928571429,73.0827380952381]
#Data from results of F8_ADNI34_VA_3g.py
ADNI34_control=[1.10804747, 1.4896111,  0.74579776]
ADNI34_mci=[2.75852225, 3.17414196, 1.3279492 ]
ADNI34_ad=[6.70709279, 7.63305388, 2.75762269]
ADNI34_age=[75.5498482572935,75.77016007019532,76.50101148376302]  # control, mci, ad
# Colors
con_colors = ["#FF9999", "#99CCFF", "#99FF99"]
mci_colors = ["#FF6666", "#6699FF", "#66FF66"]
ad_colors  = ["#CC0000", "#0000CC", "#009900"]
age_colors_adni   = ["#CFCFCF", "#6F6F6F"]
age_colors_adni34 = ["#CFCFCF", "#8C8C8C", "#4A4A4A"]

regions = ["Amygdala", "ERC/TEC", "Hippocampus"]
bar_w   = 0.25

fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharey=True)

ax = axes[0]
ax.set_title("ADNI 1/2/GO", fontsize=FS_TITLE)#, fontweight="bold")

x_control = 0.0
x_mci     = 1.4
offsets = np.array([-bar_w, 0.0, bar_w])

# Regions (left y-axis)
ax.bar(x_control + offsets, ADNI_control, width=bar_w, color=con_colors, edgecolor="black", linewidth=1.0, zorder=2)
ax.bar(x_mci     + offsets, ADNI_mci,     width=bar_w, color=mci_colors, edgecolor="black", linewidth=1.0, zorder=2)

# Ages (right y-axis)
ax.set_ylim(0,9)
ax_r = ax.twinx()
ax_r.set_ylim(65, 85)
ax_r.set_ylabel("Age (years)", fontsize=FS_LABEL)

x_age = 2.8
age_offsets_adni = np.array([-bar_w/2, +bar_w/2])
ax_r.bar(x_age + age_offsets_adni[0], ADNI_age[0], width=bar_w, color=age_colors_adni[0], edgecolor="black", linewidth=1.0, zorder=1)
ax_r.bar(x_age + age_offsets_adni[1], ADNI_age[1], width=bar_w, color=age_colors_adni[1], edgecolor="black", linewidth=1.0, zorder=1)

# Ticks/labels
ax.set_xticks([x_control, x_mci, x_age])
ax.set_xticklabels(["Control", "LMCI", "Age"], fontsize=FS_LABEL)#, fontweight="bold")
ax.set_ylabel("Mean Volume Atrophy Rate (%)", fontsize=FS_LABEL)
ax.tick_params(axis="y", labelsize=FS_TICK)
ax_r.tick_params(axis="y", labelsize=FS_TICK)
ax.grid(axis="y", linestyle="--", alpha=0.35)

# Legends
region_legend_handles = [
    Patch(facecolor=mci_colors[0], edgecolor="black", label="Amygdala"),
    Patch(facecolor=mci_colors[1], edgecolor="black", label="ERC/TEC"),
    Patch(facecolor=mci_colors[2], edgecolor="black", label="Hippocampus"),
]
age_legend_handles_adni = [
    Patch(facecolor=age_colors_adni[0], edgecolor="black", label="Age: Control"),
    Patch(facecolor=age_colors_adni[1], edgecolor="black", label="Age: LMCI"),
]
leg1 = ax.legend(handles=region_legend_handles, loc="upper left", frameon=True, fontsize=FS_LEGEND)
ax.add_artist(leg1)
ax_r.legend(handles=age_legend_handles_adni, loc="upper right", frameon=True, fontsize=FS_LEGEND)

# ---------------- Panel 2: ADNI-34 ----------------
ax2 = axes[1]
ax2.set_title("ADNI 3/4", fontsize=FS_TITLE)#, fontweight="bold")

x_control2 = 0.0
x_mci2     = 1.2
x_ad2      = 2.4

# Regions (left y-axis)
ax2.bar(x_control2 + offsets, ADNI34_control, width=bar_w, color=con_colors, edgecolor="black", linewidth=1.0, zorder=2)
ax2.bar(x_mci2     + offsets, ADNI34_mci,     width=bar_w, color=mci_colors, edgecolor="black", linewidth=1.0, zorder=2)
ax2.bar(x_ad2      + offsets, ADNI34_ad,      width=bar_w, color=ad_colors,  edgecolor="black", linewidth=1.0, zorder=2)

# Ages (right y-axis)
ax2_r = ax2.twinx()
ax2_r.set_ylim(65, 85)
ax2_r.set_ylabel("Age (years)", fontsize=FS_LABEL)

x_age2 = 3.9
age_offsets_adni34 = np.array([-bar_w, 0.0, bar_w])
for i, age in enumerate(ADNI34_age):
    ax2_r.bar(x_age2 + age_offsets_adni34[i], age, width=bar_w, color=age_colors_adni34[i],
              edgecolor="black", linewidth=1.0, zorder=1)

# Ticks/labels
ax2.set_ylim(0,9)
ax2.set_xticks([x_control2, x_mci2, x_ad2, x_age2])
ax2.set_xticklabels(["Control", "MCI", "AD", "Age"], fontsize=FS_LABEL)#, fontweight="bold")
ax2.set_ylabel("Mean Volume Atrophy Rate (%)", fontsize=FS_LABEL)
ax2.tick_params(axis="y", labelsize=FS_TICK)
ax2_r.tick_params(axis="y", labelsize=FS_TICK)
ax2.grid(axis="y", linestyle="--", alpha=0.35)

# Legends
age_legend_handles_adni34 = [
    Patch(facecolor=age_colors_adni34[0], edgecolor="black", label="Age: Control"),
    Patch(facecolor=age_colors_adni34[1], edgecolor="black", label="Age: MCI"),
    Patch(facecolor=age_colors_adni34[2], edgecolor="black", label="Age: AD"),
]
leg2 = ax2.legend(handles=region_legend_handles, loc="upper left", frameon=True, fontsize=FS_LEGEND)
ax2.add_artist(leg2)
ax2_r.legend(handles=age_legend_handles_adni34, loc="upper right", frameon=True, fontsize=FS_LEGEND)

# Aesthetics
for a in (ax, ax2):
    a.set_axisbelow(True)
    for spine in ["top", "right"]:
        a.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("Figure/ADNI_VA.png", dpi=300, bbox_inches="tight")
plt.close()
