import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FS_TITLE  = 30
FS_LABEL  = 25
FS_TICK   = 20
FS_LEGEND = 22

ADNIall_control_mean=[1.69491038, 1.84797416, 0.82880703]
ADNIall_control_std=[2.88660485, 3.27339797, 0.91302707]
ADNIall_mci_mean=[3.26346854, 3.53291121, 1.65121555]
ADNIall_mci_std=[4.34840915, 4.5360637,  1.77255352]
ADNIall_ad_mean=[7.18913829, 8.39554379, 3.22861176]
ADNIall_ad_std=[5.60275243, 6.20658187, 2.12304738]
ADNIall_age=[75.35547034093797,73.80131208883863,75.08473941228307]  # control, mci, ad
# Colors
con_colors = ["#FF9999", "#99CCFF", "#99FF99"]
mci_colors = ["#FF6666", "#6699FF", "#66FF66"]
ad_colors  = ["#CC0000", "#0000CC", "#009900"]
age_colors_adni   = ["#CFCFCF", "#6F6F6F"]
age_colors_ADNIall = ["#CFCFCF", "#8C8C8C", "#4A4A4A"]

regions = ["Amygdala", "ERC/TEC", "Hippocampus"]
bar_w   = 0.25

fig, ax2 = plt.subplots(1, 1, figsize=(15, 15), sharey=True)

x_control = 0.0
x_mci     = 1.4
offsets = np.array([-bar_w, 0.0, bar_w])

ax2.set_title("ADNI", fontsize=FS_TITLE)#, fontweight="bold")

x_control2 = 0.0
x_mci2     = 1.2
x_ad2      = 2.4

# Regions (left y-axis)
ax2.bar(x_control2 + offsets, ADNIall_control_mean, width=bar_w, color=con_colors, edgecolor="black", linewidth=1.0)#, yerr=ADNIall_control_std, capsize=5)
ax2.bar(x_mci2     + offsets, ADNIall_mci_mean,     width=bar_w, color=mci_colors, edgecolor="black", linewidth=1.0)#, yerr=ADNIall_mci_std, capsize=5)
ax2.bar(x_ad2      + offsets, ADNIall_ad_mean,      width=bar_w, color=ad_colors,  edgecolor="black", linewidth=1.0)#, yerr=ADNIall_ad_std, capsize=5)

# Ages (right y-axis)
ax2_r = ax2.twinx()
ax2_r.set_ylim(65, 85)
ax2_r.set_ylabel("Age (years)", fontsize=FS_LABEL)

x_age2 = 3.9
age_offsets_ADNIall = np.array([-bar_w, 0.0, bar_w])
for i, age in enumerate(ADNIall_age):
    ax2_r.bar(x_age2 + age_offsets_ADNIall[i], age, width=bar_w, color=age_colors_ADNIall[i],
              edgecolor="black", linewidth=1.0, zorder=1)

# Ticks/labels
ax2.set_ylim(0,9)
ax2.set_xticks([x_control2, x_mci2, x_ad2, x_age2])
ax2.set_xticklabels(["Control", "MCI", "AD", "Age"], fontsize=FS_LABEL)#, fontweight="bold")
ax2.set_ylabel("Mean Volume Atrophy Rate (%)", fontsize=FS_LABEL)
'''ax2.set_xticks([x_control2, x_mci2, x_ad2])
ax2.set_xticklabels(["Control", "MCI", "AD"], fontsize=FS_LABEL)#, fontweight="bold")
ax2.set_ylabel("Mean Volume Atrophy Rate (%)", fontsize=FS_LABEL)'''
ax2.tick_params(axis="y", labelsize=FS_TICK)
ax2_r.tick_params(axis="y", labelsize=FS_TICK)
ax2.grid(axis="y", linestyle="--", alpha=0.35)

# Legends
region_legend_handles = [
    Patch(facecolor=mci_colors[0], edgecolor="black", label="Amygdala"),
    Patch(facecolor=mci_colors[1], edgecolor="black", label="ERC/TEC"),
    Patch(facecolor=mci_colors[2], edgecolor="black", label="Hippocampus"),
]
age_legend_handles_ADNIall = [
    Patch(facecolor=age_colors_ADNIall[0], edgecolor="black", label="Age: Control"),
    Patch(facecolor=age_colors_ADNIall[1], edgecolor="black", label="Age: MCI"),
    Patch(facecolor=age_colors_ADNIall[2], edgecolor="black", label="Age: AD"),
]
leg2 = ax2.legend(handles=region_legend_handles, loc="upper left", frameon=True, fontsize=FS_LEGEND)
ax2.add_artist(leg2)
#ax2_r.legend(handles=age_legend_handles_ADNIall, loc="upper right", frameon=True, fontsize=FS_LEGEND)

# Aesthetics
'''for a in (ax, ax2):
    a.set_axisbelow(True)
    for spine in ["top", "right"]:
        a.spines[spine].set_visible(False)'''

for spine in ["top", "right", "left", "bottom"]:
    ax2.spines[spine].set_visible(True)

plt.tight_layout()
plt.savefig("Figure/F8_ADNIall_VA.png", dpi=300, bbox_inches="tight")
plt.close()
