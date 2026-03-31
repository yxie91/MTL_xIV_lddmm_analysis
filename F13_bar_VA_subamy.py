import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

con_mean=np.array([1.0556899144185825,1.0893315658321103,2.0526840144273804,1.4601754502409203,1.5279125292793743])#20 subjects
pre_mean=np.array([2.8620570433668298,5.446479276127316,4.708261992332531,2.234001016680059,3.4661472804758957])#10 subjects
conpre_mean=(con_mean*20+pre_mean*10)/30
print(conpre_mean)
mci_mean=np.array([8.240469539523467,8.313832947463979,6.340696529803018,2.523134059362458,6.323075425748518])#13 subjects

amybot,amytop=200,1900
ercbot,erctop=200,2600
hippobot,hippotop=2000,4500
labels = ["BLA","BMA","CMA","LA","All"]
conpre_colors= ["#90EE90","#DAA520","#FF7F7F","#DDA0DD","#1E90FF"]
mci_colors= ["#006400", "#8B6508", "#8B0000", "#4B0082","#00008B"]
x = np.arange(len(labels)) 
bar_width = 0.4
fontsize=35
fig, ax1 = plt.subplots(1, 1, figsize=(10,10))
ax1.bar(x - bar_width/2, conpre_mean, bar_width, color=conpre_colors,edgecolor='black')
ax1.bar(x + bar_width/2, mci_mean, bar_width, color=mci_colors,edgecolor='black')


ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=fontsize)
ax1.set_ylabel('Mean Volume Atrophy Rate', fontsize=fontsize)
ax1.set_ylim(0,9)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.tick_params(axis='y', labelsize=20)
# Create tuples of patches for legend
L_patches = tuple(Patch(facecolor=c, edgecolor='black', linewidth=1.5) for c in conpre_colors)
R_patches = tuple(Patch(facecolor=c, edgecolor='black', linewidth=1.5) for c in mci_colors)

ax1.legend(
    [(L_patches), (R_patches)],
    ['Control', 'MCI/AD'],
    handler_map={tuple: HandlerTuple(ndivide=None)},
    fontsize=fontsize-5,
    loc="upper right",
    frameon=True,
    handlelength=5,     # ← increase width
    #handleheight=3 
)
plt.tight_layout()
plt.savefig(f"/cis/home/yxie91/paper1/Figure/F13_subamyVA_bar.png",dpi=300, bbox_inches="tight")
plt.close()