from logging import raiseExceptions
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import statsmodels.api as sm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import FuncFormatter
# ---------- helpers ----------

class _HandlerSingleCI(HandlerBase):
    """Legend handler that draws one CI color box."""
    def __init__(self, color, alpha=0.5):
        super().__init__()
        self.color = color
        self.alpha = alpha

    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        r = Rectangle((x0, y0), width, height, facecolor=self.color,
                      edgecolor='black', alpha=self.alpha, transform=trans)
        return [r]
    
def clean_xy(x, y):
    df = pd.concat([pd.Series(x, name="x"), pd.Series(y, name="y")], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df["x"].to_numpy(), df["y"].to_numpy()

def add_fit_ci_pi(ax, x, y, color, label,
                  lw=5, ci_alpha=0.40, pi_alpha=0.20, ngrid=200):
    if len(x) < 2:
        return
    x, y = clean_xy(x, y)
    X = sm.add_constant(x)
    model = sm.OLS(y, X, hasconst=True).fit()
    xg = np.linspace(np.min(x), np.max(x), ngrid)
    Xg = sm.add_constant(xg)
    sf = model.get_prediction(Xg).summary_frame(alpha=0.05)  # 95%
    yfit  = sf["mean"].to_numpy()
    ci_lo = sf["mean_ci_lower"].to_numpy()
    ci_hi = sf["mean_ci_upper"].to_numpy()
    pi_lo = sf["obs_ci_lower"].to_numpy()
    pi_hi = sf["obs_ci_upper"].to_numpy()

    ax.plot(xg, yfit, "--", lw=lw, color=color, alpha=0.95)
    ax.fill_between(xg, ci_lo, ci_hi, color=color, alpha=ci_alpha)
    ax.fill_between(xg, pi_lo, pi_hi, color=color, alpha=pi_alpha)
def main(referencetype): 
    DATA=pd.read_excel(f'Results/ADNI_MTLall_FUSE_{referencetype}.xlsx',index_col=0)
    LDATA=pd.read_excel(f'Results/ADNI_MTLall_LH_{referencetype}.xlsx',index_col=0)
    RDATA=pd.read_excel(f'Results/ADNI_MTLall_RH_{referencetype}.xlsx',index_col=0)
    LA_subjects=[]
    HA_subjects=[]
    consisitency_outlier=["114_S_6597","130_S_2373","116_S_6775","007_S_6341","305_S_6810","067_S_4782","035_S_6480","035_S_7105","014_S_6765","016_S_6816","003_S_0908","305_S_6498","019_S_6483"]
    tauva_outlier=["041_S_4874","141_S_1052","006_S_6657"]+["035_S_7073","168_S_6828"]#ERC/TEC+Hippocampus
    print(len(set(consisitency_outlier))+len(set(tauva_outlier)))
    print(len(sorted(DATA.index.values)))
    SUBJECT=[s for s in sorted(DATA.index.values) if s not in consisitency_outlier and s not in tauva_outlier]#
    print(len(SUBJECT))
    median_amyloid=np.median(DATA.loc[SUBJECT,"MTL Amyloid"].values)
    #print(len(DATA.loc[SUBJECT,"MTL Amyloid"].values))
    print(median_amyloid)
    for subject in SUBJECT:
        amyloid = DATA.loc[subject, "MTL Amyloid"].tolist()
        if amyloid < median_amyloid:
            LA_subjects.append(subject)
        else:
            HA_subjects.append(subject) 

    LH_LA_DATA=LDATA.loc[list(set(LA_subjects))]
    LH_HA_DATA=LDATA.loc[list(set(HA_subjects))]

    RH_LA_DATA=RDATA.loc[list(set(LA_subjects))]
    RH_HA_DATA=RDATA.loc[list(set(HA_subjects))]
    
    regions=["Amygdala","ERC/TEC","Hippocampus"]
    palette = {"Amygdala":("#FFCCCC","#CC0000"), "ERC/TEC":("#CCDFFF","#0000CC"), "Hippocampus":("#CCFFCC","#009900")}
    xrange = {"Amygdala":(-2,12), "ERC/TEC":(-2,12), "Hippocampus":(-2,12)}

    fig, axes = plt.subplots(1, 1, figsize=(8, 8), sharey=True)
    Xmin=[]
    Xmax=[]
    fontsize = 22
    ax = axes
    for i in range(1,3):
        r=regions[i]
        COLOR_NEG, COLOR_POS = palette[r]
        HA_Tau=LH_HA_DATA[f"{r} Tau"]
        LA_Tau=LH_LA_DATA[f"{r} Tau"]
        LH_Tau=pd.concat([HA_Tau,LA_Tau],axis=0)
        x_min, x_max = min(LH_Tau)-0.1, max(LH_Tau)+0.2
        #print(VA.shape)
        #print(HA_VA)
        HA_Tau=RH_HA_DATA[f"{r} Tau"]
        LA_Tau=RH_LA_DATA[f"{r} Tau"]
        RH_Tau=pd.concat([HA_Tau,LA_Tau],axis=0)
        Xmin.append(x_min)
        Xmax.append(x_max)
        
        ax.scatter(LH_Tau, RH_Tau, alpha=0.85, label=r, color=COLOR_POS, edgecolor='black', s=80)
        '''for i, subj in enumerate(list(VA.index.values)):
            ax.annotate(subj, (VA.values[i], Tau.values[i]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=10)'''
    x_linspace = np.linspace(min(Xmin), max(Xmax), 50)
    y_linspace_pred = x_linspace
    ax.plot(x_linspace, y_linspace_pred, '--', color='black', linewidth=2)
    ax.set_xlabel("Left Hemisphere Tau PET Accumulation", fontsize=fontsize)
    ax.set_ylabel("Right Hemisphere Tau PET Accumulation", fontsize=fontsize)
    ax.set_xlim((0.4, 2.6))
    ax.set_ylim((0.4, 2.6))
    ticks = np.arange(0.5, 2.51, 0.5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    fmt = FuncFormatter(lambda x, _: f"{x:.1f}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(axis='both', labelsize=18)
    ax.legend(fontsize=14, loc='upper left', frameon=True)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    outname = f'Figure/F15_ADNI34_ERCHippo_Tau_consistency_{referencetype}.png'
    plt.savefig(outname, dpi=300, bbox_inches="tight")

#main("Pons")
main("CerebellumGM")