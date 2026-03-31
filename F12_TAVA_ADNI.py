import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import scipy.io as sio
from sklearn.linear_model import LinearRegression
from datetime import datetime
import statsmodels.api as sm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.legend_handler import HandlerBase
from matplotlib.legend_handler import HandlerTuple
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

    ax.plot(xg, yfit, "--", lw=lw, color=color, label=label, alpha=0.95)
    ax.fill_between(xg, ci_lo, ci_hi, color=color, alpha=ci_alpha, label="95% CI")
    ax.fill_between(xg, pi_lo, pi_hi, color=color, alpha=pi_alpha, label="95% PI")
    
def atrophy_cal(ages,thickness):
    ages = np.array(ages).reshape(-1, 1)
    thickness = np.array(thickness)
    model = LinearRegression()
    model.fit(ages, thickness)
    vol0=thickness[0]
    atrophy_rate=-model.coef_[0]/vol0*100
    return atrophy_rate

def main(tag):
    narrow = [
        "0023_76.8","0023_77.8","0023_78.8","0023_79.8",
        "0413_0","0413_06","0413_12","0413_24",
        "0605_0","0605_06","0605_12","0605_24",
        "4030_06","4030_12","4030_24",
        "4200_0","4200_06","4200_12","4200_24","4200_74.2",
        "4270_0","4270_03","4270_06","4270_12","4270_24","4270_48",
        "4429_0","4429_06","4429_12","4429_24","4429_81.1",
        "4488_72.6","4488_73.1","4488_73.6","4488_76.6",
        "4516_71.4","4516_71.9","4516_72.4","4516_73.4","4516_75.4",
        "4714_0","4714_06","4714_12","4714_24",
        "4757_0","4757_06","4757_12","4757_24",
        "4796_0","4796_06","4796_12","4796_24",
        "4878_0","4878_06","4878_12","4878_24",
        "4951_71.9","4951_72.9","4951_74.9"
    ]

    wide = [
        "0086_80.3","0086_80.8","0086_81.3","0086_82.3","0086_83.3",
        "1169_72.2","1169_73.3","1169_74.3","1169_75.2",
        "1249_70.8","1249_71.8","1249_72.8","1249_73.8",
        "1268_0","1268_12","1268_18","1268_24","1268_86.8",
        "4020_0","4020_06","4020_12","4020_24","4020_70.5",
        "4058_0","4058_06","4058_12",
        "4084_0","4084_06","4084_12","4084_24","4084_72.4",
        "4173_0","4173_06","4173_12","4173_24","4173_74.2",
        "4250_0","4250_06","4250_12","4250_24",
        "4263_0","4263_06","4263_12","4263_24",
        "4414_0","4414_06","4414_12","4414_24",
        "4422_70.8","4422_71.4","4422_71.8","4422_72.8","4422_74.8",
        "4448_0","4448_06","4448_12","4448_24","4448_68",
        "4552_0","4552_06","4552_12","4552_24","4552_67.4",
        "4560_70.3","4560_70.8","4560_71.2","4560_72.3",
        "4720_0","4720_06","4720_12","4720_24",
        "4739_0","4739_06","4739_12","4739_24",
        "4762_0","4762_06","4762_12","4762_24",
        "4885_0","4885_06","4885_12",
        "4888_0","4888_06","4888_12","4888_18","4888_24",
        "4929_0","4929_06","4929_12"
    ]

    weird = [
        "0260_0","0260_06","0260_12","0260_24",
        "0842_78.6","0842_79.7","0842_80.7","0842_81.6",
        "1098_0","1098_06","1098_12","1098_24","1098_48",
        "1123_75.8","1123_76.3","1123_76.8","1123_77.8",
        "1206_72.9","1206_73.4","1206_74.9",
        "4041_77.9","4041_78.4","4041_78.9","4041_79.8",
        "4103_0","4103_06","4103_12","4103_24",
        "4148_73","4148_74","4148_75","4148_77",
        "4177_0","4177_06","4177_12","4177_24","4177_89",
        "4218_80.7","4218_80.9","4218_81.7","4218_82.7",
        "4262_0","4262_03","4262_06","4262_12","4262_24","4262_48","4262_60",
        "4276_73.9","4276_74.5","4276_74.9","4276_75.9","4276_77.9",
        "4278_75","4278_75.5","4278_76","4278_77","4278_79",
        "4279_83.7","4279_84.2","4279_84.7","4279_85.9",
        "4313_0","4313_06","4313_12","4313_24","4313_82",
        "4350_72.9","4350_73.4","4350_73.9","4350_74.9","4350_76.9",
        "4385_68.2","4385_68.4","4385_68.7","4385_69.2","4385_70.2","4385_72.1",
        "4391_0","4391_06","4391_12","4391_24",
        "4393_0","4393_06","4393_12","4393_24","4393_77.6",
        "4453_0","4453_06","4453_12","4453_24","4453_70.1",
        "4505_0","4505_06","4505_12","4505_24","4505_84.4",
        "4566_83.4","4566_83.6","4566_83.9","4566_84.4","4566_85.4",
        "4579_84.9","4579_85.9","4579_86.9",
        "4598_0","4598_06","4598_12","4598_24","4598_69.1",
        "4604_65","4604_65.5","4604_66","4604_66.9",
        "4644_67.6","4644_68.2","4644_68.6","4644_72.7",
        "4668_0","4668_06","4668_12",
        "5166_65.3","5166_65.1","5166_67.1"
    ]

    control_DATA=sio.loadmat('Data/Sheets/rawthk_v13_kms_con.mat')
    pre_DATA=sio.loadmat('Data/Sheets/rawthk_v13_kms_pre.mat')
    mci_DATA=sio.loadmat('Data/Sheets/rawthk_v13_kms_mci.mat')
    SUBJECT=list(control_DATA["SUBJSc"])+list(pre_DATA["SUBJSp"])+list(mci_DATA["SUBJSm"])
    AGE=list(control_DATA["x1_c"])+list(pre_DATA["x1_p"])+list(mci_DATA["x1_m"])
    SUBJECT=[subject[0][0] for subject in SUBJECT]
    AGE=[age[0][0] for age in AGE]
    subject_age_dict={}
    for i in range(len(SUBJECT)):
        subject_age_dict[SUBJECT[i]]=AGE[i]
    manual_root="Data/ADNI12GO_manual"
    predicted_root="Data/Dataset103_ADNIall/labelsTs"
    TA_l=[]
    VA_l=[]
    name=[]
    fontsize=30
    if tag=="narrow":
        SUBJECT=sorted(list(set([f.split('_')[0] for f in narrow])))
    elif tag=="wide":
        SUBJECT=sorted(list(set([f.split('_')[0] for f in wide])))
    elif tag=="weird":
        SUBJECT=sorted(list(set([f.split('_')[0] for f in weird])))
    elif tag=="narrow & wide":
        SUBJECT=sorted(list(set([f.split('_')[0] for f in narrow+wide])))
    else:
        raise ValueError("Surface group not right")
    outlier_subjects=["0605"]
    SUBJECT=sorted([s for s in SUBJECT if s not in outlier_subjects])
    for i in range(len(SUBJECT)):
        subject=SUBJECT[i]
        ages=subject_age_dict[subject]
        manual_segfiles=sorted([file for file in os.listdir(f"{manual_root}/{subject}") if "cy2.nii.gz" in file], key=lambda x: float(x.split('_')[1]))

        if len(manual_segfiles)!=len(ages):
            print(f"{subject} is skipped due to segfile-age consistency")
            continue
        no_thickness=False
        for k in range(len(manual_segfiles)):
            prefix=f"{manual_segfiles[k].split('_')[0]}_{manual_segfiles[k].split('_')[1]}"
            if not os.path.exists(f"Data/ADNI_Surface/thickness/{prefix}_thickness.vtk"):
                print(f"{prefix} does not have thickness file")
                no_thickness=True
                continue
        if no_thickness:
            print(f"{manual_segfiles} does not have all the thickness")
            continue
        volumes=[]
        thickness_l=[]
        for j in range(len(manual_segfiles)):
            seg_path=f"{manual_root}/{subject}/{manual_segfiles[j]}"
            manual_seg= np.array(nib.load(seg_path).get_fdata()).squeeze()
            #seg_path=f"{predicted_root}/{pred_segfiles[j]}"
            #pred_seg= np.array(nib.load(seg_path).get_fdata()).squeeze()
            seg_img=manual_seg
            mask = np.ones_like(seg_img)
            voxel=np.append(mask[seg_img==2],mask[seg_img==3])
            volume=1.2*1*1*voxel.shape[0]
            volumes.append(volume)
            #print(volumes)
            prefix=f"{manual_segfiles[j].split('_')[0]}_{manual_segfiles[j].split('_')[1]}"
            #print(ages[j],prefix)
            thick_path=f"Data/ADNI_Surface/thickness/{prefix}_thickness.vtk"
            thickness=pv.read(thick_path)["displacement"]
            median_displacement = np.median(np.ravel(thickness))
            thickness_l.append(median_displacement)
        TA=atrophy_cal(ages,thickness_l)
        VA=atrophy_cal(ages,volumes)
        print(TA,VA)
        TA_l.append(TA)
        VA_l.append(VA)
        name.append(subject)
    TA_l=np.array(TA_l)
    VA_l=np.array(VA_l)

    color="#0000CC"
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.scatter(TA_l,VA_l, alpha=0.7, color="blue", marker='o', s=60, edgecolor='black')
    '''for i, subj in enumerate(name):
        plt.annotate(subj, (TA_l[i], VA_l[i]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8) '''
    add_fit_ci_pi(ax, TA_l, VA_l,  color=color,label=f"BIOCARD",lw=5, ci_alpha=0.6, pi_alpha=0.0)     
    ax.set_ylabel("Volume Atrophy Rate",fontsize=fontsize)
    ax.set_xlabel("Thickness Atrophy Rate",fontsize=fontsize)
    
    ax.set_xlim(-3,18)
    ax.set_ylim(-3,18)
    ax.tick_params(axis='both', labelsize=22)
    ax.grid(True, alpha=0.2)
    if tag=="narrow & wide":
        tag1="ADNI"
    else:
        tag1=tag
    label = f"{tag1}, ERC/TEC"
    line = Line2D([], [], linestyle="--", lw=5, color=color)
    ci_proxy = Patch(facecolor=color, edgecolor='black', linewidth=1.5)
    ax.legend(handles=[line, ci_proxy],
              labels=[label,  "95% CI"],
              handler_map={tuple: HandlerTuple(ndivide=None)},
              fontsize=18, loc='upper left', frameon=True)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.yaxis.get_major_locator().set_params(integer=True)
    plt.tight_layout()
    plt.savefig(f"Figure/F12_VATA_ADNI.png", dpi=300, bbox_inches="tight")
    plt.close()
#main("narrow")
#main("wide")
main("narrow & wide")
#main("weird")