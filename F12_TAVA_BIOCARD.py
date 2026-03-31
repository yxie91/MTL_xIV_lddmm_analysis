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
from matplotlib.ticker import FuncFormatter
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
    control_DATA=pd.read_excel('Data/Sheets/control_SUBJECT_DOB_multiple.xlsx',index_col=0,header=0)
    MCIDEM_DATA=pd.read_excel('Data/Sheets/MCIDEM_SUBJECT_DOB_multiple.xlsx',index_col=0,header=0)
    TP_pd=pd.read_excel('Data/Sheets/SUBJECT_TP_PET.xlsx',index_col=0,header=0)
    seg_root='/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset005_BIOCARD91/labelsTs_mprage_LH'#'Data/Dataset006_axis0/imagesTs_pred_mprage'
    DATA=pd.concat([control_DATA,MCIDEM_DATA])


    #LH
    narrow = os.listdir("Data/BIOCARD_Surface/LH_MCI_thickness/narrow")
    wide = os.listdir("Data/BIOCARD_Surface/LH_MCI_thickness/wide")
    weird = os.listdir("Data/BIOCARD_Surface/LH_MCI_thickness/weird")
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
    subject_narrow=sorted(list(set([f.split('_')[0] for f in narrow])))
    subject_wide=sorted(list(set([f.split('_')[0] for f in wide])))
    subject_weird=sorted(list(set([f.split('_')[0] for f in weird])))
    outlier_subjects=["PARKAT","SHEMIC"]#["MESDAV","SILKAT","WATKAT","CONLUC","HILCAR","SAUCON","WILJAY","LOGLYN","COMSUS","DUNPET","ROSJAN","ALBSUS","MORCAR"]
    SUBJECT=sorted([s for s in SUBJECT if s not in outlier_subjects])
    TAtrophy=[]
    VAtrophy=[]
    fontsize=30
    name=[]
    for subject in SUBJECT:
        print(subject)
        if subject in subject_narrow:
            root_dir="Data/BIOCARD_Surface/LH_MCI_thickness/narrow"
        elif subject in subject_wide:
            root_dir="Data/BIOCARD_Surface/LH_MCI_thickness/wide"
        elif subject in subject_weird:
            root_dir="Data/BIOCARD_Surface/LH_MCI_thickness/weird"
        else:
            raise ValueError(f"{subject} should be in weird group but not")

        #print(root_dir)
        TIMEPOINTS=TP_pd.loc[subject].dropna().values
        dob=DATA.loc[subject,'DOB']
        date_format = "%y%m%d"
        ages=[]
        thickness_l=[]
        volumes=[]
        for i in range(len(TIMEPOINTS)): 
            tp=TIMEPOINTS[i]
            tp=str(int(tp))
            #print(f"{root_dir}/{subject}_{tp}_thickness.vtk")
            if not os.path.exists(f"{root_dir}/{subject}_{tp}_thickness.vtk"):
                continue
            datetime1 = dob
            datetime2 = datetime.strptime(tp, date_format)
            age = datetime2.year - datetime1.year
            if (datetime2.month, datetime2.day) < (datetime1.month, datetime1.day):
                age -= 1
            ages.append(age)
            thick_path=os.path.join(root_dir,f"{subject}_{tp}_thickness.vtk")
            thickness=pv.read(thick_path)["displacement"]
            median_displacement = np.median(np.ravel(thickness))
            thickness_l.append(median_displacement)
            seg_path= seg_root + f'/{subject}_{tp}.nii.gz'
            seg_img= np.array(nib.load(seg_path).get_fdata()).squeeze()
            mask = np.ones_like(seg_img)
            voxel=np.append(mask[seg_img==2],mask[seg_img==3])
            volume=1.2*1*1*voxel.shape[0]
            volumes.append(volume)
        if len(thickness_l)!=len(volumes):
            raise ValueError(f"{subject} has different size in Thickness and Volumes")
        if len(thickness_l)<3:
            print(f"{subject} does not have 3 or more tps")
            continue
        ta=atrophy_cal(ages,thickness_l)
        TAtrophy.append(ta)
        va=atrophy_cal(ages,volumes)
        VAtrophy.append(va)
        print(va,ta)
        name.append(subject)


    #RH
    narrow = os.listdir("Data/BIOCARD_Surface/RH_MCI_thickness/narrow")
    wide = os.listdir("Data/BIOCARD_Surface/RH_MCI_thickness/wide")
    weird = os.listdir("Data/BIOCARD_Surface/RH_MCI_thickness/weird")
    seg_root='/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset006_axis0/imagesTs_pred_RH'
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
    subject_narrow=sorted(list(set([f.split('_')[0] for f in narrow])))
    subject_wide=sorted(list(set([f.split('_')[0] for f in wide])))
    subject_weird=sorted(list(set([f.split('_')[0] for f in weird])))
    outlier_subjects=["COMSUS","TUCJUD"]
    SUBJECT=sorted([s for s in SUBJECT if s not in outlier_subjects])
    for subject in SUBJECT:
        print(subject)
        if subject in subject_narrow:
            root_dir="Data/BIOCARD_Surface/RH_MCI_thickness/narrow"
        elif subject in subject_wide:
            root_dir="Data/BIOCARD_Surface/RH_MCI_thickness/wide"
        elif subject in subject_weird:
            root_dir="Data/BIOCARD_Surface/RH_MCI_thickness/weird"
        else:
            raise ValueError(f"{subject} should be in weird group but not")

        TIMEPOINTS=TP_pd.loc[subject].dropna().values
        dob=DATA.loc[subject,'DOB']
        date_format = "%y%m%d"
        ages=[]
        thickness_l=[]
        volumes=[]
        for i in range(len(TIMEPOINTS)): 
            tp=TIMEPOINTS[i]
            tp=str(int(tp))
            #print(f"{root_dir}/{subject}_{tp}_thickness.vtk")
            if not os.path.exists(f"{root_dir}/{subject}_{tp}_thickness.vtk"):
                continue
            datetime1 = dob
            datetime2 = datetime.strptime(tp, date_format)
            age = datetime2.year - datetime1.year
            if (datetime2.month, datetime2.day) < (datetime1.month, datetime1.day):
                age -= 1
            ages.append(age)
            thick_path=os.path.join(root_dir,f"{subject}_{tp}_thickness.vtk")
            thickness=pv.read(thick_path)["displacement"]
            median_displacement = np.median(np.ravel(thickness))
            thickness_l.append(median_displacement)
            seg_path= seg_root + f'/{subject}_{tp}.nii.gz'
            seg_img= np.array(nib.load(seg_path).get_fdata()).squeeze()
            mask = np.ones_like(seg_img)
            voxel=np.append(mask[seg_img==2],mask[seg_img==3])
            volume=1.2*1*1*voxel.shape[0]
            volumes.append(volume)
        if len(thickness_l)!=len(volumes):
            raise ValueError(f"{subject} has different size in Thickness and Volumes")
        if len(thickness_l)<3:
            print(f"{subject} does not have 3 or more tps")
            continue
        ta=atrophy_cal(ages,thickness_l)
        TAtrophy.append(ta)
        va=atrophy_cal(ages,volumes)
        VAtrophy.append(va)
        print(va,ta)
        name.append(subject)
    #print(TAtrophy,VAtrophy)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    model = LinearRegression()
    model.fit(np.array(TAtrophy).reshape(-1,1),np.array(VAtrophy))
    x_min, x_max = min(TAtrophy), max(TAtrophy)
    x_linspace = np.linspace(x_min, x_max, 50)
    y_linspace_pred = model.predict(x_linspace.reshape(-1, 1))
    #plt.plot(x_linspace, x_linspace, '--', linewidth=3,color="black")
    slope = model.coef_[0]
    intercept = model.intercept_
    equation = f'y = {slope:.2f}x{intercept:.2f}'
    print(equation)
    plt.scatter(TAtrophy,VAtrophy,label=f'ERCTEC', color="blue",alpha=0.7, marker='o', s=60, edgecolor='black')
    '''for i, subj in enumerate(name):
        plt.annotate(subj, (TAtrophy[i], VAtrophy[i]),
                    textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)'''
    color="#0000CC"#"#CCDFFF"
    add_fit_ci_pi(ax, TAtrophy, VAtrophy,  color=color,label=f"BIOCARD",lw=5, ci_alpha=0.6, pi_alpha=0.0)     
    ax.set_ylabel("Volume Atrophy Rate",fontsize=fontsize)
    ax.set_xlabel("Thickness Atrophy Rate",fontsize=fontsize)
    ax.set_xlim(-1.5,7)
    ax.set_ylim(-1.5,7)
    ax.grid(True, alpha=0.2)
    label = f"BIOCARD, ERC/TEC"
    line = Line2D([], [], linestyle="--", lw=5, color=color)
    ci_proxy = Patch(facecolor=color, edgecolor='black', linewidth=1.5)
    ax.legend(handles=[line, ci_proxy],
              labels=[label,  "95% CI"],
              handler_map={tuple: HandlerTuple(ndivide=None)},
              fontsize=18, loc='upper right', frameon=True)
    ticks = np.arange(0,9,5)#(-1,9, 2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    fmt = FuncFormatter(lambda x, _: f"{x:.0f}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(axis='both', labelsize=22)
    plt.tight_layout()
    if tag=="narrow & wide":
        tag1='BIOCARD'
    else:
        tag1=tag
    plt.savefig(f"Figure/F12_VATA_BIOCARD.png",dpi=300, bbox_inches="tight")
    plt.close()
#main("narrow")
#main("wide")
#main("weird")
main("narrow & wide")
