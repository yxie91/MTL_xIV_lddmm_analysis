import os
from ast import Str, Sub
from cmath import tau
from datetime import datetime
from math import nan
from pickletools import uint8

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io as sio
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels import robust
from collections import defaultdict
from collections import Counter


'''ADNI LH
257.5261245609133 347.18536214827424 481.0771039527792
ADNI RH
251.12306731868117 331.0693117630194 478.7100448813985
ADNI fuse
491.2706445585293,634.1674837765971,933.8100679831414'''

def STD_volumes(folder, outlier_path="",tag=""):
    resolution = 1.2 * 1.0 * 1.0
    if outlier_path!="":
        outlier_set=pd.read_excel(outlier_path,index_col=0).index.values
    else:
        outlier_set=[]
    files=[s for s in os.listdir(folder) if s[:-7] not in outlier_set]
    print(len(os.listdir(folder)),len(files))
    subjects_scandate=list(set(['__'.join(s.split('__')[:2]) for s in files]))
    #print(subjects_scandate)
    seg_subjects=[s.split('__')[0] for s in subjects_scandate]
    #print(seg_subjects)
    subject_count = Counter(seg_subjects)
    SUBJECT = sorted([subj for subj, cnt in subject_count.items() if cnt >= 3])
    print(len(SUBJECT))

    V_amy=[];V_erc=[];V_hip=[]
    for file in sorted(files):
        if not file.endswith(".nii.gz"):
            continue
        if not file.split('__')[0] in SUBJECT:
            continue
        #print(file[:-7])
        if file[:-7] in outlier_set:
            #print(f"{file} is an outlier removed")
            raise ValueError(f"{file[:-7]} is an outlier but not filtered")
        seg_path = os.path.join(folder, file)
        seg = np.asarray(nib.load(seg_path).get_fdata()).squeeze()
        ones = np.ones_like(seg, dtype=np.uint8)

        # labels: 1 (amygdala), 2/3 (ERC/TEC), 4/5 (hippocampus)
        V_amy.append(resolution * ones[seg == 1].size)
        V_erc.append(resolution * (np.append(ones[seg == 2], ones[seg == 3]).size))
        V_hip.append(resolution * (np.append(ones[seg == 4], ones[seg == 5]).size))
    #print(tag)
    return np.std(V_amy),np.std(V_erc),np.std(V_hip)

def tp_atrophy_cal(ages,volumes):
    ages = np.array(ages).reshape(-1, 1)
    volumes = np.array(volumes)
    model = LinearRegression()
    model.fit(ages, volumes)
    vol0=volumes[0]
    atrophy_rate=-model.coef_[0]/vol0*100
    return atrophy_rate,model
def Atrophy_Plot(controltype, areatype,Hemi="LH",outlier_subject=[]):
    print('-------------------')
    print(Hemi, controltype, areatype)
    print('-------------------')
    DATA=pd.read_csv('/cis/project/adni_yxie91/ADNI_T1all/Datasheets/ADNI_T1all_2_04_2026.csv',index_col=1)
    #print(*list(DATA.index.values),sep=",")
    outlier_path='/cis/project/adni_yxie91/ADNI_T1all/Datasheets/ADNIall_Outlier_updated.xlsx'
    outlier_file=pd.read_excel(outlier_path,index_col=0).index.values
    if Hemi=="LH":
        predicted_root="/cis/project/adni_yxie91/ADNI_T1all/labelsTs_nnUNet"
    elif Hemi=="RH":
        predicted_root="/cis/project/adni_yxie91/ADNI_T1all/labelsTs_nnUNet_RH"
    elif Hemi=="FUSE":
        predicted_root="/cis/project/adni_yxie91/ADNI_T1all/labelsTs_nnUNet_fuse"
    else:
        raise TypeError('Hemishpere not recognized')

    if controltype=="Control":
        SUBJECT=sorted(list(set(list(DATA[DATA['Group'].isin(['CN', 'SMC'])].index.values))))
    elif controltype=="MCI":
        SUBJECT=sorted(list(set(list(DATA[DATA['Group'].isin(['EMCI', 'LMCI',"MCI"])].index.values))))
    elif controltype=="AD":
        SUBJECT=sorted(list(set(list(DATA[DATA['Group'].isin(["AD"])].index.values))))
    else:
        raise TypeError('Control type not recognized')
    subject_idx=0
    tp_idx=0
    outdict={}
    VA=[]
    AGE_avg=[]
    std_amy,std_ERCTEC,std_hippo=491.2706445585293,634.1674837765971,933.8100679831414
    #std_amy,std_ERCTEC,std_hippo=STD_volumes(predicted_root,outlier_path)
    #print(f"STD for three regions,amygdala:{std_amy},ERC:{std_ERCTEC},hippoampus:{std_hippo}")
    #STD for three regions,amygdala:479.8022500901658,ERC:637.8925133080927,hippoampus:887.6609674833333

    for i in range(len(SUBJECT)):
        subject=SUBJECT[i]
        segfiles=sorted([file for file in os.listdir(predicted_root) if subject in file])
        #print(segfiles)
        if segfiles == [] or subject in outlier_subject:
            continue
        subject_rows = DATA.loc[[subject]]
        #print(subject)
        #print(subject_rows['Acq Date'].tolist(), subject_rows['Age'].tolist())
        date2age = dict(zip(subject_rows['Acq Date'].tolist(), subject_rows['Age'].tolist()))
        date2age = {datetime.strptime(k, '%m/%d/%Y').date(): v for k, v in date2age.items()} #====Change it to the entry date and entry disease group maybe====
        baseline_tp=sorted(date2age.keys())[0]
        baseline_age=date2age[baseline_tp]
        valid_volumes=[]
        valid_ages=[]
        ages=[]
        volumes=[]
        for k in range(len(segfiles)):
            name=segfiles[k][:-7]
            if name in outlier_file: 
                #print(f"{name} not valid segfile")
                continue
            tp=datetime.strptime(name.split('__')[1], '%Y%m%d').date()
            age=baseline_age+(tp-baseline_tp).days/365.25
            seg_path=f"{predicted_root}/{segfiles[k]}"
            '''if "_accel_" in seg_path or "_grappa2_" in seg_path or "ir_spgr" in seg_path:
                continue'''
            seg_img= np.array(nib.load(seg_path).get_fdata()).squeeze()
            mask = np.ones_like(seg_img)
            if areatype=='Amygdala':
                voxel=mask[seg_img==1]
                volume=1.2*1*1*voxel.shape[0]
                std=std_amy
            elif areatype=='ERCTEC':
                voxel=np.append(mask[seg_img==2],mask[seg_img==3])
                volume=1.2*1*1*voxel.shape[0]
                std=std_ERCTEC
            elif areatype=='Hippocampus':
                voxel=np.append(mask[seg_img==4],mask[seg_img==5])
                volume=1.2*1*1*voxel.shape[0]
                std=std_hippo
            else:
                raise TypeError("Areatype not recognized")
            ages.append(age)
            volumes.append(volume)
        for k in range(len(ages)):
            volume=volumes[k]
            if np.abs(volume-np.mean(volumes))< 2.5 * std:
                age=ages[k]
                valid_ages.append(age)
                valid_volumes.append(volume)
            else:
                print(f"{segfiles[k]} is removed for 2.5 sigma")
        if len(set(valid_ages)) < 3:
            continue
        
        age_dict = defaultdict(list)
        for age, vol in zip(valid_ages, valid_volumes):
            age_dict[age].append(vol)
        group_valid_ages = []
        group_valid_volumes = []
        for age in sorted(age_dict.keys()):
            vals = np.asarray(age_dict[age], float)
            group_valid_ages.append(age)
            group_valid_volumes.append(sum(age_dict[age]) / len(age_dict[age]))

            if len(vals) > 1:
                    if len(vals) == 2:#Percent difference between the two measurements > 5%
                        pct_diff = abs(vals[0] - vals[1]) / vals.mean() * 100
                        msg = f"age={age}, #replicates=2, pct_diff={pct_diff:.2f}%"
                        if pct_diff > 5:
                            #print(subject)
                            msg += "  <-- WARNING: large replicate disagreement"
                            #print(msg)

                    else:
                        mean = vals.mean()#Z-score > 3 (3 standard deviations from the mean)
                        std  = vals.std(ddof=1)
                        z = np.abs((vals - mean) / std) if std > 0 else np.zeros_like(vals)
                        msg = f"age={age}, #replicates={len(vals)}, std={std:.4f}"
                        if std > 0 and np.any(z > 3):
                            #print(subject)
                            msg += "  <-- WARNING: potential outlier replicate"
                            #print(msg)
        subject_idx+=1
        assert len(set(valid_ages))==len(group_valid_ages) 
        tp_idx+=len(group_valid_ages)
        atrophy_rate,model=tp_atrophy_cal(group_valid_ages,group_valid_volumes)
        #print(subject,group_valid_ages,group_valid_volumes,atrophy_rate)
        VA.append(atrophy_rate)
        AGE_avg.append(np.mean(group_valid_ages))
        out=[atrophy_rate]+group_valid_volumes
        outdict[subject]=out
    #print(f"All has {subject_idx} subjects and {tp_idx} timepoints in total, {tp_idx/subject_idx:.3f} tps per subject")
    output=pd.DataFrame(outdict.values(),index=outdict.keys())
    num_columns = output.shape[1]
    column_names=['atrophy rate']+[f'tp{i}' for i in range(1,num_columns)]
    output.columns=column_names
    output.to_csv(f'/cis/home/yxie91/ADNI/ADNI_T1all_Result/3g_{Hemi}_{controltype}_{areatype}.csv',header=True)
    return np.mean(VA), np.std(VA), np.mean(AGE_avg)
if __name__ == "__main__":
    outlier_subject=[]
    Hemi="FUSE"
    controltype="Control"
    amymean,amystd,control_age=Atrophy_Plot(controltype=controltype, areatype="Amygdala", Hemi=Hemi,outlier_subject=outlier_subject)
    ercmean,ercstd,_=Atrophy_Plot(controltype=controltype, areatype="ERCTEC", Hemi=Hemi,outlier_subject=outlier_subject)
    hippomean,hippostd,_=Atrophy_Plot(controltype=controltype, areatype="Hippocampus", Hemi=Hemi,outlier_subject=outlier_subject)
    ADNIall_control_mean=np.array([amymean,ercmean,hippomean])
    ADNIall_control_std=np.array([amystd,ercstd,hippostd])
    #ADNIall_control_mean=np.array([1.62645996, 1.85790138, 0.75856693])
    #ADNIall_control_std=np.array([2.33882398, 2.99550011, 0.93565364])
    #control_age=75.32988620515611
    '''[1.69491038 1.84797416 0.82880703]
[2.88660485 3.27339797 0.91302707]
75.35547034093797'''
    print(ADNIall_control_mean)
    print(ADNIall_control_std)
    print(control_age)
    controltype="MCI"
    amymean,amystd,mci_age=Atrophy_Plot(controltype=controltype, areatype="Amygdala", Hemi=Hemi,outlier_subject=outlier_subject)
    ercmean,ercstd,_=Atrophy_Plot(controltype=controltype, areatype="ERCTEC", Hemi=Hemi,outlier_subject=outlier_subject)
    hippomean,hippostd,_=Atrophy_Plot(controltype=controltype, areatype="Hippocampus", Hemi=Hemi,outlier_subject=outlier_subject)
    ADNIall_mci_mean=np.array([amymean,ercmean,hippomean])
    ADNIall_mci_std=np.array([amystd,ercstd,hippostd])
    #ADNIall_mci_mean=np.array([3.28304105, 3.58189407, 1.41696319])
    #ADNIall_mci_std=np.array([4.20759294, 4.1936489,  1.47100033])
    #mci_age=73.80131208883863
    '''[3.26346854 3.53291121 1.65121555]
[4.34840915 4.5360637  1.77255352]
73.80131208883863'''
    print(ADNIall_mci_mean)
    print(ADNIall_mci_std)
    print(mci_age)

    controltype="AD"
    amymean,amystd,ad_age=Atrophy_Plot(controltype=controltype, areatype="Amygdala", Hemi=Hemi,outlier_subject=outlier_subject)
    ercmean,ercstd,_=Atrophy_Plot(controltype=controltype, areatype="ERCTEC", Hemi=Hemi,outlier_subject=outlier_subject)
    hippomean,hippostd,_=Atrophy_Plot(controltype=controltype, areatype="Hippocampus", Hemi=Hemi,outlier_subject=outlier_subject)
    ADNIall_ad_mean=np.array([amymean,ercmean,hippomean])
    ADNIall_ad_std=np.array([amystd,ercstd,hippostd])
    #ADNIall_ad_mean=np.array([7.31445332, 9.30266091, 2.65063582])
    #ADNIall_mci_std=np.array([5.59903502, 6.32313355, 1.61412047])
    #ad_age=75.08473941228307
    '''[7.18913829 8.39554379 3.22861176]
[5.60275243 6.20658187 2.12304738]
75.08473941228307'''
    print(ADNIall_ad_mean)
    print(ADNIall_ad_std)
    print(ad_age)
    ADNIall_age=[control_age,mci_age,ad_age]
    FS_TITLE  = 30
    FS_LABEL  = 25
    FS_TICK   = 20
    FS_LEGEND = 22

    # Colors
    con_colors = ["#FF9999", "#99CCFF", "#99FF99"]
    mci_colors = ["#FF6666", "#6699FF", "#66FF66"]
    ad_colors  = ["#CC0000", "#0000CC", "#009900"]
    age_colors_adni   = ["#CFCFCF", "#6F6F6F"]
    age_colors_ADNIall = ["#CFCFCF", "#8C8C8C", "#4A4A4A"]

    regions = ["Amygdala", "ERC/TEC", "Hippocampus"]
    bar_w   = 0.25

    fig, ax2 = plt.subplots(1, 1, figsize=(15, 10), sharey=True)

    x_control = 0.0
    x_mci     = 1.4
    offsets = np.array([-bar_w, 0.0, bar_w])
    # ---------------- Panel 2: ADNI-34 ----------------

    ax2.set_title("ADNI all", fontsize=FS_TITLE)#, fontweight="bold")

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
    ax2.set_ylim(-1,13)
    ax2.set_xticks([x_control2, x_mci2, x_ad2, x_age2])
    ax2.set_xticklabels(["Control", "MCI", "AD", "Age"], fontsize=FS_LABEL)#, fontweight="bold")
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
    plt.savefig("Figure/ADNI_all_VA.png", dpi=300, bbox_inches="tight")
    plt.close()
