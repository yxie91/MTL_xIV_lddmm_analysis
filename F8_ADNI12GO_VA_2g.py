import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from sklearn.linear_model import LinearRegression


def tp_atrophy_cal(ages,volumes):
    ages = np.array(ages).reshape(-1, 1)
    volumes = np.array(volumes)
    model = LinearRegression()
    model.fit(ages, volumes)
    vol0=volumes[0]
    atrophy_rate=-model.coef_[0]/vol0*100
    return atrophy_rate,model
def Atrophy_Plot(controltype, areatype,outlier_subject=[]):
    print('-------------------')
    print(controltype, areatype)
    print('-------------------')
    control_AGE=[s[0][0] for s in list(sio.loadmat('Data/Sheets/rawthk_v13_kms_con.mat')["x1_c"])]
    pre_AGE=[s[0][0] for s in list(sio.loadmat('Data/Sheets/rawthk_v13_kms_pre.mat')["x1_p"])]
    mci_AGE=[s[0][0] for s in list(sio.loadmat('Data/Sheets/rawthk_v13_kms_mci.mat')["x1_m"])]
    control_subjects=[s[0][0] for s in list(sio.loadmat('Data/Sheets/rawthk_v13_kms_con.mat')["SUBJSc"])]
    pre_subjects=[s[0][0] for s in list(sio.loadmat('Data/Sheets/rawthk_v13_kms_pre.mat')["SUBJSp"])]
    mci_subjects=[s[0][0] for s in list(sio.loadmat('Data/Sheets/rawthk_v13_kms_mci.mat')["SUBJSm"])]
    predicted_root="Data/Dataset103_ADNIall/labelsTs_fuse"

    if controltype=='Control':
        SUBJECT=control_subjects+pre_subjects
        AGE=control_AGE+pre_AGE
    elif controltype=='MCI/DAT':
        SUBJECT=mci_subjects
        SUBJECT=[s for s in SUBJECT if s not in ['1268', '1117']]#MCI subject, only want LMCI subjects
        SUBJECT=[s for s in SUBJECT if s not in ['4035','4668']]#TP, segfiles not corresponding
        AGE=mci_AGE
    else:
        raise TypeError('Controltype not recognized')

    subject_idx=0
    tp_idx=0
    outdict={}
    VA=[]
    AGE_avg=[]
    plt.figure(figsize=(10, 10))
    for i in range(len(SUBJECT)):
        subject=SUBJECT[i]
        ages=AGE[i]
        #print(subject,ages)
        if subject in outlier_subject:
            print(f"{subject} is skipped due to outlier")
            continue
        volumes=[]
        segfiles=sorted([file for file in os.listdir(predicted_root) if subject in file], key=lambda x: float(x.split('_')[1].split('.')[0]))
        segfiles=sorted([file for file in segfiles if file not in ["4888_18.nii.gz"]])
        common_len=min(len(segfiles),len(ages))
        segfiles=segfiles[:common_len]
        ages=ages[:common_len]
        valid_ages=[]
        valid_volumes=[]
        for k in range(len(segfiles)):
            seg_path=f"{predicted_root}/{segfiles[k]}"
            seg_img= np.array(nib.load(seg_path).get_fdata()).squeeze()
            mask = np.ones_like(seg_img)
            if areatype=='Amygdala':
                voxel=mask[seg_img==1]
                volume=1.2*1*1*voxel.shape[0]
                std=464.19775089577183
            elif areatype=='ERC':
                voxel=np.append(mask[seg_img==2],mask[seg_img==3])
                volume=1.2*1*1*voxel.shape[0]
                std=655.2641658499751
            elif areatype=='Hippocampus':
                voxel=np.append(mask[seg_img==4],mask[seg_img==5])
                volume=1.2*1*1*voxel.shape[0]
                std=860.5986319404608
            else:
                raise TypeError("Areatype not recognized")
            volumes.append(volume)
        for k in range(len(ages)):
            volume=volumes[k]
            if np.abs(volume-np.mean(volumes))< 2.5 * std:
                age=ages[k]
                valid_ages.append(age)
                valid_volumes.append(volume)
            else:
                print(f"{segfiles[k]} is removed for 2.5 sigma")
        if len(valid_ages)<3:
            print(f"{subject} has less than three tps")
            continue
        subject_idx+=1
        tp_idx+=len(valid_ages)
        atrophy_rate,_=tp_atrophy_cal(valid_ages,valid_volumes)
        VA.append(atrophy_rate)
        AGE_avg.append(np.mean(valid_ages))
        out=[atrophy_rate]+valid_volumes
        outdict[subject]=out
    print(f"All has {subject_idx} subjects and {tp_idx} timepoints in total, {tp_idx/subject_idx:.3f} tps per subject")
    output=pd.DataFrame(outdict.values(),index=outdict.keys())
    num_columns = output.shape[1]
    column_names=['atrophy rate'] + [f'tp{i}' for i in range(1, num_columns)]
    output.columns=column_names
    if controltype=="Control":
        output.to_csv(f'Results/ADNI12GO_VA_Control_{areatype}.csv',header=True)
    elif controltype=="MCI/DAT":
        output.to_csv(f'Results/ADNI12GO_VA_MCIDAT_{areatype}.csv',header=True)
    else:
        raise TypeError('Control type not recognized')
    return np.mean(VA), np.std(VA), np.mean(AGE_avg)

outlier_subject=[]

Hemi="FUSE"
manual=False


amymean,amystd,con_age=Atrophy_Plot(controltype="Control", areatype="Amygdala",outlier_subject=outlier_subject)
ercmean,ercstd,_=Atrophy_Plot(controltype="Control", areatype="ERC",outlier_subject=outlier_subject)
hippomean,hippostd,_=Atrophy_Plot(controltype="Control", areatype="Hippocampus",outlier_subject=outlier_subject)
con_mean=np.array([amymean,ercmean,hippomean])
con_std=np.array([amystd,ercstd,hippostd])
print(con_mean)
print(con_std)
print(con_age)

amymean,amystd,mci_age=Atrophy_Plot(controltype="MCI/DAT", areatype="Amygdala",outlier_subject=outlier_subject)
ercmean,ercstd,_=Atrophy_Plot(controltype="MCI/DAT", areatype="ERC",outlier_subject=outlier_subject)
hippomean,hippostd,_=Atrophy_Plot(controltype="MCI/DAT", areatype="Hippocampus",outlier_subject=outlier_subject)
mci_mean=np.array([amymean,ercmean,hippomean])
mci_std=np.array([amystd,ercstd,hippostd])
print(mci_mean)
print(mci_std)
print(mci_age)