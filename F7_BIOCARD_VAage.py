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


def tp_atrophy_cal(ages,volumes):
    ages = np.array(ages).reshape(-1, 1)
    volumes = np.array(volumes)
    model = LinearRegression()
    model.fit(ages, volumes)
    vol0=volumes[0]
    atrophy_rate=-model.coef_[0]/vol0*100
    return atrophy_rate
def Atrophy_Plot(seg_root,controltype, areatype):
    print('-------------------')
    print(controltype, areatype)
    print('-------------------')
    control_DATA=pd.read_excel('Data/Sheets/control_SUBJECT_DOB_multiple.xlsx',index_col=0,header=0)
    MCIDEM_DATA=pd.read_excel('Data/Sheets/MCIDEM_SUBJECT_DOB_multiple.xlsx',index_col=0,header=0)
    control_Subjects=control_DATA.index
    MCIDEM_Subjects=MCIDEM_DATA.index
    TP_pd=pd.read_excel('Data/Sheets/SUBJECT_TP_PET.xlsx',index_col=0,header=0)
    if controltype=='Control':
        SUBJECT=control_Subjects
        DATA=control_DATA
    elif controltype=='MCIDEM':
        SUBJECT=MCIDEM_Subjects
        DATA=MCIDEM_DATA
    else:
        raise TypeError('Controltype not recognized')
    
    VA=[]
    AGE_avg=[]
    for i in range(len(SUBJECT)):
        subject=SUBJECT[i]
        TIMEPOINTS=TP_pd.loc[subject].dropna().values
        if len(TIMEPOINTS)<=2:
            print(f"Subject {subject} less than 3 time points")
        dob=DATA.loc[subject,'DOB']
        date_format = "%y%m%d"
        ages=[]
        for tp in TIMEPOINTS:
            tp=str(int(tp))
            datetime1 = dob
            datetime2 = datetime.strptime(tp, date_format)
            age = datetime2.year - datetime1.year
            if (datetime2.month, datetime2.day) < (datetime1.month, datetime1.day):
                age -= 1
            ages.append(age)
        valid_ages=[]
        valid_volumes=[]
        volumes=[]
        for k in range(len(TIMEPOINTS)):
            timepoint=str(int(TIMEPOINTS[k]))
            seg_path= seg_root + f'/{subject}_{timepoint}.nii.gz'
            seg_img= np.array(nib.load(seg_path).get_fdata()).squeeze()
            mask = np.ones_like(seg_img)
            if areatype=='Amygdala':
                voxel=mask[seg_img==1]
                volume=1.2*1*1*voxel.shape[0]
                std=216.67356115587256
            elif areatype=='ERCTEC':
                voxel=np.append(mask[seg_img==2],mask[seg_img==3])
                volume=1.2*1*1*voxel.shape[0]
                std=446.50853548460185
            elif areatype=='Hippocampus':
                voxel=np.append(mask[seg_img==4],mask[seg_img==5])
                volume=1.2*1*1*voxel.shape[0]
                std=718.5271953107335
            else:
                raise TypeError("Areatype not recognized")
            volumes.append(volume)
        for k in range(len(TIMEPOINTS)):
            volume=volumes[k]
            if np.abs(volume-np.mean(volumes))< 2.5 * std:
                age=ages[k]
                valid_ages.append(age)
                valid_volumes.append(volume)
            else:
                print(f"{str(int(TIMEPOINTS[k]))} is removed for 2.5 sigma")
        if len(valid_ages)<3:
            #print(f"{subject} has less than 3 tps")
            continue
        atrophy_rate=tp_atrophy_cal(valid_ages,valid_volumes)
        AGE_avg.append(np.mean(valid_ages))
        VA.append(atrophy_rate)
    return np.mean(VA),np.std(VA), np.mean(AGE_avg)

seg_root="/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset005_BIOCARD91/labelsTs_mprage_fuse"#'Data/Dataset006_axis0/imagesTs_pred_fuse'
amymean,amystd,control_age=Atrophy_Plot(seg_root,controltype="Control",areatype="Amygdala")
ercmean,ercstd,_=Atrophy_Plot(seg_root,controltype="Control",areatype="ERCTEC")
hippomean,hippostd,_=Atrophy_Plot(seg_root,controltype="Control",areatype="Hippocampus")
control_mean=np.array([amymean,ercmean,hippomean])
control_std=np.array([amystd,ercstd,hippostd])
print(control_mean)
print(control_std)
print(control_age)
'''[1.04470515 2.63108641 0.02164953]
[0.94341036 1.20866284 0.48099681]
71.24395604395605'''
controltype="MCI/DAT"
amymean,amystd,mcidat_age=Atrophy_Plot(seg_root,controltype="MCIDEM",areatype="Amygdala")
ercmean,ercstd,_=Atrophy_Plot(seg_root,controltype="MCIDEM",areatype="ERCTEC")
hippomean,hippostd,_=Atrophy_Plot(seg_root,controltype="MCIDEM",areatype="Hippocampus")
mcidat_mean=np.array([amymean,ercmean,hippomean])
mcidat_std=np.array([amystd,ercstd,hippostd])
print(mcidat_mean)
print(mcidat_std)
print(mcidat_age)
'''[2.01153786 3.41857671 0.67207094]
[1.58709796 1.7159493  0.91980122]
73.35641025641026'''

'''control_mean=np.array([1.29336779, 2.94205936, 0.01848135])
control_age=71.36590909090908
mcidat_mean=np.array([2.23801894, 3.91390272, 0.66574796])
mcidat_age=73.35904761904763'''
