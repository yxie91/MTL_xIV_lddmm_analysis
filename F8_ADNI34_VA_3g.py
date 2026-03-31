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
    return atrophy_rate,model
def Atrophy_Plot(controltype, areatype,Hemi="LH",outlier_subject=[]):
    print('-------------------')
    print(Hemi, controltype, areatype)
    print('-------------------')
    DATA=pd.read_csv('Data/Sheets/ADNI34_MPRAGE_metadata.csv',index_col=0)
    #print(*list(DATA.index.values),sep=",")
    outlier_file=pd.read_excel('Data/Sheets/ADNI34Outlier.xlsx',index_col=0).index.values
    if Hemi=="LH":
        predicted_root="Data/Dataset102_ADNI34/labelsTs"
    elif Hemi=="RH":
        predicted_root="Data/nnUNet_raw/Dataset102_ADNI34/labelsTs_RH"
    elif Hemi=="FUSE":
        predicted_root="Data/Dataset102_ADNI34/labelsTs_fuse"
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
    plt.figure(figsize=(10, 10))
    for i in range(len(SUBJECT)):
        subject=SUBJECT[i]
        segfiles=sorted([file for file in os.listdir(predicted_root) if subject in file])
        if segfiles == [] or subject in outlier_subject:
            continue
        subject_rows = DATA.loc[subject]
        date2age = dict(zip(subject_rows['Acq Date'], subject_rows['Age']))
        date2age = {datetime.strptime(k, '%m/%d/%Y').date(): v for k, v in date2age.items()}
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
            seg_img= np.array(nib.load(seg_path).get_fdata()).squeeze()
            mask = np.ones_like(seg_img)
            if areatype=='Amygdala':
                voxel=mask[seg_img==1]
                volume=1.2*1*1*voxel.shape[0]
                std=439.92524946082494
            elif areatype=='ERCTEC':
                voxel=np.append(mask[seg_img==2],mask[seg_img==3])
                volume=1.2*1*1*voxel.shape[0]
                std=614.3123051905417
            elif areatype=='Hippocampus':
                voxel=np.append(mask[seg_img==4],mask[seg_img==5])
                volume=1.2*1*1*voxel.shape[0]
                std=844.2199459018403
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
        subject_idx+=1
        tp_idx+=len(set(valid_ages))
        atrophy_rate,model=tp_atrophy_cal(valid_ages,valid_volumes)
        #print(subject,valid_ages,valid_volumes,atrophy_rate)
        VA.append(atrophy_rate)
        AGE_avg.append(np.mean(valid_ages))
        out=[atrophy_rate]+valid_volumes
        outdict[subject]=out
    #print(f"All has {subject_idx} subjects and {tp_idx} timepoints in total, {tp_idx/subject_idx:.3f} tps per subject")
    output=pd.DataFrame(outdict.values(),index=outdict.keys())
    num_columns = output.shape[1]
    column_names=['atrophy rate']+[f'tp{i}' for i in range(1,num_columns)]
    output.columns=column_names
    print(np.min(VA),np.max(VA))
    return np.mean(VA), np.std(VA), np.mean(AGE_avg)
if __name__ == "__main__":
    outlier_subject=[]
    Hemi="FUSE"
    controltype="Control"
    amymean,amystd,control_age=Atrophy_Plot(controltype=controltype, areatype="Amygdala", Hemi=Hemi,outlier_subject=outlier_subject)
    ercmean,ercstd,_=Atrophy_Plot(controltype=controltype, areatype="ERCTEC", Hemi=Hemi,outlier_subject=outlier_subject)
    hippomean,hippostd,_=Atrophy_Plot(controltype=controltype, areatype="Hippocampus", Hemi=Hemi,outlier_subject=outlier_subject)
    FUSE_control_mean=np.array([amymean,ercmean,hippomean])
    FUSE_control_std=np.array([amystd,ercstd,hippostd])
    #FUSE_control_mean=np.array([1.10804747, 1.4896111,  0.74579776])
    #control_age=75.5498482572935
    print(FUSE_control_mean)
    print(FUSE_control_std)
    print(control_age)
    controltype="MCI"
    amymean,amystd,mci_age=Atrophy_Plot(controltype=controltype, areatype="Amygdala", Hemi=Hemi,outlier_subject=outlier_subject)
    ercmean,ercstd,_=Atrophy_Plot(controltype=controltype, areatype="ERCTEC", Hemi=Hemi,outlier_subject=outlier_subject)
    hippomean,hippostd,_=Atrophy_Plot(controltype=controltype, areatype="Hippocampus", Hemi=Hemi,outlier_subject=outlier_subject)
    FUSE_mci_mean=np.array([amymean,ercmean,hippomean])
    FUSE_mci_std=np.array([amystd,ercstd,hippostd])
    #FUSE_mci_mean=[2.75852225, 3.17414196, 1.3279492 ]
    #mci_age=75.77016007019532
    print(FUSE_mci_mean)
    print(FUSE_mci_std)
    print(mci_age)
    controltype="AD"
    amymean,amystd,ad_age=Atrophy_Plot(controltype=controltype, areatype="Amygdala", Hemi=Hemi,outlier_subject=outlier_subject)
    ercmean,ercstd,_=Atrophy_Plot(controltype=controltype, areatype="ERCTEC", Hemi=Hemi,outlier_subject=outlier_subject)
    hippomean,hippostd,_=Atrophy_Plot(controltype=controltype, areatype="Hippocampus", Hemi=Hemi,outlier_subject=outlier_subject)
    FUSE_ad_mean=np.array([amymean,ercmean,hippomean])
    FUSE_ad_std=np.array([amystd,ercstd,hippostd])
    #FUSE_ad_mean=[6.70709279, 7.63305388, 2.75762269]
    #ad_age=76.50101148376302
    print(FUSE_ad_mean)
    print(FUSE_ad_std)
    print(ad_age)
