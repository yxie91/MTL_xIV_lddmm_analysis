import scipy.io as sio
from ast import Str, Sub
from cmath import tau
from math import nan
from pickletools import uint8
import nibabel as nib
import os
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from datetime import datetime
from matplotlib.patches import Patch
from statsmodels import robust
from matplotlib.legend_handler import HandlerTuple
from collections import defaultdict

def main(controltype, areatype,Hemi="FUSE",outlier_subject=[]):
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


    if controltype=="CN":
        SUBJECT=sorted(list(set(list(DATA[DATA['Group']=='CN'].index.values))))
    elif controltype=="SMC":
        SUBJECT=sorted(list(set(list(DATA[DATA['Group']=='SMC'].index.values))))
    elif controltype=="EMCI":
        SUBJECT=sorted(list(set(list(DATA[DATA['Group']=='EMCI'].index.values))))
    elif controltype=='MCI':
        SUBJECT=sorted(list(set(list(DATA[DATA['Group']=='MCI'].index.values))))
    elif controltype=="LMCI":
        SUBJECT=sorted(list(set(list(DATA[DATA['Group']=='LMCI'].index.values))))
    elif controltype=='AD':
        SUBJECT=sorted(list(set(list(DATA[DATA['Group']=='AD'].index.values))))
    else:
        raise TypeError('Control type not recognized')
    #print(SUBJECT)
    #print(len(SUBJECT))
    subject_idx=0
    BS_AGE=[]
    scan_range=[]
    TP_len=[]
    std_amy,std_ERCTEC,std_hippo=491.2706445585293,634.1674837765971,933.8100679831414
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
        BS_AGE.append(baseline_age)
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
                
        '''if len(set(valid_ages)) < 3:
            continue'''
        
        age_dict = defaultdict(list)
        for age, vol in zip(valid_ages, valid_volumes):
            age_dict[age].append(vol)
        group_valid_ages = []
        group_valid_volumes = []
        for age in sorted(age_dict.keys()):
            group_valid_ages.append(age)
            group_valid_volumes.append(sum(age_dict[age]) / len(age_dict[age]))  
        #print(group_valid_ages,group_valid_volumes)  
        if len(group_valid_ages) >= 3:
            ages=group_valid_ages
            volumes=group_valid_volumes
        else:
            continue
        subject_idx+=1
        #print(ages,volumes)
        TP_len.append(len(list(set(ages))))
        scan_range.append(max(ages)-min(ages))
    print('====================================')
    print(controltype)
    print('Number of subjects',len(SUBJECT))
    print('Baseline Age mean',np.mean(BS_AGE))
    print('Baseline Age STD',np.std(BS_AGE))
    print('# of Scan mean',np.mean(TP_len))
    print('# of Scan STD', np.std(TP_len))
    print('Number of tps', np.sum(TP_len))
    print('Scan range mean',np.mean(scan_range))
    print('Scan range STD', np.std(scan_range))
    print('----------------------------------------')
    return len(SUBJECT),np.sum(TP_len) 



Hemi="FUSE"
diseasetype=["CN","SMC","EMCI","MCI","LMCI","AD"]
Num_subject=0
Num_TP=0
for controltype in diseasetype:
    ns, ntp = main(controltype,"ERCTEC")
    Num_subject+=ns
    Num_TP+=ntp
print(Num_subject,Num_TP)
#3218 5030