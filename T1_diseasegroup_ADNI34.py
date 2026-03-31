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
def main(controltype, areatype,Hemi="FUSE",outlier_subject=[]):
    print('-------------------')
    print(Hemi, controltype, areatype)
    print('-------------------')
    DATA=pd.read_csv('/cis/home/yxie91/ADNI/ADNI34_MPRAGE_metadata.csv',index_col=0)
    outlier_file=pd.read_excel('/cis/home/yxie91/ADNI/ADNI34Outlier.xlsx',index_col=0).index.values
    if Hemi=="LH":
        predicted_root="/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset102_ADNI34/labelsTs"
    elif Hemi=="RH":
        predicted_root="/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset102_ADNI34/labelsTs_RH"
    elif Hemi=="FUSE":
        predicted_root="/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset102_ADNI34/labelsTs_fuse"
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
    #print(len(SUBJECT))
    subject_idx=0
    BS_AGE=[]
    scan_range=[]
    TP_len=[]
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
        BS_AGE.append(baseline_age)
        valid_ages=[]
        volumes=[]
        
        for k in range(len(segfiles)):
            name=segfiles[k][:-7]
            if name in outlier_file: 
                print(f"{name} not valid segfile")
                continue
            tp=datetime.strptime(name.split('__')[1], '%Y%m%d').date()
            age=baseline_age+(tp-baseline_tp).days/365.25
            
            seg_path=f"{predicted_root}/{segfiles[k]}"
            seg_img= np.array(nib.load(seg_path).get_fdata()).squeeze()
            mask = np.ones_like(seg_img)
            if areatype=='Amygdala':
                voxel=mask[seg_img==1]
                volume=1.2*1*1*voxel.shape[0]
            elif areatype=='ERCTEC':
                voxel=np.append(mask[seg_img==2],mask[seg_img==3])
                volume=1.2*1*1*voxel.shape[0]
            elif areatype=='Hippocampus':
                voxel=np.append(mask[seg_img==4],mask[seg_img==5])
                volume=1.2*1*1*voxel.shape[0]
            else:
                raise TypeError("Areatype not recognized")
            
            valid_ages.append(age)
            volumes.append(volume)
        if len(set(valid_ages)) >= 3:
            ages=valid_ages
            volumes=volumes
        else:
            continue
        #print(subject)
        #print(ages,volumes)
        subject_idx+=1
        '''if len(list(set(ages)))!=len(ages) or max(ages)!=ages[-1] or min(ages)!=ages[0]:
            print(ages)
            raise ValueError("redundant time points")'''
        TP_len.append(len(list(set(ages))))
        scan_range.append(max(ages)-min(ages))
    print('====================================')
    print(controltype)
    print('Number of subjects',len(SUBJECT))
    print('Baseline Age mean',np.mean(BS_AGE))
    print('Baseline Age STD',np.std(BS_AGE))
    print('# of Scan mean',np.mean(TP_len))
    print('# of Scan STD', np.std(TP_len))
    print(np.sum(TP_len))
    print('Scan range mean',np.mean(scan_range))
    print('Scan range STD', np.std(scan_range))
    print('----------------------------------------')




Hemi="FUSE"
diseasetype=["CN","SMC","EMCI","MCI","LMCI","AD"]
for controltype in diseasetype:
    main(controltype,"ERCTEC")