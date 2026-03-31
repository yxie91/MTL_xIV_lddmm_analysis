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
from collections import Counter

resolution = 1.2 * 1.0 * 1.0

'''def STD_volumes(folder, outlier_file="",tag=""):
    subjects_scandate=list(set(['__'.join(s.split('__')[:2]) for s in os.listdir(folder)]))
    seg_subjects=[s.split('__')[0] for s in subjects_scandate]
    #print(seg_subjects)
    subject_count = Counter(seg_subjects)
    SUBJECT = sorted([subj for subj, cnt in subject_count.items() if cnt >= 3])
    print(len(SUBJECT))
    if outlier_file!="":
        outlier_set=pd.read_excel(outlier_file,index_col=0).index.values
    else:
        outlier_set=[]
    print(outlier_set)
    V_amy=[];V_erc=[];V_hip=[]
    for file in sorted(os.listdir(folder)):
        if not file.endswith(".nii.gz"):
            continue
        #print(file[:-7])
        if not file.split('__')[0] in SUBJECT:
            continue
        if file[:-7] in outlier_set:
            #print(f"{file} is an outlier removed")
            continue
        seg_path = os.path.join(folder, file)
        seg = np.asarray(nib.load(seg_path).get_fdata()).squeeze()
        ones = np.ones_like(seg, dtype=np.uint8)

        # labels: 1 (amygdala), 2/3 (ERC/TEC), 4/5 (hippocampus)
        V_amy.append(resolution * ones[seg == 1].size)
        V_erc.append(resolution * (np.append(ones[seg == 2], ones[seg == 3]).size))
        V_hip.append(resolution * (np.append(ones[seg == 4], ones[seg == 5]).size))
    print(tag)
    print(np.std(V_amy),np.std(V_erc),np.std(V_hip))'''

def STD_volumes(folder, outlier_file="",tag=""):
    if outlier_file!="":
        outlier_set=pd.read_excel(outlier_file,index_col=0).index.values
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
    print(tag)
    print(np.std(V_amy),np.std(V_erc),np.std(V_hip))

STD_volumes("/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset006_axis0/imagesTs_pred_fuse",tag="BIOCARD 3T")
STD_volumes("/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset006_axis0/labelsTs_15mm_fuse",'/cis/home/yxie91/1520TBIOCARD/15Outlier.xlsx',tag="BIOCARD 1.5 mm")
STD_volumes("/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset006_axis0/labelsTs_20mm_fuse",'/cis/home/yxie91/1520TBIOCARD/20Outlier.xlsx',tag="BIOCARD 2.0 mm")

STD_volumes("/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset103_ADNIall/labelsTs_fuse",tag="ADNI 1/2/GO")
STD_volumes("/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset102_ADNI34/labelsTs_fuse",'/cis/home/yxie91/ADNI/ADNI34Outlier.xlsx',tag="ADNI 3/4")
STD_volumes("/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset006_axis0/imagesTs_pred_mprage",tag="BIOCARD 3T LH")
STD_volumes("/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset006_axis0/imagesTs_pred_RH",tag="BIOCARD 3T RH")
STD_volumes("/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset103_ADNIall/labelsTs",tag="ADNI 1/2/GO LH")
STD_volumes("/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset103_ADNIall/labelsTs_RH",tag="ADNI 1/2/GO RH")

#STD_volumes("/cis/project/adni_yxie91/ADNI_T1all/labelsTs_nnUNet",'/cis/project/adni_yxie91/ADNI_T1all/Datasheets/ADNIall_Outlier_updated.xlsx',tag="ADNI LH")
#STD_volumes("/cis/project/adni_yxie91/ADNI_T1all/labelsTs_nnUNet_RH",'/cis/project/adni_yxie91/ADNI_T1all/Datasheets/ADNIall_Outlier_updated.xlsx',tag="ADNI RH")
#STD_volumes("/cis/project/adni_yxie91/ADNI_T1all/labelsTs_nnUNet_fuse",'/cis/project/adni_yxie91/ADNI_T1all/Datasheets/ADNIall_Outlier_updated.xlsx',tag="ADNI fuse")

