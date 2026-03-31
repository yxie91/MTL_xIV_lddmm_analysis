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
from collections import defaultdict
import scipy.io as sio
import numpy as np

def load_subject_age_pairs(mat_path, subj_key, age_key):
    """Load (subject, age) pairs from a single .mat, verifying 1:1 index alignment."""
    M = sio.loadmat(mat_path)
    subjects = [s[0][0] for s in list(M[subj_key])]
    ages = [a[0][0] for a in list(M[age_key])]
    print(subjects)
    print(ages)
    assert len(subjects) == len(ages), f"Length mismatch in {mat_path}: {len(subjects)} subjects vs {len(ages)} ages"
    return list(zip(subjects, ages))

def merge_pairs_to_big_dict(*pair_lists):
    """Merge multiple (subject, age) lists into subject -> list[age] dict."""
    big = defaultdict(list)
    for pairs in pair_lists:
        for subj, age in pairs:
            big[subj]=age
    return dict(big)

# --- Load each cohort by secure index pairing inside its own file ---
con_pairs = load_subject_age_pairs(
    '/cis/home/yxie91/ADNI/rawthk_v13_kms_con.mat',
    subj_key='SUBJSc', age_key='x1_c'
)
pre_pairs = load_subject_age_pairs(
    '/cis/home/yxie91/ADNI/rawthk_v13_kms_pre.mat',
    subj_key='SUBJSp', age_key='x1_p'
)
mci_pairs = load_subject_age_pairs(
    '/cis/home/yxie91/ADNI/rawthk_v13_kms_mci.mat',
    subj_key='SUBJSm', age_key='x1_m'
)
big_dict = merge_pairs_to_big_dict(con_pairs, pre_pairs, mci_pairs)
#print(big_dict)
control_subjects=[s[0][0] for s in list(sio.loadmat('/cis/home/yxie91/ADNI/rawthk_v13_kms_con.mat')["SUBJSc"])]
pre_subjects=[s[0][0] for s in list(sio.loadmat('/cis/home/yxie91/ADNI/rawthk_v13_kms_pre.mat')["SUBJSp"])]
mci_subjects=[s[0][0] for s in list(sio.loadmat('/cis/home/yxie91/ADNI/rawthk_v13_kms_mci.mat')["SUBJSm"])]

print(*control_subjects,sep=",")
print(*pre_subjects,sep=",")
print(*mci_subjects,sep=",")
DATA=pd.read_csv("/cis/home/yxie91/ADNI/T1w_MRI_Cohort__67_Subjects_Study_Entry_05Oct2025.csv",index_col=0)
print(len(control_subjects+pre_subjects+mci_subjects)==len(DATA.index.values))
cn_sub=[]
smc_sub=[]
lmci_sub=[]
mci_sub=[]
for subject in DATA.index.values:
    sub=subject.split('_')[2]
    subject_dg=DATA.loc[subject,"entry_research_group"]
    if subject_dg=="CN":
        cn_sub.append(subject)
    if subject_dg=="SMC":
        smc_sub.append(subject)
    if subject_dg=="LMCI":
        lmci_sub.append(subject)
    if subject_dg=="MCI":
        mci_sub.append(subject)
'''print(*cn_sub,sep=",")
print(*smc_sub,sep=",")
print(*lmci_sub,sep=",")
print(*mci_sub,sep=",")'''
CN_subjects=[f.split('_')[2] for f in cn_sub]
SMC_subjects=[f.split('_')[2] for f in smc_sub]
LMCI_subjects=[f.split('_')[2] for f in lmci_sub]
MCI_subjects=[f.split('_')[2] for f in mci_sub]
'''print(list(set(mci_subjects) ^ set(MCI_subjects)))'''
print(len(cn_sub),len(smc_sub),len(lmci_sub),len(mci_sub))

print("CN subjects:")
SUBJECT=CN_subjects
TP_len=[]
scan_range=[]
baseline_age=[]
for subject in SUBJECT:
    ages=big_dict[subject]
    #print(ages)
    baseline_age.append(ages[0])
    scan_range.append(ages[-1]-ages[0])
    TP_len.append(len(ages))
print('Number of subjects',len(SUBJECT))
print('Baseline Age mean',np.mean(baseline_age))
print('Baseline Age STD',np.std(baseline_age))
print('# of Scan mean',np.mean(TP_len))
print('# of Scan STD', np.std(TP_len))
print(np.sum(TP_len))
print('Scan range mean',np.mean(scan_range))
print('Scan range STD', np.std(scan_range))
print('----------------------------------------')

print("SMC subjects:")
SUBJECT=SMC_subjects
TP_len=[]
ages=[]
scan_range=[]
baseline_age=[]

for subject in SUBJECT:
    ages=big_dict[subject]
    baseline_age.append(ages[0])
    scan_range.append(ages[-1]-ages[0])
    TP_len.append(len(ages))
print('Number of subjects',len(SUBJECT))
print('Baseline Age mean',np.mean(baseline_age))
print('Baseline Age STD',np.std(baseline_age))
print('# of Scan mean',np.mean(TP_len))
print('# of Scan STD', np.std(TP_len))
print(np.sum(TP_len))
print('Scan range mean',np.mean(scan_range))
print('Scan range STD', np.std(scan_range))
print('----------------------------------------')

print("MCI subjects:")
SUBJECT=MCI_subjects
TP_len=[]
baseline_age=[]
scan_range=[]
for subject in SUBJECT:
    ages=big_dict[subject]
    baseline_age.append(ages[0])
    scan_range.append(ages[-1]-ages[0])
    TP_len.append(len(ages))
print(SUBJECT)
print('Number of subjects',len(SUBJECT))
print('Baseline Age mean',np.mean(baseline_age))
print('Baseline Age STD',np.std(baseline_age))
print('# of Scan mean',np.mean(TP_len))
print('# of Scan STD', np.std(TP_len))
print(np.sum(TP_len))
print('Scan range mean',np.mean(scan_range))
print('Scan range STD', np.std(scan_range))
print('----------------------------------------')

print("LMCI subjects:")
SUBJECT=LMCI_subjects
TP_len=[]
baseline_age=[]
scan_range=[]
print(SUBJECT)
for subject in SUBJECT:
    ages=big_dict[subject]
    baseline_age.append(ages[0])
    scan_range.append(ages[-1]-ages[0])
    TP_len.append(len(ages))
print('Number of subjects',len(SUBJECT))
print('Baseline Age mean',np.mean(baseline_age))
print('Baseline Age STD',np.std(baseline_age))
print('# of Scan mean',np.mean(TP_len))
print('# of Scan STD', np.std(TP_len))
print(np.sum(TP_len))
print('Scan range mean',np.mean(scan_range))
print('Scan range STD', np.std(scan_range))
print('----------------------------------------')



print("LMCI+ MCI subjects:")
SUBJECT=LMCI_subjects+MCI_subjects
TP_len=[]
baseline_age=[]
scan_range=[]
for subject in SUBJECT:
    ages=big_dict[subject]
    baseline_age.append(ages[0])
    scan_range.append(ages[-1]-ages[0])
    TP_len.append(len(ages))
print('Number of subjects',len(SUBJECT))
print('Baseline Age mean',np.mean(baseline_age))
print('Baseline Age STD',np.std(baseline_age))
print('# of Scan mean',np.mean(TP_len))
print('# of Scan STD', np.std(TP_len))
print(np.sum(TP_len))
print('Scan range mean',np.mean(scan_range))
print('Scan range STD', np.std(scan_range))
print('----------------------------------------')