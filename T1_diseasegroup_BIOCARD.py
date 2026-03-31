from logging import raiseExceptions
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime
control_data=pd.read_excel('/cis/home/yxie91/Datasets/mprage_inference/control_SUBJECT_DOB_multiple.xlsx',index_col=0,header=0)
mcidem_data=pd.read_excel('/cis/home/yxie91/Datasets/mprage_inference/MCIDEM_SUBJECT_DOB_multiple.xlsx',index_col=0,header=0)
TP_pd=pd.read_excel('/cis/home/yxie91/Datasets/mprage_inference/SUBJECT_TP_PET.xlsx',index_col=0,header=0)
ID_subjects=control_data.index
#ID_subjects=[s for s in ID_subjects if s not in ["MESDAV","SILKAT","WATKAT"]]
#control_data=control_data[ID_subjects]
subjects=[s for s in ID_subjects if control_data.loc[s,'DIAGNOSIS']=="NORMAL"]
DATA=control_data
print("NORMAL subjects:")
TP_len=[]
ages=[]
scan_range=[]
for i in range(len(subjects)):
    subject=subjects[i]
    TIMEPOINTS=TP_pd.loc[subject].dropna().values
    if len(TIMEPOINTS)<=2:
        print(f"Subject {subject} less than 3 time points")
    dob=DATA.loc[subject,'DOB']
    date_format = "%y%m%d"
    for tp in TIMEPOINTS:
        tp=str(int(tp))
        datetime1 = dob
        datetime2 = datetime.strptime(tp, date_format)
        age = datetime2.year - datetime1.year
        if (datetime2.month, datetime2.day) < (datetime1.month, datetime1.day):
            age -= 1
        ages.append(age)
        break
    scan_ages=[]
    for tp in TIMEPOINTS:
        tp=str(int(tp))
        datetime1 = dob
        datetime2 = datetime.strptime(tp, date_format)
        age = datetime2.year - datetime1.year
        if (datetime2.month, datetime2.day) < (datetime1.month, datetime1.day):
            age -= 1
        scan_ages.append(age)
    scan_range.append(scan_ages[-1]-scan_ages[0])
    TP_len.append(len(TIMEPOINTS))
print('Number of subjects',len(subjects))
print('Age mean',np.mean(ages))
print('Age STD',np.std(ages))
print('Scan mean',np.mean(TP_len))
print('Scan STD', np.std(TP_len))
print('Scan range mean',np.mean(scan_range))
print('Scan range STD', np.std(scan_range))
subjects=[s for s in ID_subjects if control_data.loc[s,'DIAGNOSIS']=="IMPAIRED NOT MCI"]
DATA=control_data
print("Impaired not MCI subjects:")
TP_len=[]
ages=[]
scan_range=[]
for i in range(len(subjects)):
    subject=subjects[i]
    TIMEPOINTS=TP_pd.loc[subject].dropna().values
    if len(TIMEPOINTS)<=2:
        print(f"Subject {subject} less than 3 time points")
    dob=DATA.loc[subject,'DOB']
    date_format = "%y%m%d"
    for tp in TIMEPOINTS:
        tp=str(int(tp))
        datetime1 = dob
        datetime2 = datetime.strptime(tp, date_format)
        age = datetime2.year - datetime1.year
        if (datetime2.month, datetime2.day) < (datetime1.month, datetime1.day):
            age -= 1
        ages.append(age)
        break
    scan_ages=[]
    for tp in TIMEPOINTS:
        tp=str(int(tp))
        datetime1 = dob
        datetime2 = datetime.strptime(tp, date_format)
        age = datetime2.year - datetime1.year
        if (datetime2.month, datetime2.day) < (datetime1.month, datetime1.day):
            age -= 1
        scan_ages.append(age)
    scan_range.append(scan_ages[-1]-scan_ages[0])
    TP_len.append(len(TIMEPOINTS))
print('Number of subjects',len(subjects))
print('Age mean',np.mean(ages))
print('Age STD',np.std(ages))
print('Scan mean',np.mean(TP_len))
print('Scan STD', np.std(TP_len))
print('Scan range mean',np.mean(scan_range))
print('Scan range STD', np.std(scan_range))

ID_subjects=mcidem_data.index
#ID_subjects=[s for s in ID_subjects if s not in ["CONLUC","HILCAR","SAUCON","WILJAY"]]
#mcidem_data=mcidem_data[ID_subjects]
subjects=[s for s in ID_subjects if mcidem_data.loc[s,'DIAGNOSIS']=="MCI"]
DATA=mcidem_data
print("MCI subjects:")
TP_len=[]
ages=[]
scan_range=[]
for i in range(len(subjects)):
    subject=subjects[i]
    TIMEPOINTS=TP_pd.loc[subject].dropna().values
    if len(TIMEPOINTS)<=2:
        print(f"Subject {subject} less than 3 time points")
    dob=DATA.loc[subject,'DOB']
    date_format = "%y%m%d"
    for tp in TIMEPOINTS:
        tp=str(int(tp))
        datetime1 = dob
        datetime2 = datetime.strptime(tp, date_format)
        age = datetime2.year - datetime1.year
        if (datetime2.month, datetime2.day) < (datetime1.month, datetime1.day):
            age -= 1
        ages.append(age)
        break
    scan_ages=[]
    for tp in TIMEPOINTS:
        tp=str(int(tp))
        datetime1 = dob
        datetime2 = datetime.strptime(tp, date_format)
        age = datetime2.year - datetime1.year
        if (datetime2.month, datetime2.day) < (datetime1.month, datetime1.day):
            age -= 1
        scan_ages.append(age)
    scan_range.append(scan_ages[-1]-scan_ages[0])
    TP_len.append(len(TIMEPOINTS))
print('Number of subjects',len(subjects))
print('Age mean',np.mean(ages))
print('Age STD',np.std(ages))
print('Scan mean',np.mean(TP_len))
print('Scan STD', np.std(TP_len))
print('Scan range mean',np.mean(scan_range))
print('Scan range STD', np.std(scan_range))

subjects=[s for s in ID_subjects if mcidem_data.loc[s,'DIAGNOSIS']=="DEMENTIA"]
DATA=mcidem_data
print("Dementia subjects:")
TP_len=[]
ages=[]
scan_range=[]
for i in range(len(subjects)):
    subject=subjects[i]
    TIMEPOINTS=TP_pd.loc[subject].dropna().values
    if len(TIMEPOINTS)<=2:
        print(f"Subject {subject} less than 3 time points")
    dob=DATA.loc[subject,'DOB']
    date_format = "%y%m%d"
    for tp in TIMEPOINTS:
        tp=str(int(tp))
        datetime1 = dob
        datetime2 = datetime.strptime(tp, date_format)
        age = datetime2.year - datetime1.year
        if (datetime2.month, datetime2.day) < (datetime1.month, datetime1.day):
            age -= 1
        ages.append(age)
        break
    scan_ages=[]
    for tp in TIMEPOINTS:
        tp=str(int(tp))
        datetime1 = dob
        datetime2 = datetime.strptime(tp, date_format)
        age = datetime2.year - datetime1.year
        if (datetime2.month, datetime2.day) < (datetime1.month, datetime1.day):
            age -= 1
        scan_ages.append(age)
    scan_range.append(scan_ages[-1]-scan_ages[0])
    TP_len.append(len(TIMEPOINTS))
print('Number of subjects',len(subjects))
print('Age mean',np.mean(ages))
print('Age STD',np.std(ages))
print('Scan mean',np.mean(TP_len))
print('Scan STD', np.std(TP_len))
print('Scan range mean',np.mean(scan_range))
print('Scan range STD', np.std(scan_range))