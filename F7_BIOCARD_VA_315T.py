from cmath import tau
from math import nan
import nibabel as nib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime
from statsmodels import robust
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
def tp_atrophy_cal(ages,volumes):
    ages = np.array(ages).reshape(-1, 1)
    volumes = np.array(volumes)
    model = LinearRegression()
    model.fit(ages, volumes)
    vol0=volumes[0]
    atrophy_rate=-model.coef_[0]/vol0*100
    return atrophy_rate,model
def Atrophy_Plot(controltype, areatype):
    outdict12={}
    seg_root='/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset005_BIOCARD91/labelsTs_mprage_fuse'#'Data/Dataset006_axis0/imagesTs_pred_fuse'
    control_DATA=pd.read_excel('Data/Sheets/control_SUBJECT_DOB_multiple.xlsx',index_col=0,header=0)
    MCIDEM_DATA=pd.read_excel('Data/Sheets/MCIDEM_SUBJECT_DOB_multiple.xlsx',index_col=0,header=0)
    control_Subjects=control_DATA.index
    MCIDEM_Subjects=MCIDEM_DATA.index
    TP_pd=pd.read_excel('Data/Sheets/SUBJECT_TP_PET.xlsx',index_col=0,header=0)
    if controltype=='Control':
        SUBJECT=control_Subjects
        DATA=control_DATA
        #SUBJECT=[s for s in SUBJECT if s not in ["MESDAV","SILKAT","WATKAT"]]
    elif controltype=="MCI/DAT":
        SUBJECT=MCIDEM_Subjects
        DATA=MCIDEM_DATA
        #SUBJECT=[s for s in SUBJECT if s not in ["CONLUC","HILCAR","SAUCON","WILJAY"]]
    else:
        raise TypeError('Controltype not recognized')
    
    VA_12=[]
    AGE_avg=[]
    subject_idx=0
    tp_idx=0
    plt.figure(figsize=(10, 10))
    for i in range(len(SUBJECT)):
        subject=SUBJECT[i]
        subject_idx+=1
        TIMEPOINTS=TP_pd.loc[subject].dropna().values
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

        atrophy_rate,model=tp_atrophy_cal(valid_ages,valid_volumes)
        #print(subject,valid_ages,valid_volumes,atrophy_rate)
        AGE_avg.append(np.mean(valid_ages))
        VA_12.append(atrophy_rate)
        out=[atrophy_rate]+valid_volumes
        outdict12[subject]=out
    output=pd.DataFrame(outdict12.values(),index=outdict12.keys())
    num_columns = output.shape[1]
    column_names=['atrophy rate']+[f'tp{i}' for i in range(1,num_columns)]
    output.columns=column_names
    print("1.2mm average age",np.mean(AGE_avg))
    '''if controltype=="Control":
        output.to_csv(f'Results/12Fuse_Control_{areatype}.csv',header=True)
    elif controltype=="MCI/DAT":
        output.to_csv(f'Results/12Fuse_MCIDAT_{areatype}.csv',header=True)
    else:
        raise TypeError('Control type not recognized')'''
    #print('-------------------------------------')
    
    
    
    AGE_avg=[] 
    outlier_file=pd.read_excel('Data/Sheets/15Outlier.xlsx',index_col=0).index.values
    outdict15={}
    DATA=pd.read_csv('Data/Sheets/Demographic_15T.csv',index_col=0)
    acq_dates=pd.read_csv('Data/Sheets/set2a_acquisition_dates.csv',index_col=0)
    predicted_root="Data/Dataset006_axis0/labelsTs_15mm_fuse"
    if controltype=="Control":
        SUBJECT=DATA[DATA['DIAGNOSIS'].isin(['NORMAL', 'IMPAIRED NOT MCI'])].index.values
    elif controltype=="MCI/DAT":
        SUBJECT=DATA[DATA['DIAGNOSIS'].isin(['MCI', 'DEMENTIA'])].index.values
    else:
        raise TypeError('Control type not recognized')
    subject_idx=0
    tp_idx=0
    VA_15=[]
    outlier_subject=[]
    for i in range(len(SUBJECT)):
        subject=SUBJECT[i]
        segfiles=sorted([file for file in os.listdir(predicted_root) if subject in file])
        if subject in outlier_subject:
            continue
        segfile_names = [f.split('.')[0] for f in segfiles]
        TIMEPOINTS = []
        valid_segfiles = []

        for name, fullfile in zip(segfile_names, segfiles):
            if name in outlier_file: 
                #print(f"{fullfile} not valid segfile")
                continue
            try:
                tp = acq_dates.loc[name, " acquisition_date"]
                if tp==' ':
                    continue
                TIMEPOINTS.append(tp)
                valid_segfiles.append(fullfile)
            except KeyError:
                continue
        segfiles=valid_segfiles
        if len(segfiles)<=2:
            continue
        dob=DATA.loc[subject,'DOB']
        ages=[]
        for tp in TIMEPOINTS:
            datetime1 = datetime.strptime(dob, "%Y-%m-%d")
            datetime2 = datetime.strptime(tp.strip(), "%Y%m%d")
            age_days = (datetime2 - datetime1).days 
            age = age_days / 365.25
            ages.append(age)
        valid_ages=[]
        valid_volumes=[]
        volumes=[]
        for k in range(len(segfiles)):
            seg_path=f"{predicted_root}/{segfiles[k]}"
            seg_img= np.array(nib.load(seg_path).get_fdata()).squeeze()
            mask = np.ones_like(seg_img)
            if areatype=='Amygdala':
                voxel=mask[seg_img==1]
                volume=1.2*1*1*voxel.shape[0]
                std = 261.69225722876206
            elif areatype=='ERCTEC':
                voxel=np.append(mask[seg_img==2],mask[seg_img==3])
                volume=1.2*1*1*voxel.shape[0]
                std = 405.6347459701783
            elif areatype=='Hippocampus':
                voxel=np.append(mask[seg_img==4],mask[seg_img==5])
                volume=1.2*1*1*voxel.shape[0]
                std = 722.9688489063983
            else:
                raise TypeError("Areatype not recognized")
            volumes.append(volume)
        for k in range(len(segfiles)):
            volume=volumes[k]
            if np.abs(volume-np.mean(volumes))< 2.5 * std:
                age=ages[k]
                valid_ages.append(age)
                valid_volumes.append(volume)
            else:
                print(f"{segfiles[k]} is removed for 2.5 sigma")
        if len(valid_ages)<3:
            #print(f"1.5mm {subject} has less than 3 tps")
            continue
        atrophy_rate,model=tp_atrophy_cal(valid_ages,valid_volumes)
        #print(subject,valid_ages,valid_volumes,atrophy_rate)
        VA_15.append(atrophy_rate)
        AGE_avg.append(np.mean(valid_ages))
        out=[atrophy_rate]+valid_volumes
        outdict15[subject]=out
    output=pd.DataFrame(outdict15.values(),index=outdict15.keys())
    num_columns = output.shape[1]
    column_names=['atrophy rate']+[f'tp{i}' for i in range(1,num_columns)]
    output.columns=column_names
    '''if controltype=="Control":
        output.to_csv(f'Results/15Fuse_Control_{areatype}.csv',header=True)
    elif controltype=="MCI/DAT":
        output.to_csv(f'Results/15Fuse_MCIDAT_{areatype}.csv',header=True)
    else:
        raise TypeError('Control type not recognized')'''
    #print('-------------------------------------')
    print("1.5mm average age",np.mean(AGE_avg))



    AGE_avg=[] 
    outlier_file=pd.read_excel('Data/Sheets/20Outlier.xlsx',index_col=0).index.values
    outdict20={}
    DATA=pd.read_csv('Data/Sheets/Demographic_20T.csv',index_col=0)
    acq_dates=pd.read_csv('Data/Sheets/set2b_acquisition_dates.csv',index_col=0)
    predicted_root="Data/Dataset006_axis0/labelsTs_20mm_fuse"
    if controltype=="Control":
        SUBJECT=DATA[DATA['DIAGNOSIS'].isin(['NORMAL', 'IMPAIRED NOT MCI'])].index.values
    elif controltype=="MCI/DAT":
        SUBJECT=DATA[DATA['DIAGNOSIS'].isin(['MCI', 'DEMENTIA'])].index.values
    else:
        raise TypeError('Control type not recognized')
    subject_idx=0
    tp_idx=0
    VA_20=[]
    outlier_subject=[]
    for i in range(len(SUBJECT)):
        subject=SUBJECT[i]
        segfiles=sorted([file for file in os.listdir(predicted_root) if subject in file])
        if subject in outlier_subject:
            continue
        segfile_names = [f.split('.')[0] for f in segfiles]
        TIMEPOINTS = []
        valid_segfiles = []

        for name, fullfile in zip(segfile_names, segfiles):
            if name in outlier_file: 
                #print(f"{fullfile} not valid segfile")
                continue
            try:
                tp = acq_dates.loc[name, " acquisition_date"]
                if tp==' ':
                    continue
                TIMEPOINTS.append(tp)
                valid_segfiles.append(fullfile)
            except KeyError:
                continue
        segfiles=valid_segfiles
        if len(segfiles)<=2:
            continue
        dob=DATA.loc[subject,'DOB']
        ages=[]
        for tp in TIMEPOINTS:
            datetime1 = datetime.strptime(dob, "%Y-%m-%d")
            datetime2 = datetime.strptime(tp.strip(), "%Y%m%d")
            age_days = (datetime2 - datetime1).days 
            age = age_days / 365.25
            ages.append(age)
        valid_ages=[]
        valid_volumes=[]
        volumes=[]
        for k in range(len(segfiles)):
            tp_idx+=1
            seg_path=f"{predicted_root}/{segfiles[k]}"
            seg_img= np.array(nib.load(seg_path).get_fdata()).squeeze()
            mask = np.ones_like(seg_img)
            if areatype=='Amygdala':
                voxel=mask[seg_img==1]
                volume=1.2*1*1*voxel.shape[0]
                std=229.99065491545463
            elif areatype=='ERCTEC':
                voxel=np.append(mask[seg_img==2],mask[seg_img==3])
                volume=1.2*1*1*voxel.shape[0]
                std=423.8013266088286
            elif areatype=='Hippocampus':
                voxel=np.append(mask[seg_img==4],mask[seg_img==5])
                volume=1.2*1*1*voxel.shape[0]
                std=672.3162684812407
            else:
                raise TypeError("Areatype not recognized")
            volumes.append(volume)
        for k in range(len(segfiles)):
            volume=volumes[k]
            if np.abs(volume-np.mean(volumes))< 2.5 * std:
                age=ages[k]
                valid_ages.append(age)
                valid_volumes.append(volume)
            else:
                print(f"{segfiles[k]} is removed for 2.5 sigma")
        if len(valid_ages)<3:
            print(f"2.0 mm {subject} has less than 3 tps")
            continue

        atrophy_rate,model=tp_atrophy_cal(valid_ages,valid_volumes)
        #print(subject,valid_ages,valid_volumes,atrophy_rate)
        VA_20.append(atrophy_rate)
        AGE_avg.append(np.mean(valid_ages))
        out=[atrophy_rate]+valid_volumes
        outdict20[subject]=out
    print("2.0mm average age",np.mean(AGE_avg))
    output=pd.DataFrame(outdict20.values(),index=outdict20.keys())
    num_columns = output.shape[1]
    column_names=['atrophy rate']+[f'tp{i}' for i in range(1,num_columns)]
    output.columns=column_names
    '''if controltype=="Control":
        output.to_csv(f'Results/20Fuse_Control_{areatype}.csv',header=True)
    elif controltype=="MCI/DAT":
        output.to_csv(f'Results/20Fuse_MCIDAT_{areatype}.csv',header=True)
    else:
        raise TypeError('Control type not recognized')'''
    #print('-------------------------------------')
    return [np.mean(VA_12),np.mean(VA_15),np.mean(VA_20)], [np.std(VA_12),np.std(VA_15),np.std(VA_20)]


amymean, amystd=Atrophy_Plot("MCI/DAT","Amygdala")
ercmean, ercstd=Atrophy_Plot("MCI/DAT","ERCTEC")
hippomean, hippostd=Atrophy_Plot("MCI/DAT","Hippocampus")

mean12=np.array([amymean[0],ercmean[0],hippomean[0]])
mean15=np.array([amymean[1],ercmean[1],hippomean[1]])
mean20=np.array([amymean[2],ercmean[2],hippomean[2]])
print(mean12)
print(mean15)
print(mean20)
std12=np.array([amystd[0],ercstd[0],hippostd[0]])
std15=np.array([amystd[1],ercstd[1],hippostd[1]])
std20=np.array([amystd[2],ercstd[2],hippostd[2]])
print(std12)
print(std15)
print(std20)