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
def tp_atrophy_cal(ages,volumes):
    ages = np.array(ages).reshape(-1, 1)
    volumes = np.array(volumes)
    model = LinearRegression()
    model.fit(ages, volumes)
    vol0=volumes[0]
    atrophy_rate=-model.coef_[0]/vol0*100
    return atrophy_rate,model
def Atrophy_Plot(seg_root,controltype, areatype,bottom=2000,top=4250,Hemi="LH"):
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
    outdict={}
    VA=[]
    plt.figure(figsize=(10, 10))
    fontsize=30
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
            age=ages[k]
            timepoint=str(int(TIMEPOINTS[k]))
            seg_path= seg_root + f'/{subject}_{timepoint}.nii.gz'
            seg_img= np.array(nib.load(seg_path).get_fdata()).squeeze()
            mask = np.ones_like(seg_img)
            if areatype=='Amygdala':
                voxel=mask[seg_img==1]
                volume=1.2*1*1*voxel.shape[0]
                if Hemi=="LH":
                    std=115.961381312612
                elif Hemi=="RH":
                    std=112.01319204941122
                elif Hemi=="FUSE":
                    std=216.67356115587256
                else:
                    raise TypeError("Hemishpere not recognized") 
            elif areatype=='ERCTEC':
                voxel=np.append(mask[seg_img==2],mask[seg_img==3])
                volume=1.2*1*1*voxel.shape[0]
                if Hemi=="LH":
                    std=252.01811732990512
                elif Hemi=="RH":
                    std=237.6609752942728
                elif Hemi=="FUSE":
                    std=446.50853548460185
                else:
                    raise TypeError("Hemishpere not recognized") 
            elif areatype=='Hippocampus':
                voxel=np.append(mask[seg_img==4],mask[seg_img==5])
                volume=1.2*1*1*voxel.shape[0]
                if Hemi=="LH":
                    std=374.5545896243793
                elif Hemi=="RH":
                    std=365.61216726278883
                elif Hemi=="FUSE":
                    std=718.5271953107335
                else:
                    raise TypeError("Hemishpere not recognized") 
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
        print(subject,valid_ages,valid_volumes,atrophy_rate)
        VA.append(atrophy_rate)
        ages1=np.linspace(valid_ages[0],valid_ages[-1],100) 
        y1=model.predict(ages1.reshape(-1,1))
        if areatype=='Amygdala':
            color='red'
        elif areatype=='ERCTEC':
            color='blue'
        elif areatype=='Hippocampus':
            color='green'
        else:
            raise TypeError("Areatype not recognized")
        plt.scatter(valid_ages,valid_volumes,color=color,marker='o',s=60)
        plt.plot(ages1,y1,'--',linewidth=1.5,color=color)
        out=[atrophy_rate]+valid_volumes
        outdict[subject]=out
    output=pd.DataFrame(outdict.values(),index=outdict.keys())
    num_columns = output.shape[1]
    column_names=['atrophy rate'] + [f'tp{i}' for i in range(1, num_columns)]
    output.columns=column_names
    #output.to_csv(f'Results/{manual}_{Hemi}_{controltype}_{areatype}.csv',header=True)
    if controltype=='Control':
        plt.xlim(left=40,right=95)
    elif controltype=='MCIDEM':
        plt.xlim(left=45,right=95)
    else:
        raise TypeError('Controltype not recognized')
    plt.ylim(bottom=bottom,top=top)
    plt.xlabel('Age (years)',fontsize=fontsize)
    plt.ylabel('Tissue Volume (mm$^3$)',fontsize=fontsize)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    if Hemi=="LH":
        plt.scatter(0,0,marker="o",s=60,color=color,label=f'Left Hemishpere, {np.mean(VA):.3f} $\pm$ {np.std(VA):.3f} % / yr')
    elif Hemi=="RH":
        plt.scatter(0,0,marker="o",s=60,color=color,label=f'Right Hemishpere, {np.mean(VA):.3f} $\pm$ {np.std(VA):.3f} % / yr')
    elif Hemi=="FUSE":
        plt.scatter(0,0,marker="o",s=60,color=color,label=f'Left & Right, {np.mean(VA):.3f} $\pm$ {np.std(VA):.3f} % / yr')
    else:
        raise TypeError('Hemishpere not recognized')
    plt.legend(loc='upper left',fontsize=fontsize-6)
    plt.tight_layout()
    plt.savefig(f"Figure/F6_BIOCARD_{Hemi}_VA_{controltype}_{areatype}.png", dpi=300, bbox_inches="tight")
    plt.close()

seg_root="/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset005_BIOCARD91/labelsTs_mprage_RH"#'Data/Dataset006_axis0/imagesTs_pred_mprage'
Hemi="RH"
Atrophy_Plot(seg_root,controltype="Control",areatype="ERCTEC",bottom=200,top=2100,Hemi=Hemi)
Atrophy_Plot(seg_root,controltype="MCIDEM",areatype="ERCTEC",bottom=200,top=2100,Hemi=Hemi)

seg_root="/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset005_BIOCARD91/labelsTs_mprage_LH"#'Data/Dataset006_axis0/imagesTs_pred_RH'
Hemi="LH"
Atrophy_Plot(seg_root,controltype="Control",areatype="ERCTEC",bottom=200,top=2100,Hemi=Hemi)
Atrophy_Plot(seg_root,controltype="MCIDEM",areatype="ERCTEC",bottom=200,top=2100,Hemi=Hemi)

'''seg_root='Data/Dataset006_axis0/imagesTs_pred_fuse'
Hemi="FUSE"
Atrophy_Plot(seg_root,controltype="MCIDEM",areatype="ERCTEC",bottom=400,top=4200,Hemi=Hemi)'''