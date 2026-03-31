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
from matplotlib.lines import Line2D
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
def Atrophy_Plot(controltype, areatype, manual,Hemi="LH",bottom=200,top=1300,outlier_subject=[]):
    print('-------------------')
    print(controltype, areatype, manual,Hemi)
    print('-------------------')
    control_DATA=sio.loadmat('Data/Sheets/rawthk_v13_kms_con.mat')
    pre_DATA=sio.loadmat('Data/Sheets/rawthk_v13_kms_pre.mat')
    mci_DATA=sio.loadmat('Data/Sheets/rawthk_v13_kms_mci.mat')
    if Hemi=="LH":
        predicted_root="Data/Dataset103_ADNIall/labelsTs"
    elif Hemi=="RH":
        predicted_root="Data/Dataset103_ADNIall/labelsTs_RH"      
    elif Hemi=="FUSE":
        predicted_root="Data/Dataset103_ADNIall/labelsTs_fuse"
    else:
        raise TypeError('Hemishpere not recognized')
    manual_root="Data/ADNI12GO_manual"
    if controltype=='con':
        SUBJECT=control_DATA["SUBJSc"]
        AGE=control_DATA["x1_c"]
    elif controltype=='pre':
        SUBJECT=pre_DATA["SUBJSp"]
        AGE=pre_DATA["x1_p"]
    elif controltype=='mci':
        SUBJECT=mci_DATA["SUBJSm"]
        AGE=mci_DATA["x1_m"]
    else:
        raise TypeError('Controltype not recognized')

    outdict={}
    VA=[]
    plt.figure(figsize=(10, 10))
    fontsize=30
    for i in range(len(SUBJECT)):
        subject=SUBJECT[i][0][0]
        ages=AGE[i][0][0]
        if len(ages)<=2:
            print(f"Subject {subject} less than 3 time points")
        volumes=[]
        if manual:
            subject_folder=os.listdir(f"{manual_root}/{subject}")
            segfiles=sorted([file for file in subject_folder if "cy2.nii.gz" in file], key=lambda x: float(x.split('_')[1]))
        else:
            segfiles=sorted([file for file in os.listdir(predicted_root) if subject in file], key=lambda x: float(x.split('_')[1].split('.')[0]))
        if len(segfiles)!=len(ages):
            print(f"{subject} is skipped due to segfile-age consistency")
            continue
        if subject in outlier_subject:
            print(f"{subject} is skipped due to outlier")
            continue
        valid_ages=[]
        valid_volumes=[]
        for k in range(len(segfiles)):
            if manual:
                seg_path= f"{manual_root}/{subject}/{segfiles[k]}"
            else:
                seg_path=f"{predicted_root}/{segfiles[k]}"
            seg_img= np.array(nib.load(seg_path).get_fdata()).squeeze()
            mask = np.ones_like(seg_img)
            if areatype=='Amygdala':
                voxel=mask[seg_img==1]
                volume=1.2*1*1*voxel.shape[0]
                if Hemi=="LH":
                    std=246.23545244842853
                elif Hemi=="RH":
                    std=231.46963566862954
                elif Hemi=="FUSE":
                    std=464.19775089577183
                else:
                    raise TypeError("Hemishpere not recognized") 
            elif areatype=='ERCTEC':
                voxel=np.append(mask[seg_img==2],mask[seg_img==3])
                volume=1.2*1*1*voxel.shape[0]
                if Hemi=="LH":
                    std=357.64888083964075
                elif Hemi=="RH":
                    std=333.9343029707212
                elif Hemi=="FUSE":
                    std=655.2641658499751
                else:
                    raise TypeError("Hemishpere not recognized") 
            elif areatype=='Hippocampus':
                voxel=np.append(mask[seg_img==4],mask[seg_img==5])
                volume=1.2*1*1*voxel.shape[0]
                if Hemi=="LH":
                    std=419.4770699870232
                elif Hemi=="RH":
                    std=455.00704988336736
                elif Hemi=="FUSE":
                    std=860.5986319404608
                else:
                    raise TypeError("Hemishpere not recognized") 
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
            print(f"{subject} has less than three tps")
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
    plt.xlim(left=45,right=95)
    plt.ylim(bottom=bottom,top=top)
    plt.xlabel('Age (years)',fontsize=fontsize)
    plt.ylabel('Tissue Volume (mm$^3$)',fontsize=fontsize)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    if Hemi=="LH":
        plt.scatter(0,0,marker="o",s=60,color=color,label=f'Left Hemishpere, {np.mean(VA):.3f} $\pm$ {np.std(VA):.3f} % / yr')
    elif Hemi=="RH":
        plt.scatter(0,0,marker="o",s=60,color=color,label=f'Right Hemisphere, {np.mean(VA):.3f} $\pm$ {np.std(VA):.3f} % / yr')
    elif Hemi=="FUSE":
        plt.scatter(0,0,marker="o",s=60,color=color,label=f'Left & Right, {np.mean(VA):.3f} $\pm$ {np.std(VA):.3f} % / yr')
    else:
        raise TypeError('Hemishpere not recognized')
    plt.legend(loc='upper left',fontsize=fontsize-6)
    plt.tight_layout()
    plt.savefig(f"Figure/F6_ADNI_{Hemi}_VA_{controltype}_{areatype}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
outlier_subject=[]
ercbot,erctop=200,2100
#Atrophy_Plot(controltype="mci", areatype="ERCTEC", manual=False, Hemi="LH",bottom=ercbot,top=erctop,outlier_subject=outlier_subject)
#Atrophy_Plot(controltype="mci", areatype="ERCTEC", manual=False, Hemi="RH",bottom=ercbot,top=erctop,outlier_subject=outlier_subject)
ercbot,erctop=400,4200
Atrophy_Plot(controltype="mci", areatype="ERCTEC", manual=False, Hemi="FUSE",bottom=ercbot,top=erctop,outlier_subject=outlier_subject)
