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
    DATA=pd.read_csv('/cis/project/adni_yxie91/ADNI_T1all/Datasheets/ADNI_T1all_2_04_2026.csv',index_col=1)
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

    if controltype=="Control":
        SUBJECT=sorted(list(set(list(DATA[DATA['Group'].isin(['CN', 'SMC'])].index.values))))
    elif controltype=="MCI":
        SUBJECT=sorted(list(set(list(DATA[DATA['Group'].isin(["MCI"])].index.values))))#'EMCI', 'LMCI',
    elif controltype=="AD":
        SUBJECT=sorted(list(set(list(DATA[DATA['Group'].isin(["AD"])].index.values))))
    else:
        raise TypeError('Control type not recognized')
    outdict={}
    VA=[]
    plt.figure(figsize=(10, 10))
    fontsize=30
    outdict={}
    VA=[]
    #std_amy,std_ERCTEC,std_hippo=STD_volumes(predicted_root,outlier_path)
    #print(f"STD for three regions,amygdala:{std_amy},ERC:{std_ERCTEC},hippoampus:{std_hippo}")
    #STD for three regions,amygdala:479.8022500901658,ERC:637.8925133080927,hippoampus:887.6609674833333
    print(len(SUBJECT))
    for i in range(len(SUBJECT)):
        subject=SUBJECT[i]
        segfiles=sorted([file for file in os.listdir(predicted_root) if subject in file])
        if segfiles == [] or subject in outlier_subject:
            continue
        subject_rows = DATA.loc[[subject]]
        #print(subject)
        #print(subject_rows['Acq Date'].tolist(), subject_rows['Age'].tolist())
        date2age = dict(zip(subject_rows['Acq Date'].tolist(), subject_rows['Age'].tolist()))
        date2age = {datetime.strptime(k, '%m/%d/%Y').date(): v for k, v in date2age.items()} #====Change it to the entry date and entry disease group maybe====
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
            '''if "_accel_" in seg_path or "_grappa2_" in seg_path or "ir_spgr" in seg_path:
                continue'''
            seg_img= np.array(nib.load(seg_path).get_fdata()).squeeze()
            mask = np.ones_like(seg_img)
            if areatype=='Amygdala':
                voxel=mask[seg_img==1]
                volume=1.2*1*1*voxel.shape[0]
                if Hemi=="LH":
                    std=257.5261245609133
                elif Hemi=="RH":
                    std=251.12306731868117
                elif Hemi=="FUSE":
                    std=491.2706445585293
                else:
                    raise TypeError("Hemishpere not recognized") 
            elif areatype=='ERCTEC':
                voxel=np.append(mask[seg_img==2],mask[seg_img==3])
                volume=1.2*1*1*voxel.shape[0]
                if Hemi=="LH":
                    std=347.18536214827424
                elif Hemi=="RH":
                    std=331.0693117630194
                elif Hemi=="FUSE":
                    std=634.1674837765971
                else:
                    raise TypeError("Hemishpere not recognized") 
            elif areatype=='Hippocampus':
                voxel=np.append(mask[seg_img==4],mask[seg_img==5])
                volume=1.2*1*1*voxel.shape[0]
                if Hemi=="LH":
                    std=481.0771039527792
                elif Hemi=="RH":
                    std=478.7100448813985
                elif Hemi=="FUSE":
                    std=933.8100679831414
                else:
                    raise TypeError("Hemishpere not recognized") 
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
    plt.savefig(f"Figure/F6_ADNIall_{Hemi}_VA_{controltype}_{areatype}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
outlier_subject=[]
ercbot,erctop=200,2100
Atrophy_Plot(controltype="MCI", areatype="ERCTEC", manual=False, Hemi="LH",bottom=ercbot,top=erctop,outlier_subject=outlier_subject)
Atrophy_Plot(controltype="MCI", areatype="ERCTEC", manual=False, Hemi="RH",bottom=ercbot,top=erctop,outlier_subject=outlier_subject)
Atrophy_Plot(controltype="AD", areatype="ERCTEC", manual=False, Hemi="LH",bottom=ercbot,top=erctop,outlier_subject=outlier_subject)
Atrophy_Plot(controltype="AD", areatype="ERCTEC", manual=False, Hemi="RH",bottom=ercbot,top=erctop,outlier_subject=outlier_subject)
#ercbot,erctop=400,4200
#Atrophy_Plot(controltype="mci", areatype="ERCTEC", manual=False, Hemi="FUSE",bottom=ercbot,top=erctop,outlier_subject=outlier_subject)
