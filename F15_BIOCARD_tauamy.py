import scipy.io as sio
from ast import Str, Sub
from pickletools import uint8
import nibabel as nib
import os
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime
def find_and_load_image(filename, folders):
    for folder in folders:
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            return np.array(nib.load(filepath).get_fdata()).squeeze()
    
    print("Image not found in any folder.")
    return None
def tp_atrophy_cal(ages,volumes):
    ages = np.array(ages).reshape(-1, 1)
    volumes = np.array(volumes)
    model = LinearRegression()
    model.fit(ages, volumes)
    vol0=volumes[0]
    atrophy_rate=-model.coef_[0]/vol0*100
    return atrophy_rate

def calculation_all(referencetype="CerebellumGM",datatype='tau',Hemi="FUSE",outlier_subject=[]):#["CONLUC","HILCAR","SAUCON","WILJAY"]):#
    TP_pd=pd.read_excel('Data/Sheets/SUBJECT_TP_PET.xlsx',index_col=0,header=0)
    file1=os.listdir("Data/04_pet_nii_a")
    file2=os.listdir("Data/04_pet_nii_a_94pib")
    file3=os.listdir("Data/04_pet_nii_b")
    FILE=file1+file2+file3
    
    noSUV_subjects=['KORMIC', 'MCETHO', 'SHIJAM','SUMMAR']
    noMRICloud_subjects=[]
    
    if Hemi=="LH":
        seg_root='Data/Dataset006_axis0/imagesTs_pred_mprage'#'/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset005_BIOCARD91/labelsTs_mprage_LH'
    elif Hemi=="RH":
        seg_root='Data/Dataset006_axis0/imagesTs_pred_RH'#'/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset005_BIOCARD91/labelsTs_mprage_RH'
    elif Hemi=="FUSE":
        seg_root='Data/Dataset006_axis0/imagesTs_pred_fuse'#'/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset005_BIOCARD91/labelsTs_mprage_fuse'
    else:
        raise ValueError("Hemishpere not found")
    PET_folder=["Data/04_pet_nii_a",
                "Data/04_pet_nii_a_94pib",
                "Data/04_pet_nii_b"]
    PET_files=[file for file in FILE if ".nii.gz" in file]

    MRICloud_root="Data/MRICloud_BIOCARD"
    MRICloud_files=os.listdir(MRICloud_root)
    DATA=pd.read_excel('Data/Sheets/MCIDEM_SUBJECT_DOB_multiple.xlsx',index_col=0,header=0)
    SUBJECT_ALL=DATA.index
    TAU_files=[file for file in PET_files if 'tau' in file]
    TAU_SUBJECT=[name.split('_')[2] for name in TAU_files]

    PIB_files=[file for file in PET_files if 'pib' in file]
    PIB_SUBJECT=[name.split('_')[2] for name in PIB_files]
    print(len(SUBJECT_ALL))
    SUBJECT=[s for s in SUBJECT_ALL if s in TAU_SUBJECT and s in PIB_SUBJECT and s not in outlier_subject and s not in noSUV_subjects]

    print(len(SUBJECT))
    VA1={};VA2={};VA3={};IL1={};IL2={};IL3={};IL4={};IL5={}
    sout=[]
    subject_idx=0
    tp_idx=0
    PET_num=[]
    for i in range(len(SUBJECT)):
        subject=SUBJECT[i]
        #print(subject)
        TIMEPOINTS=TP_pd.loc[subject].dropna().values
        #print(TIMEPOINTS)
        dob=DATA.loc[subject,'DOB']
        date_format = "%y%m%d"
        ages=[]
        volumes1=[];volumes2=[];volumes3=[]
        for k in range(len(TIMEPOINTS)):
            tp=str(int(TIMEPOINTS[k]))
            mc_file = [f for f in MRICloud_files if f"{subject}_{tp}" in f]
            if len(mc_file)<1:
                noMRICloud_subjects.append(subject)
                #print(f"{subject} no MRI Cloud segmentation")
                continue
            if len(mc_file)>1:
                raise ValueError(f"{subject} has more than one MRI Cloudsegmentation")
            mc_file=mc_file[0]
            datetime1 = dob
            datetime2 = datetime.strptime(tp, date_format)
            age = datetime2.year - datetime1.year
            if (datetime2.month, datetime2.day) < (datetime1.month, datetime1.day):
                age -= 1
            ages.append(age)
            seg_path=seg_root+f'/{subject}_{tp}.nii.gz'
            seg_img= np.array(nib.load(seg_path).get_fdata()).squeeze()
            mask = np.ones_like(seg_img)

            voxel=mask[seg_img==1]
            volume=1.2*1*1*voxel.shape[0]
            volumes1.append(volume)

            voxel=np.append(mask[seg_img==2],mask[seg_img==3])
            volume=1.2*1*1*voxel.shape[0]
            volumes2.append(volume)

            voxel=np.append(mask[seg_img==4],mask[seg_img==5])
            volume=1.2*1*1*voxel.shape[0]
            volumes3.append(volume)
        if len(ages) < 3:
            continue
        #print(ages,volumes1)
        subject_idx+=1
        tp_idx+=len(set(ages))
        VA1[subject]=tp_atrophy_cal(ages,volumes1)
        VA2[subject]=tp_atrophy_cal(ages,volumes2)
        VA3[subject]=tp_atrophy_cal(ages,volumes3)
        
        if datatype=='Tau':
            PETfiles=[s for s in TAU_files if subject in s]
        elif datatype=='Amyloid':
            PETfiles=[s for s in PIB_files if subject in s]
        
        IL_amygdala=[]
        IL_ERCTEC=[]
        IL_hippo=[]
        IL_all=[]
        SUV_reference=[]
        f_path=""
        lPf=0
        for f in PETfiles:
            for folder in PET_folder:
                filepath = os.path.join(folder, f)
                if os.path.exists(filepath):
                    f_path=filepath
            segname="_".join(f.split("_")[2:4])
            seg_path=seg_root+f'/{segname}.nii.gz'
            seg_img= np.array(nib.load(seg_path).get_fdata()).squeeze() 
            PET_img= np.array(nib.load(f_path).get_fdata()).squeeze() 
            #PETraw_img = nib.load(f_path)
            #print(PETraw_img.header)
            #print(f)
            #print(PET_img.shape)
            
            '''if np.isnan(PET_img).any():
                print(f"{f_path} has nan values")'''
            amygdala_PET=PET_img[seg_img==1]
            ERCTEC_PET=np.append(PET_img[seg_img==2],PET_img[seg_img==3])
            hippo_PET = np.append(PET_img[seg_img==4],PET_img[seg_img==5]) 
            
            mask = np.ones_like(seg_img)
            voxel=mask[seg_img==1]
            amy_volume=1.2*1*1*voxel.shape[0]
            voxel=np.append(mask[seg_img==2],mask[seg_img==3])
            erc_volume=1.2*1*1*voxel.shape[0]
            voxel=np.append(mask[seg_img==4],mask[seg_img==5])
            hippo_volume=1.2*1*1*voxel.shape[0]
            
            MRICloud_seg_path=f"{MRICloud_root}/{mc_file}/{mc_file}_286Labels.img"
            mc_seg_img= np.array(nib.load(MRICloud_seg_path).get_fdata()).squeeze()
            #print(mc_seg_img.shape)
            if PET_img.shape!=mc_seg_img.shape:
                continue
            mask = np.ones_like(mc_seg_img)
            if referencetype=="CerebellumGM":
                if Hemi=="LH":
                    voxel=mask[mc_seg_img==96]#CerebellumGM_L
                    reference_volume=1.2*1*1*voxel.shape[0]
                    reference_PET=PET_img[mc_seg_img==96]
                elif Hemi=="RH":
                    voxel=mask[mc_seg_img==95]#CerebellumGM_R
                    reference_volume=1.2*1*1*voxel.shape[0]
                    reference_PET=PET_img[mc_seg_img==95]
                elif Hemi=="FUSE":
                    voxel=np.append(mask[mc_seg_img==95],mask[mc_seg_img==96])#CerebellumGM
                    reference_volume=1.2*1*1*voxel.shape[0]
                    reference_PET=np.append(PET_img[mc_seg_img==95],PET_img[mc_seg_img==96])
                else:
                    raise ValueError("Hemishpere not found")

            elif referencetype=="Pons":
                if Hemi=="LH":
                    voxel=mask[mc_seg_img==113]#Pons_L
                    reference_volume=1.2*1*1*voxel.shape[0]
                    reference_PET=PET_img[mc_seg_img==113]
                elif Hemi=="RH":
                    voxel=mask[mc_seg_img==114]#Pons_R
                    reference_volume=1.2*1*1*voxel.shape[0]
                    reference_PET=PET_img[mc_seg_img==114]
                elif Hemi=="FUSE":
                    voxel=np.append(mask[mc_seg_img==113],mask[mc_seg_img==114])#Pons
                    reference_volume=1.2*1*1*voxel.shape[0]
                    reference_PET=np.append(PET_img[mc_seg_img==113],PET_img[mc_seg_img==114])
                else:
                    raise ValueError("Hemishpere not found")

            if np.nansum(reference_PET)/reference_volume<100:
                '''print(np.nanmean(amygdala_PET))
                print(np.nanmean(ERCTEC_PET))
                print(np.nanmean(hippo_PET))
                print(np.nansum(amygdala_PET)+np.nansum(ERCTEC_PET)+np.nansum(hippo_PET)*1.2*1*1)/(amy_volume+erc_volume+hippo_volume))
                IL_amygdala.append(np.nanmean(amygdala_PET))
                IL_ERCTEC.append(np.nanmean(ERCTEC_PET))
                IL_hippo.append(np.nanmean(hippo_PET))
                IL_all.append((np.nansum(amygdala_PET)+np.nansum(ERCTEC_PET)+np.nansum(hippo_PET)*1.2*1*1)/(amy_volume+erc_volume+hippo_volume))'''
                continue
            else:
                lPf+=1
                '''print(np.nansum(amygdala_PET)/amy_volume)
                print(np.nansum(ERCTEC_PET)/erc_volume)
                print(np.nansum(hippo_PET)/hippo_volume)
                print(np.nansum(reference_PET)/reference_volume)'''
            
            IL_amygdala.append((np.nansum(amygdala_PET)/amy_volume)/(np.nansum(reference_PET)/reference_volume))
            IL_ERCTEC.append((np.nansum(ERCTEC_PET)/erc_volume)/(np.nansum(reference_PET)/reference_volume))
            IL_hippo.append((np.nansum(hippo_PET)/hippo_volume)/(np.nansum(reference_PET)/reference_volume))
            IL_all.append(((np.nansum(amygdala_PET)+np.nansum(ERCTEC_PET)+np.nansum(hippo_PET))/(amy_volume+erc_volume+hippo_volume))/(np.nansum(reference_PET)/reference_volume))
            SUV_reference.append(np.nansum(reference_PET)/reference_volume)
        if IL_amygdala!=[] and IL_ERCTEC!=[] and IL_hippo!=[]:
            IL1[subject]=np.mean(IL_amygdala)
            IL2[subject]=np.mean(IL_ERCTEC)
            IL3[subject]=np.mean(IL_hippo)
            IL4[subject]=np.mean(IL_all)
            IL5[subject]=np.mean(SUV_reference)
            #print(np.mean(IL_amygdala),np.mean(IL_ERCTEC),np.mean(IL_hippo))
        else:
            noSUV_subjects.append(subject)
            #continue
            raise ValueError("Empty IL")
        if lPf>1:
            print(f"{subject} has more than one {datatype} file")
        sout.append(subject)
        PET_num.append(lPf) 
    d={"Amygdala VA":VA1,"ERC/TEC VA":VA2,"Hippocampus VA":VA3,f"Amygdala {datatype}":IL1,f"ERC/TEC {datatype}":IL2,f"Hippocampus {datatype}":IL3,f"MTL {datatype}":IL4,f"{datatype} Reference SUV":IL5}
    pdout= pd.DataFrame.from_dict(d, orient="columns")
    #print(pdout)
    print(noSUV_subjects)
    print("Subjects with no MRICloud segmentations")
    print(*list(set(noMRICloud_subjects)),sep=",")
    print(f"In total we have {np.sum(PET_num)} {datatype} files, on average we have {np.mean(PET_num)} for each of the {subject_idx} subjects and {tp_idx} time points")
    return pdout

def main_fn_ROI0(referencetype,Hemi):
    
    amy=calculation_all(referencetype=referencetype,datatype='Amyloid',Hemi=Hemi)
    tau=calculation_all(referencetype=referencetype,datatype='Tau',Hemi=Hemi)

    OUT_ROI0=pd.concat([tau,amy],axis=1)
    OUT_ROI0.to_excel(f'Results/BIOCARD_MTLall_{Hemi}_{referencetype}.xlsx',index=True)

'''main_fn_ROI0("Pons","LH")
main_fn_ROI0("Pons","RH")
main_fn_ROI0("Pons","FUSE")'''

main_fn_ROI0("CerebellumGM","LH")
main_fn_ROI0("CerebellumGM","RH")
main_fn_ROI0("CerebellumGM","FUSE")