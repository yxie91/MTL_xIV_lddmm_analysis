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
from collections import defaultdict
def tp_atrophy_cal(ages,volumes):
    ages = np.array(ages).reshape(-1, 1)
    volumes = np.array(volumes)
    model = LinearRegression()
    model.fit(ages, volumes)
    vol0=volumes[0]
    atrophy_rate=-model.coef_[0]/vol0*100
    return atrophy_rate

def calculation_all(referencetype="CerebellumGM",datatype='tau',Hemi="FUSE",outlier_subject=[]):
    DATA=pd.read_csv('Data/Sheets/ADNI34_MPRAGE_metadata.csv',index_col=0)
    outlier_file=pd.read_excel('Data/Sheets/ADNI34Outlier.xlsx',index_col=0).index.values
    Amyloid_data=pd.read_csv('Data/Sheets/All_Subjects_UCBERKELEY_AMY_6MM_28Sep2025.csv',index_col=1)
    PET_files=os.listdir("Data/05_pet_niigz1")
    PETseg_match=pd.read_csv('Data/Sheets/PET_T1_matches.csv',index_col=2)
    if Hemi=="LH":
        predicted_root="Data/Dataset102_ADNI34/labelsTs"
    elif Hemi=="RH":
        predicted_root="Data/Dataset102_ADNI34/labelsTs_RH"
    elif Hemi=="FUSE":
        predicted_root="Data/Dataset102_ADNI34/labelsTs_fuse"
    else:
        raise TypeError('Hemishpere not recognized')
    SUBJECT_ALL=set(list(DATA[DATA['Group'].isin(['EMCI', 'LMCI',"MCI","AD"])].index.values))
    TAU_files=[f for f in PET_files if "Tau_AV1451" in f]
    #TAU_SUBJECT=list(set([f.split("__")[0] for f in TAU_files]))

    PIB_files=[f for f in PET_files if "Tau_AV1451" in f]
    #PIB_SUBJECT=list(set([f.split("__")[0] for f in PIB_files]))
    
    print(len(SUBJECT_ALL))

    GDATA=pd.read_csv('Data/Sheets/ADNI34_MPRAGE_metadata.csv',index_col=0)
    SUBJECT=set(list(GDATA[GDATA['Group'].isin(['EMCI', 'LMCI',"MCI","AD"])].index.values))
    noMRICloud_subjects=["114_S_4404","036_S_6878","168_S_6541","137_S_4631","041_S_1418","116_S_6100","036_S_6179","002_S_4654","130_S_6072","068_S_2184","023_S_6535","022_S_6796","023_S_6661","033_S_7079","019_S_6315","037_S_6377","123_S_4127","127_S_6241","016_S_6926",
                         "019_S_6668","036_S_6885","068_S_2315","023_S_6334","007_S_2394","035_S_7000","123_S_4170","011_S_6303","168_S_6591","053_S_6598","003_S_4354","094_S_2238","023_S_4115","137_S_4536","041_S_4271","016_S_6839","053_S_4813","024_S_4674","023_S_6702",
                         "002_S_4229","301_S_6297","024_S_6033","023_S_6369","033_S_7088","130_S_4294","168_S_6619","033_S_6705","024_S_2239","014_S_2308","006_S_4713"]
    outlier_subjects=[]
    SUBJECT=sorted([s for s in SUBJECT if s not in outlier_subjects])
    print(len(SUBJECT))
    MRICloud_root="Data/MRICloud_ADNI34"
    MRICloud_files=os.listdir(MRICloud_root)
    subjects=[]
    LA_subjects=[]
    HA_subjects=[]
    for subject in SUBJECT:
        amyloid_status = Amyloid_data.loc[[subject], "AMYLOID_STATUS"].tolist()
        #print(amyloid_status)
        if list(set(amyloid_status))[0]==0:
            subjects.append(subject)
            LA_subjects.append(subject)
        elif list(set(amyloid_status))[0]==1:
            subjects.append(subject)
            HA_subjects.append(subject) 
        if all(pd.isna(amyloid_status)):
            print(f"subject {subject} does not have amyloid level")
    print(len(subjects))
    VA1={};VA2={};VA3={};IL1={};IL2={};IL3={};IL4={};IL5={}
    sout=[]
    subject_idx=0
    tp_idx=0
    PET_num=[]
    noSUV_subjects=[]
    for i in range(len(subjects)):
        subject=subjects[i]
        segfiles=sorted([file for file in os.listdir(predicted_root) if subject in file])
        if segfiles == [] or subject in outlier_subject:
            continue
        subject_rows = DATA.loc[subject]
        date2age = dict(zip(subject_rows['Acq Date'], subject_rows['Age']))
        date2age = {datetime.strptime(k, '%m/%d/%Y').date(): v for k, v in date2age.items()}
        baseline_tp=sorted(date2age.keys())[0]
        baseline_age=date2age[baseline_tp]
        valid_ages=[];volumes1=[];volumes2=[];volumes3=[]
        for k in range(len(segfiles)):
            name=segfiles[k][:-7]
            img_id=segfiles[k].split("__")[3]
            if name in outlier_file: 
                #print(f"{name} not valid segfile")
                continue
            mc_file = [f for f in MRICloud_files if f"{img_id}" in f]
            if len(mc_file)<1:
                #print(f"{name} does not have MRICloud segmentation")
                noMRICloud_subjects.append(subject)
                continue
                #raise ValueError(f"{name} does not have MRICloud segmentation")
            if len(mc_file)>1:
                raise ValueError(f"{name} has more than one MRICloud segmentation")
            mc_file=mc_file[0]
            tp=datetime.strptime(name.split('__')[1], '%Y%m%d').date()
            age=baseline_age+(tp-baseline_tp).days/365.25
            seg_path=f"{predicted_root}/{segfiles[k]}"
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
            
            valid_ages.append(age)      

        if len(set(valid_ages)) >= 3:
            ages=valid_ages
        else:
            continue

        age_dict = defaultdict(list)
        for age, vol in zip(valid_ages, volumes1):
            age_dict[age].append(vol)
        group_valid_ages1 = []
        group_valid_volumes1 = []
        for age in sorted(age_dict.keys()):
            vals = np.asarray(age_dict[age], float)
            group_valid_ages1.append(age)
            group_valid_volumes1.append(sum(age_dict[age]) / len(age_dict[age]))

        age_dict = defaultdict(list)
        for age, vol in zip(valid_ages, volumes2):
            age_dict[age].append(vol)
        group_valid_ages2 = []
        group_valid_volumes2 = []
        for age in sorted(age_dict.keys()):
            vals = np.asarray(age_dict[age], float)
            group_valid_ages2.append(age)
            group_valid_volumes2.append(sum(age_dict[age]) / len(age_dict[age]))

        age_dict = defaultdict(list)
        for age, vol in zip(valid_ages, volumes3):
            age_dict[age].append(vol)
        #print(age_dict)
        group_valid_ages3 = []
        group_valid_volumes3 = []
        for age in sorted(age_dict.keys()):
            vals = np.asarray(age_dict[age], float)
            group_valid_ages3.append(age)
            group_valid_volumes3.append(sum(age_dict[age]) / len(age_dict[age]))
        assert group_valid_ages1==group_valid_ages2
        assert group_valid_ages1==group_valid_ages3
        subject_idx+=1
        tp_idx+=len(set(ages))
        VA1[subject]=tp_atrophy_cal(group_valid_ages1,group_valid_volumes1)
        VA2[subject]=tp_atrophy_cal(group_valid_ages2,group_valid_volumes2)
        VA3[subject]=tp_atrophy_cal(group_valid_ages3,group_valid_volumes3)
        
        if datatype=='Tau':
            PETfiles=sorted([s for s in TAU_files if subject in s])
        elif datatype=="Amyloid":
            PETfiles=sorted([s for s in PIB_files if subject in s])
        
        IL_amygdala=[]
        IL_ERCTEC=[]
        IL_hippo=[]
        IL_all=[]
        SUV_reference=[]
        f_path=""
        lPf=0
        for f in PETfiles:
            seg_name=PETseg_match.loc[f[:-7],"Matched_T1"]
            #print(seg_name)
            temp=[]
            for name in segfiles:
                if name.split("__")[3]==seg_name.split("__")[3]:
                    temp.append(name)
            if len(temp)!=1:
                print(name)
                raise ValueError(f"{temp} length != 1")
            seg_img= np.array(nib.load(f"{predicted_root}/{temp[0]}").get_fdata()).squeeze() 
            PET_img = np.array(nib.load(f"Data/05_pet_niigz1/{f}").get_fdata()).squeeze() 
            if np.isnan(PET_img).any():
                print(f"{f} has nan values")
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
            #print(f)
            mc_file = [f for f in MRICloud_files if f"{seg_name}" in f]
            if len(mc_file)!=1:
                print(mc_file)
                raise ValueError("MRICLoud segmentation number != 1")
            mc_file=mc_file[0]
            MRICloud_seg_path=f"{MRICloud_root}/{mc_file}/{mc_file}_286Labels.img"
            mc_seg_img= np.array(nib.load(MRICloud_seg_path).get_fdata()).squeeze()
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
                print(np.nanmean(amygdala_PET))
                print(np.nanmean(ERCTEC_PET))
                print(np.nanmean(hippo_PET))
                print((np.nansum(amygdala_PET)+np.nansum(ERCTEC_PET)+np.nansum(hippo_PET)*1.2*1*1)/(amy_volume+erc_volume+hippo_volume))
                '''IL_amygdala.append(np.nanmean(amygdala_PET))
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
            raise ValueError("Empty IL")
        '''if lPf>1:
            print(f"{subject} has more than one {datatype} file")'''
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
    print("=="*20)
    print(Hemi)
    amy=calculation_all(referencetype=referencetype,datatype='Amyloid',Hemi=Hemi)
    tau=calculation_all(referencetype=referencetype,datatype='Tau',Hemi=Hemi)

    OUT_ROI0=pd.concat([tau,amy],axis=1)
    OUT_ROI0.to_excel(f'Results/ADNI_MTLall_{Hemi}_{referencetype}.xlsx',index=True)

'''main_fn_ROI0("Pons","LH")
main_fn_ROI0("Pons","RH")
main_fn_ROI0("Pons","FUSE")'''

main_fn_ROI0("CerebellumGM","LH")
main_fn_ROI0("CerebellumGM","RH")
main_fn_ROI0("CerebellumGM","FUSE")