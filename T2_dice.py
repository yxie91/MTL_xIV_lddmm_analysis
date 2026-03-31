import SimpleITK as sitk
import scipy.io as sio
import numpy as np
import nibabel as nib
import pandas as pd
import random
import logging
import os
from scipy.stats import median_abs_deviation
def save_nifti(data, affine, file_path):
    """Save a 3D numpy array as a NIfTI file."""
    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, file_path)
import SimpleITK as sitk
import numpy as np
import nibabel as nib


def l1_loss(seg1, seg2):
    return np.sum(np.abs(seg1 - seg2))

def dice_loss(seg1, seg2, labels=[1, 2, 3, 4, 5]):
    dice_scores = []
    for label in labels:
        mask1 = (seg1 == label)
        mask2 = (seg2 == label)
        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1) + np.sum(mask2)
        #print(f"Intersection:{intersection},Union{union}")
        dice_score = 2 * intersection / (union + 1e-6)
        dice_scores.append(dice_score)
    #ERCTEC
    mask1 = (seg1 == 2) + (seg1 == 3)
    mask2 = (seg2 == 2) + (seg2 == 3)
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1) + np.sum(mask2)
    dice_score = 2 * intersection / (union + 1e-6)
    dice_scores.append(dice_score)
    #Hippocampus
    mask1 = (seg1 == 4) + (seg1 == 5)
    mask2 = (seg2 == 4) + (seg2 == 5)
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1) + np.sum(mask2)
    dice_score = 2 * intersection / (union + 1e-6)
    dice_scores.append(dice_score)
    return dice_scores #Amygdala, ERC,TEC, HA, HP, ERC/TEC. Hippocampus



# Load segmentation images
def main():
    with open("/cis/home/yxie91/paper1/Results/dicef1loss.log", "w"):
        pass
    logging.basicConfig(filename="/cis/home/yxie91/paper1/Results/dicef1loss.log", level=logging.INFO, format="%(asctime)s - %(message)s")
    manual_root='/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset006_axis0/labelsTr'
    predicted_root='/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset006_axis0/imagesTs_pred_mprage'
    FILE=sorted(os.listdir(manual_root))
    L1=[]
    DICE=[[],[],[],[],[],[],[]]
    for i in range(len(FILE)):
        file=FILE[i]
        logging.info('---------------------------------')
        logging.info(file)
        #print(file)
        seg_path=f"{manual_root}/{file}"
        manual_seg= np.array(nib.load(seg_path).get_fdata()).squeeze()
        seg_path=f"{predicted_root}/{file}"
        pred_seg= np.array(nib.load(seg_path).get_fdata()).squeeze()
        l1=l1_loss(manual_seg,pred_seg)
        dice=dice_loss(manual_seg,pred_seg)
        logging.info(f"{file} L1 loss {l1}")
        logging.info(f"{file} Dice loss {dice}")            
        L1.append(l1)
        for i in range(7):
            DICE[i].append(dice[i])
    logging.info(f"Overall  L1 loss Mean: {np.mean(L1)}, STD: {np.std(L1)}")
    for i in range(7):
        logging.info(f"Structure {i+1} Dice loss Mean: {np.mean(DICE[i])}, STD: {np.std(DICE[i])}")
        print(f"Structure {i+1} Dice loss Mean: {np.mean(DICE[i])}, STD: {np.std(DICE[i])}")

    manual_root='/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset103_ADNIall/labelsTr'
    predicted_root='/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset103_ADNIall/labelsTs'
    FILE=sorted(os.listdir(manual_root))
    L1=[]
    DICE=[[],[],[],[],[],[],[]]
    for i in range(len(FILE)):
        file=FILE[i]
        logging.info('---------------------------------')
        logging.info(file)
        seg_path=f"{manual_root}/{file}"
        manual_seg= np.array(nib.load(seg_path).get_fdata()).squeeze()
        seg_path=f"{predicted_root}/{file}"
        pred_seg= np.array(nib.load(seg_path).get_fdata()).squeeze()
        l1=l1_loss(manual_seg,pred_seg)
        dice=dice_loss(manual_seg,pred_seg)
        logging.info(f"{file} L1 loss {l1}")
        logging.info(f"{file} Dice loss {dice}")            
        L1.append(l1)
        for i in range(7):
            DICE[i].append(dice[i])
    logging.info(f"Overall  L1 loss Mean: {np.mean(L1)}, STD: {np.std(L1)}")
    for i in range(7):
        logging.info(f"Structure {i+1} Dice loss Mean: {np.mean(DICE[i])}, STD: {np.std(DICE[i])}")
        print(f"Structure {i+1} Dice loss Mean: {np.mean(DICE[i])}, STD: {np.std(DICE[i])}")
if __name__ == "__main__":
    main()