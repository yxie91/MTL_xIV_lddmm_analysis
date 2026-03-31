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
SUBJECT_12=pd.read_excel('/cis/home/yxie91/paper2025/Data/Sheets/MCIDEM_SUBJECT_DOB_multiple.xlsx',index_col=0,header=0).index.values
DATA_15=pd.read_csv('/cis/home/yxie91/paper2025/Data/Sheets/Demographic_15T.csv',index_col=0)
SUBJECT_15=DATA_15[DATA_15['DIAGNOSIS'].isin(['MCI', 'DEMENTIA'])].index.values
DATA_20=pd.read_csv('/cis/home/yxie91/paper2025/Data/Sheets/Demographic_20T.csv',index_col=0)
SUBJECT_20=DATA_20[DATA_20['DIAGNOSIS'].isin(['MCI', 'DEMENTIA'])].index.values
#ONLY USE MCI GROUP
def volumes(folder, outlier_file="", subjects=[], tag=""):
    print(subjects)
    if outlier_file != "":
        outlier_set = pd.read_excel(outlier_file, index_col=0).index.values
    else:
        outlier_set = []

    files = [s for s in os.listdir(folder) if s[:-7] not in outlier_set and s.split('_')[0] in subjects]
    print(len(os.listdir(folder)), len(files))


    V_amy = []
    V_erc = []
    V_hip = []

    for file in sorted(files):
        if not file.endswith(".nii.gz"):
            continue
        if file[:-7] in outlier_set:
            raise ValueError(f"{file[:-7]} is an outlier but not filtered")
        if file.split("_")[0] not in subjects:
            raise ValueError(f"{file.split('_')[0]} is not in the right diagnosis group but not filtered")
        
        seg_path = os.path.join(folder, file)
        seg = np.asarray(nib.load(seg_path).dataobj).squeeze()
        ones = np.ones_like(seg, dtype=np.uint8)

        # labels: 1 (amygdala), 2/3 (ERC/TEC), 4/5 (hippocampus)
        V_amy.append(resolution * ones[seg == 1].size)
        V_erc.append(resolution * (np.append(ones[seg == 2], ones[seg == 3]).size))
        V_hip.append(resolution * (np.append(ones[seg == 4], ones[seg == 5]).size))

    print(tag)
    print("mean:", np.mean(V_amy), np.mean(V_erc), np.mean(V_hip))
    print("std:", np.std(V_amy), np.std(V_erc), np.std(V_hip))
    print("n:", len(V_amy), len(V_erc), len(V_hip))

    return V_amy, V_erc, V_hip


def compare_volumes_independent(V_ref, V_cmp, ref_tag="1.2mm", cmp_tag="1.5mm"):
    region_names = ["Amygdala", "ERC", "Hippocampus"]

    print("\n" + "=" * 80)
    print(f"{cmp_tag} > {ref_tag} (one-sided independent t-test)")
    print("=" * 80)

    for i, region in enumerate(region_names):
        x = np.array(V_ref[i], dtype=float)
        y = np.array(V_cmp[i], dtype=float)

        # Welch's independent t-test, one-sided: cmp > ref
        t_stat, p_val = stats.ttest_ind(
            y, x,
            equal_var=False,
            nan_policy="omit",
            alternative="greater"
        )

        print(region)
        print(f"{ref_tag}: mean={np.mean(x):.4f}, std={np.std(x, ddof=1):.4f}, n={len(x)}")
        print(f"{cmp_tag}: mean={np.mean(y):.4f}, std={np.std(y, ddof=1):.4f}, n={len(y)}")
        print(f"one-sided t-test H1: {cmp_tag} > {ref_tag}")
        print(f"t = {t_stat:.6g}, p = {p_val:.6g}")
        print("------------------------------")


V_amy_3T, V_erc_3T, V_hip_3T = volumes(
    "/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset006_axis0/imagesTs_pred_fuse",subjects=SUBJECT_12,
    tag="BIOCARD 3T"
)

V_amy_15, V_erc_15, V_hip_15 = volumes(
    "/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset006_axis0/labelsTs_15mm_fuse",
    '/cis/home/yxie91/1520TBIOCARD/15Outlier.xlsx',subjects=SUBJECT_15,
    tag="BIOCARD 1.5 mm"
)

V_amy_20, V_erc_20, V_hip_20 = volumes(
    "/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset006_axis0/labelsTs_20mm_fuse",
    '/cis/home/yxie91/1520TBIOCARD/20Outlier.xlsx',subjects=SUBJECT_20,
    tag="BIOCARD 2.0 mm"
)

compare_volumes_independent(
    [V_amy_3T, V_erc_3T, V_hip_3T],
    [V_amy_15, V_erc_15, V_hip_15],
    ref_tag="1.2 mm (3T)",
    cmp_tag="1.5 mm"
)

compare_volumes_independent(
    [V_amy_3T, V_erc_3T, V_hip_3T],
    [V_amy_20, V_erc_20, V_hip_20],
    ref_tag="1.2 mm (3T)",
    cmp_tag="2.0 mm"
)