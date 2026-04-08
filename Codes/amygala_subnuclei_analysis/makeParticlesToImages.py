import torch
import numpy as np
import os
import glob
import scipy.io as sio
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from EmpiricalDistributions import EmpiricalDistributions
import nibabel as nib
from datetime import datetime
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

def buildSegmentation(sFile, ld, areatype, res=0.5, nn=True, thresh=0.005):
    brain_example = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

    if '.pt' in sFile:
        info = torch.load(sFile,weights_only=True)
        S = info['D'].cpu().numpy()
        if 'nu_D2' not in info:
            raise ValueError(f"{sFile} does not contain nu_D2 for substructures")
        nuS = info['nu_D2'].cpu().numpy()
    else:
        info = np.load(sFile,weights_only=True)
        S = info[info.files[0]]
        nuS = info[info.files[2]]  # Adjust index if needed

    # Grid
    minCoords = np.min(S, axis=0) - 2.0 * res
    maxCoords = np.max(S, axis=0) + 2.0 * res
    x0 = np.arange(minCoords[0], maxCoords[0], res)
    x1 = np.arange(minCoords[1], maxCoords[1], res)
    x2 = np.arange(minCoords[2], maxCoords[2], res)
    X0, X1, X2 = np.meshgrid(x0, x1, x2, indexing='ij')
    G = np.stack([X0.ravel(), X1.ravel(), X2.ravel()], axis=-1)
    nuG = np.ones((G.shape[0], 1))
    #print(x0.shape)
    # Assign
    ed = EmpiricalDistributions(
        torch.tensor(G, dtype=torch.float32),
        torch.tensor(S, dtype=torch.float32),
        torch.tensor(nuG, dtype=torch.float32),
        torch.tensor(nuS, dtype=torch.float32)
    )
    reAssign = ed.NNAssign() if nn else ed.GaussianAssign(res)
    reAssign = reAssign.cpu().numpy()  # [Ngrid, Nsub]
    #print(np.max(reAssign[:,0]))
    # Build segmentation mask
    seg = np.zeros(G.shape[0], dtype=np.int32)
    volumes=[]
    # For each substructure (1 to 15), assign label if exceeds threshold
    for i in range(len(brain_example)):
        mask = reAssign[:, i] > thresh
        seg[mask] = brain_example[i]
        volumes.append(float(np.sum(mask)*res**3))
    #mask = reAssign[:, 0] > thresh
    #print(np.sum(mask))
    seg = seg.reshape(X0.shape)
    aff = np.eye(4) * res
    aff[-1, -1] = 1.0
    nib.save(nib.Nifti1Image(seg.astype(np.float32), aff), f'/cis/home/yxie91/HF2BIOCARD/Substructure_mask/{ld}_{areatype}.nii.gz')
    return volumes

# Example usage
s='SubjectAtlas_Params2/example/amygdala/atlas_deformationSummary.pt'
volumes=buildSegmentation(s, "example", "amygdala", res=0.5, nn=True, thresh=0.005)