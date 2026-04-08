import numpy as np
import glob
import torch
import sys
import os
import scipy.io as sio

def splitHighField(npzFile,segMap):
    '''
    Split set of particles into sets according to mass associated to each region and subregion.
    Assume first index is regions, second is subregions and resave all.
    '''
    lenA = len(segMap[0])
    lenE = len(segMap[1])
    lenH = len(segMap[2])
    
    if '.pt' in npzFile:
        info = torch.load(npzFile)
        k = list(info.keys())
        X = info[k[0]].detach().cpu().numpy()
        nuR = info[k[1]].detach().cpu().numpy()
        nuS = info[k[2]].detach().cpu().numpy()
        suff = '.pt'
    else:
        info = np.load(npzFile)
        X = info[info.files[0]]# Positions of the tissues (nunm_voxels, 3) x,y,z
        nuR = info[info.files[1]] # Regions weight for Amygdala/ ERC/TEC/ Hippocampus (nunm_voxels, 3)
        nuS = info[info.files[2]] # Regions weight for subregions (nunm_voxels, 15)
        suff = '.npz'
    
    # split by region
    # amygdala
    inds = nuR[:,0] > 0
    Xr = X[inds,...]
    nuRr = nuR[inds,...]
    nuRr[:,1:] = 0.0
    nuSr = nuS[inds,...]
    nuSr[:,lenA:] = 0.0
    np.savez(npzFile.replace(suff,'_amygdala.npz'),X=Xr,nuR=nuRr,nuS=nuSr)
    print(np.sum(np.sum(nuSr,axis=-1) - np.sum(nuRr,axis=-1)))
    
    # ERC
    inds = nuR[:,1] > 0
    Xr = X[inds,...]
    nuRr = nuR[inds,...]
    nuRr[:,0] = 0.0
    nuRr[:,2] = 0.0
    nuSr = nuS[inds,...]
    nuSr[:,0:lenA] = 0.0
    nuSr[:,lenA+lenE:] = 0.0
    np.savez(npzFile.replace(suff,'_erctec.npz'),X=Xr,nuR=nuRr,nuS=nuSr)
    print(np.sum(np.sum(nuSr,axis=-1) - np.sum(nuRr,axis=-1)))


    # hippocampus
    inds = nuR[:,2] > 0
    Xr = X[inds,...]
    nuRr = nuR[inds,...]
    nuRr[:,0] = 0.0
    nuRr[:,1] = 0.0
    nuSr = nuS[inds,...]
    nuSr[:,0:lenA+lenE] = 0.0
    np.savez(npzFile.replace(suff,'_hippocampus.npz'),X=Xr,nuR=nuRr,nuS=nuSr)
    print(np.sum(np.sum(nuSr,axis=-1) - np.sum(nuRr,axis=-1))) 
    return

def splitLowField(npzfile):
    
    info = np.load(npzfile)
    X = info[info.files[0]]
    nuR = info[info.files[1]]
    
    # amygdala
    inds = nuR[:,0] > 0
    Xr = X[inds,...]
    nuRr = nuR[inds,...]
    nuRr[:,1:] = 0.0
    np.savez(npzfile.replace('.npz','_amygdala.npz'),X=Xr,nuR=nuRr)
    #print(np.sum(np.sum(nuR[inds,...],axis=-1) - np.sum(nuRr,axis=-1)))

    # ERC
    inds = nuR[:,1] > 0
    Xr = X[inds,...]
    nuRr = nuR[inds,...]
    nuRr[:,0] = 0.0
    nuRr[:,2] = 0.0
    np.savez(npzfile.replace('.npz','_erctec.npz'),X=Xr,nuR=nuRr)
    #print(np.sum(np.sum(nuR[inds,...],axis=-1) - np.sum(nuRr,axis=-1)))
    
    # hippocampus
    inds = nuR[:,2] > 0
    Xr = X[inds,...]
    nuRr = nuR[inds,...]
    nuRr[:,0:2] = 0.0
    np.savez(npzfile.replace('.npz','_hippocampus.npz'),X=Xr,nuR=nuRr)
    #print(np.sum(np.sum(nuR[inds,...],axis=-1) - np.sum(nuRr,axis=-1)))

    return

if __name__ == '__main__':
    # Dummy files and brain label exmaples
    brain_example = [[1,2,3,4],#BLA, BMA, CM, LA
            [5,6], #ERC, Extension
            [7,8,9,10,11,12,13,14,15]] ##para, pre, sub, ca1, ca2, ca3, mol, gran, hil
    file1="Atlas_ShiftedAll/Atlases/example_meanAtlas.pt"
    splitHighField(file1,brain_example)
    file2="Atlas_ShiftedAll/TargetsShifted/example.npz"
    splitLowField(file2)