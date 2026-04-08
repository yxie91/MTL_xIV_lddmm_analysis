import os
import os
import glob
import numpy as np
import pandas as pd
import sys
import scipy.io as sio

from vtkFunctions import writeVTK
import nibabel as nib

def makeBrainSegs(filename,segMap,savename):
    image = nib.load(filename)
    #print("image Affine Matrix",image.affine)
    res0 = np.abs(image.affine[0,0]) # voxel size in x-dimension 1.2
    res1 = np.abs(image.affine[1,1]) # voxel size in y-dimension 1
    res2 = np.abs(image.affine[2,2]) # voxel size in z-dimension 1
    im = np.asarray(image.dataobj)
    #real world coordinates
    x = np.arange(im.shape[0])*res0 # 170 * 1.2
    y = np.arange(im.shape[1])*res1 # 256 * 1
    z = np.arange(im.shape[2])*res2 #256 * 1
    # centered around zero
    x = x - np.mean(x)
    y = y - np.mean(y)
    z = z - np.mean(z)
    
    X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
    XYZ = np.stack((X.ravel(),Y.ravel(),Z.ravel()),axis=-1) # Coordinates flattened (170*170*256,3)
    imR = im.ravel() #flattened voxel intensities (170*170*256)
    #Each element in imR corresponds to the intensity of the voxel at that coordinate in XYZ.
    
    # remove background first:
    inds = imR > 0
    XYZtissue = XYZ[inds,...] #(N,3), N is the number of labels in seg_labels
    tissue = imR[inds,...] #(N,)
    # Make Particles for All Subfields
    nuAll = np.zeros((XYZtissue.shape[0],sum([len(s) for s in segMap]))) # (N,5)
    numReg = len(segMap) # numReg=3
    nus = []
    for i in range(numReg):
        nu = np.zeros((XYZtissue.shape[0],len(segMap[i]))) #(N,1/2/2)
        for j in range(len(segMap[i])):
            inds = tissue == segMap[i][j]
            nu[inds,j] = 1.0 #nu includes the indices for all the tissue equal to segMap[i][j]
        nus.append(nu) #[(N,1), (N,2),(N,2)]
    nuAll = np.hstack(nus) 
    #print(nuAll.shape)
    nuSub = np.hstack([np.sum(n,axis=-1)[...,None] for n in nus]) # [...,None] converts (N,) to (N,1), (N, 3)

    # remove extraneous
    inds = np.sum(nuSub,axis=-1) > 0
    XYZtissue = XYZtissue[inds,...]
    nuAll = nuAll[inds,...]
    nuSub = nuSub[inds,...]
    
    print("number of particles: ", XYZtissue.shape[0])
    print("should all be 1 particle", np.unique(nuAll))
    nuAll = nuAll * res0*res1*res2 # make weights into tissue area
    nuSub = nuSub * res0*res1*res2
    np.savez(savename,X=XYZtissue,nu_Sub=nuSub,nu_All=nuAll)
    writeVTK(XYZtissue,[np.argmax(nuAll,axis=-1)+1.0,np.argmax(nuSub,axis=-1)+1.0, np.sum(nuAll,axis=-1), np.sum(nuSub,axis=-1)],['Subregion','Region','Weight_Subregion','Weight_Region'],savename.replace('.npz','.vtk'))
    #np.argmax(nuSub,axis=-1)+1.0 Finds which subregion has the highest weight at each voxel (indexing starts from 1).
    return

if __name__=='__main__':
    # Dummy files and brain label exmaples
    f="LF_example.nii.gz"
    savename="LF_example.npz"
    segMap = [[1],#Amygdala
          [2,3],#ERC,TEC
          [4,5]]#Hippocampus Anterior, Hippocampus Posterior
    makeBrainSegs(f,segMap,savename)