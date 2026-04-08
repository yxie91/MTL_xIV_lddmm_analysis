import os
import os
import glob
import numpy as np
import sys


from vtkFunctions import writeVTK
import nibabel as nib


def downsampleBrainSegs(filename,segMap,origRes,savename):
    image = nib.load(filename)
    print(image.affine)
    sl = np.asarray(image.dataobj)
    
    sl0 = sl[0::5,...]  # X/5 x Y x Z
    sl1 = sl[1::5,...]
    sl2 = sl[2::5,...]
    sl3 = sl[3::5,...]
    sl4 = sl[4::5,...]
    if sl4.shape != sl0.shape:
        sl4t = np.zeros_like(sl0)
        sl4t[0:sl4.shape[0],0:sl4.shape[1],0:sl4.shape[2],...] = sl4
        sl4 = sl4t
    if sl3.shape != sl0.shape:
        sl3t = np.zeros_like(sl0)
        sl3t[0:sl3.shape[0],0:sl3.shape[1],0:sl3.shape[2],...] = sl3
        sl3 = sl3t
    if sl2.shape != sl0.shape:
        sl2t = np.zeros_like(sl0)
        sl2t[0:sl2.shape[0],0:sl2.shape[1],0:sl2.shape[2],...] = sl2
        sl2 = sl2t
    if sl1.shape != sl0.shape:
        sl1t = np.zeros_like(sl0)
        sl1t[0:sl1.shape[0],0:sl1.shape[1],0:sl1.shape[2],...] = sl1
        sl1 = sl1t
            
    sls = np.stack((sl0,sl1,sl2,sl3,sl4),axis=-1) # X/5 x Y x Z x 5
        
    sls0 = sls[:,0::5,...] # X/5 x Y/5 x Z x 5
    sls1 = sls[:,1::5,...]
    sls2 = sls[:,2::5,...]
    sls3 = sls[:,3::5,...]
    sls4 = sls[:,4::5,...]
    if sls4.shape != sls0.shape:
        sl4t = np.zeros_like(sls0)
        sl4t[0:sls4.shape[0],0:sls4.shape[1],0:sls4.shape[2],...] = sls4
        sls4 = sl4t
    if sls3.shape != sls0.shape:
        sl3t = np.zeros_like(sls0)
        sl3t[0:sls3.shape[0],0:sls3.shape[1],0:sls3.shape[2],...] = sls3
        sls3 = sl3t
    if sls2.shape != sls0.shape:
        sl2t = np.zeros_like(sls0)
        sl2t[0:sls2.shape[0],0:sls2.shape[1],0:sls2.shape[2],...] = sls2
        sls2 = sl2t
    if sls1.shape != sls0.shape:
        sl1t = np.zeros_like(sls0)
        sl1t[0:sls1.shape[0],0:sls1.shape[1],0:sls1.shape[2],...] = sls1
        sls1 = sl1t
    
    slls = np.stack((sls0,sls1,sls2,sls3,sls4),axis=-1) # X/5 x Y/5 x Z x 5 x 5
    
    slls0 = slls[:,:,0::5,...] # X/5 x Y/5 x Z/5 x 5 x 5
    slls1 = slls[:,:,1::5,...]
    slls2 = slls[:,:,2::5,...]
    slls3 = slls[:,:,3::5,...]
    slls4 = slls[:,:,4::5,...]
    
    print("shapes of slls")
    print(slls0.shape)
    print(slls1.shape)
    print(slls2.shape)
    print(slls3.shape)
    print(slls4.shape)
    
    if slls4.shape != slls0.shape:
        sl4t = np.zeros_like(slls0)
        sl4t[0:slls4.shape[0],0:slls4.shape[1],0:slls4.shape[2],...] = slls4
        slls4 = sl4t
    if slls3.shape != slls0.shape:
        sl3t = np.zeros_like(slls0)
        sl3t[0:slls3.shape[0],0:slls3.shape[1],0:slls3.shape[2],...] = slls3
        slls3 = sl3t
    if slls2.shape != slls0.shape:
        sl2t = np.zeros_like(slls0)
        sl2t[0:slls2.shape[0],0:slls2.shape[1],0:slls2.shape[2],...] = slls2
        slls2 = sl2t
    if slls1.shape != slls0.shape:
        sl1t = np.zeros_like(slls0)
        sl1t[0:slls1.shape[0],0:slls1.shape[1],0:slls1.shape[2],...] = slls1
        slls1 = sl1t
    
    x = np.arange(slls0.shape[0])*5*origRes
    x = x - np.mean(x)
    y = np.arange(slls0.shape[1])*5*origRes
    y = y - np.mean(y)
    z = np.arange(slls0.shape[2])*5*origRes
    z = z - np.mean(z)
        
    X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
    XYZ = np.zeros((X.shape[0]*X.shape[1]*X.shape[2],3))
    XYZ[:,0] = np.ravel(X)
    XYZ[:,1] = np.ravel(Y)
    XYZ[:,2] = np.ravel(Z)
    
        
    newshape = (slls0.shape[0]*slls0.shape[1]*slls0.shape[2],5,5)
    slls0 = np.reshape(slls0,newshape)
    slls1 = np.reshape(slls1,newshape)
    slls2 = np.reshape(slls2,newshape)
    slls3 = np.reshape(slls3,newshape)
    slls4 = np.reshape(slls4,newshape)
    
    nuAll = np.zeros((XYZ.shape[0],sum([len(s) for s in segMap])))
    numReg = len(segMap)
    nus = []
    for i in range(numReg):
        nu = np.zeros((XYZ.shape[0],len(segMap[i])))
        for j in range(len(segMap[i])):
            for ds in range(5):
                for dss in range(5):
                    inds = slls0[:,ds,dss] == segMap[i][j]
                    nu[inds,j] += 1.0
                    inds = slls1[:,ds,dss] == segMap[i][j]
                    nu[inds,j] += 1.0
                    inds = slls2[:,ds,dss] == segMap[i][j]
                    nu[inds,j] += 1.0
                    inds = slls3[:,ds,dss] == segMap[i][j]
                    nu[inds,j] += 1.0
                    inds = slls4[:,ds,dss] == segMap[i][j]
                    nu[inds,j] += 1.0
        nus.append(nu)
        
    nuAll = np.hstack(nus)
    nuSub = np.hstack([np.sum(n,axis=-1)[...,None] for n in nus])
    
    # remove extraneous
    inds = np.sum(nuSub,axis=-1) > 0
    XYZtissue = XYZ[inds,...]
    nuAll = nuAll[inds,...]*((origRes)**3)
    nuSub = nuSub[inds,...]*((origRes)**3)
    zAll = nuAll/np.sum(nuAll,axis=-1)[...,None]
    zSub = nuSub / np.sum(nuSub,axis=-1)[...,None]
    eAll = np.zeros_like(zAll)
    eSub = np.zeros_like(zSub)
    eAll[zAll > 0] = zAll[zAll > 0] * np.log(zAll[zAll > 0])
    eSub[zSub > 0] = zSub[zSub > 0] * np.log(zSub[zSub > 0])
    
    print("number of particles: ", XYZtissue.shape[0])
    np.savez(savename,X=XYZtissue,nu_Sub=nuSub,nu_All=nuAll)
    writeVTK(XYZtissue,[np.argmax(nuAll,axis=-1)+1.0,np.argmax(nuSub,axis=-1)+1.0,np.sum(nuAll,axis=-1),np.sum(nuSub,axis=-1),-1.0 * np.sum(eAll,axis=-1),-1.0*np.sum(eSub,axis=-1)],['Subregion','Region','Weight_Subregion','Weight_Region','Entropy_Subregion','Entropy_Region'],savename.replace('.npz','.vtk'))
        
    return


def makeBrainSegs(filename,segMap,res,savename):
    image = nib.load(filename)
    im = np.asarray(image.dataobj)
    
    x = np.arange(im.shape[0])*res
    y = np.arange(im.shape[1])*res
    z = np.arange(im.shape[2])*res
    
    x = x - np.mean(x)
    y = y - np.mean(y)
    z = z - np.mean(z)
    
    X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
    XYZ = np.stack((X.ravel(),Y.ravel(),Z.ravel()),axis=-1)
    imR = im.ravel()
    
    # remove background first:
    inds = imR > 0
    XYZtissue = XYZ[inds,...]
    tissue = imR[inds,...]
    
    # Make Particles for All Subfields
    nuAll = np.zeros((XYZtissue.shape[0],sum([len(s) for s in segMap])))
    numReg = len(segMap)
    nus = []
    for i in range(numReg):
        nu = np.zeros((XYZtissue.shape[0],len(segMap[i])))
        for j in range(len(segMap[i])):
            inds = tissue == segMap[i][j]
            nu[inds,j] = 1.0
        nus.append(nu)
    nuAll = np.hstack(nus)
    nuSub = np.hstack([np.sum(n,axis=-1)[...,None] for n in nus])
    
    # remove extraneous
    inds = np.sum(nuSub,axis=-1) > 0
    XYZtissue = XYZtissue[inds,...]
    nuAll = nuAll[inds,...]
    nuSub = nuSub[inds,...]
    
    print("number of particles: ", XYZtissue.shape[0])
    np.savez(savename,X=XYZtissue,nu_Sub=nuSub,nu_All=nuAll)
    writeVTK(XYZtissue,[np.argmax(nuAll,axis=-1)+1.0,np.argmax(nuSub,axis=-1)+1.0],['Subregion','Region'],savename.replace('.npz','.vtk'))
    return

if __name__=='__main__':
    # Dummy files and brain label exmaples
    brain_example = [[1,2,3,4],#BLA, BMA, CM, LA
            [5,6], #ERC, Extension
            [7,8,9,10,11,12,13,14,15]] ##para, pre, sub, ca1, ca2, ca3, mol, gran, hil
    res = 0.125
    brain2name = 'HF_example.nii.gz'
    savename1="HF_example.npz"
    savename2="HF_example_ds5.npz"
    makeBrainSegs(brain2name,brain_example,res,savename1)
    downsampleBrainSegs(brain2name,brain_example,res,savename2)