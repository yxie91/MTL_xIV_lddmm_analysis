import os
import sys
import torch

from sys import path as sys_path
sys_path.append("..")
sys_path.append("../..")
import glob
import xmodmap
import numpy as np
import scipy.io as sio
from xmodmap.preprocess.preprocess import resizeData, rescaleDataSM
from xmodmap.io.getOutput import writeParticleVTK, writeVTK, getJacobian, getEntropy
from xmodmap.io.getInput import getFromFile

def computeMomentum(dirBase, letterCode):
    '''
    Compute average initial momentum.
    '''
    print(dirBase + letterCode + '*/sm.pt') 
    fils = glob.glob(dirBase + letterCode + '*/precondVar.pt')
    
    # average initial momentum
    pxList = []
    pwList = []
    
    # average initial starting points of source 
    qxList = []
    qwList = []
    
    for f in fils:
        info = torch.load(f)
        pxList.append(info['px'])
        print(info['px'].shape)
        pwList.append(info['pw'])
        qxList.append(info['qx'])
        qwList.append(info['qw'])
    
    pxFinal = sum(pxList)/len(pxList)
    pwFinal = sum(pwList)/len(pwList)
    qxFinal = sum(qxList)/len(qxList)
    qwFinal = sum(qwList)/len(qwList)
    
    return pxFinal,pwFinal,qxFinal,qwFinal

def computeParameters(dirBase, letterCode):

    fils = glob.glob(dirBase + letterCode + '*/sm.pt')
    f0 = torch.load(fils[0])
    cA = f0['cA'] # affine transformation weight, source is all the same
    cT = f0['cT'] # target deformation weight
    cS = f0['cS'] # source deformation weight
    dimEff = f0['dimEff'] # Effective dimension (usually 3D)
    single = f0['single'] # If True, uses single-modality mapping
    sigmaRKHS = f0['sigmaRKHS'] # Regularization kernel width for smoothness
    
    s = f0['s'] # scaling factor for rescaling coordinates
    m = f0['m'] # mean shift for centering
    
    for i in range(1,len(fils)):
        f = torch.load(fils[i])
        s += f['s']
        m += f['m']
    s = s / len(fils)
    m = m / len(fils)
    
    return cA,cT,cS,dimEff,single,sigmaRKHS,s,m

def shootTemplate(dirbase, lettercode, zeta_S, dirsave,zeta_S2 = None):
    px,pw,qx,qw = computeMomentum(dirbase, lettercode)
    cA,cT,cS,dimEff,single,sigmaRKHS,s,m = computeParameters(dirbase,lettercode)
    shooting = xmodmap.deformation.Shooting(sigmaRKHS, qx, cA=cA, cS=cS,  cT=cT, dimEff=dimEff, single=single)
    px1,pw1,qx1,qw1 = shooting(px, pw, qx, qw)[-1] # get Deformed Atlas
    
    nu_S = qw1*zeta_S
    nu_S2 = None
    if zeta_S2 is not None:
        nu_S2 = qw1*zeta_S2
    qx1_resize = resizeData(qx1,s,m)
    
    summary = {
        "D": qx1_resize,
        "nu_D": nu_S,
        "nu_D2": nu_S2
    }
    torch.save(summary,os.path.join(dirsave,lettercode + "_meanAtlas.pt"))
    params = {
        "px": px, #momentum in position space
        "pw": pw, #momentum in weight space
        "cA": cA, #affine transformation weight
        "cT": cT, #target deformation weight
        "cS": cS, #source deformation weight
        "dimEff": dimEff,
        "single": single,
        "sigmaRKHS": sigmaRKHS,
        "s": s,
        "m": m,
    }
        
    imageVals = [
        np.sum(nu_S.detach().cpu().numpy(),axis=-1),
        np.argmax(nu_S.detach().cpu().numpy(),axis=-1)+1,
        np.argmax(nu_S2.detach().cpu().numpy(),axis=-1)+1
    ]
    imageNames = [
        "Weight_AtlasFeatures",
        "MaxVal_AtlasFeatures",
        "MaxsubVal_AtlasFeatures"
    ]
    writeVTK(qx1_resize, imageVals, imageNames, os.path.join(dirsave,lettercode + "_meanAtlas.vtk"))
    return

def shootColumns(dirbase, lettercode, dirsave, colS):
    # first compute average momentum and then shoot colS (N x 3) with shooting grid 
    # will then need to reshoot with each of the individual ERCTEC deformations per subject
    px,pw,qx,qw = computeMomentum(dirbase, lettercode)
    cA,cT,cS,dimEff,single,sigmaRKHS,s,m = computeParameters(dirbase,lettercode)
    '''
    # This works: we get same atlas as shooting in template 
    shooting = xmodmap.deformation.Shooting(sigmaRKHS, qx, cA=cA, cS=cS,  cT=cT, dimEff=dimEff, single=single)
    px1,pw1,qx1,qw1 = shooting(px, pw, qx, qw)[-1] # get Deformed Atlas
    qx1_resize = resizeData(qx1,s,m)
    writeVTK(qx1_resize,[qw1.detach().cpu().numpy()],['Weights'],os.path.join(dirsave,lettercode+"_meanAtlas2inCols.vtk"))
    '''
    shootinggrid = xmodmap.deformation.ShootingGrid(sigmaRKHS, torch.clone(qx).detach().requires_grad_(True), cA=cA, cS=cS,  cT=cT, dimEff=dimEff, single=single)
    colS_tilde = rescaleDataSM(colS,s,m)
    wc = torch.ones((colS_tilde.shape[0],1))*0.0004
    px1,pw1,qx1,qw1,colS1,wc1 = shootinggrid(px, pw, qx, qw,colS_tilde,wc)[-1] # get Deformed columns subject specific
    #px1,pw1,qx1,qw1,colS1,wc1 = shootinggrid(torch.clone(px).detach().requires_grad_(True), torch.clone(pw).detach().requires_grad_(True), torch.clone(qx).detach().requires_grad_(True), torch.clone(qw).detach().requires_grad_(True),torch.clone(qx).detach().requires_grad_(True),torch.clone(qw).detach().requires_grad_(True))[-1] # get Deformed columns subject specific

    colS1_resize = resizeData(colS1,s,m)
    summary = {
        "D": colS1_resize,
        "nu_D": wc1
    }
    torch.save(summary,os.path.join(dirsave,lettercode+"_meanAtlas_columns.pt"))
    writeVTK(colS1_resize,[wc1],['Weights'],os.path.join(dirsave,lettercode+"_meanAtlas_columns.vtk"))
    qx1_resize = resizeData(qx1,s,m)
    writeVTK(qx1_resize,[qw1.detach().cpu().numpy()],['Weights'],os.path.join(dirsave,lettercode+"_meanAtlas2inCols.vtk"))

    return 

def moveTarget(diroutput,lettercode,dirtarget,dirsave):
    paramFiles = glob.glob(diroutput + lettercode + "*/sm.pt")
    dirT = dirtarget + lettercode + '/' 
    for pf in paramFiles:
        params = torch.load(pf)
        comT = params['comT'].cpu().numpy() #Center of Mass of the target data
        letterDate = pf.split('/')[-2]
        fil = dirT + letterDate + '.npz'
        fInfo = np.load(fil)
        X = fInfo[fInfo.files[0]] - comT
        np.savez(dirsave + letterDate + '.npz',X=X,nu_R=fInfo[fInfo.files[1]],nu_S=fInfo[fInfo.files[2]])
    return
if __name__ == "__main__":
    dirbase = "Atlas_ShiftedAll/"
    dirsave = "Atlas_ShiftedAll/Atlases/"
    dirsave2 = "Atlas_ShiftedAll/TargetsShifted/"
    os.makedirs(dirsave,exist_ok=True)
    os.makedirs(dirsave2,exist_ok=True)
    dirtarget = ''
    b =  'LF_example.npz'
    S,nu_S = getFromFile(b,featIndex=1)
    _,nu_S2 = getFromFile(b,featIndex=2)
    zeta_S = nu_S / torch.sum(nu_S,axis=-1)[...,None]
    zeta_S2 = nu_S2 / torch.sum(nu_S2,axis=-1)[...,None]
    
    shootTemplate(dirbase,"example",zeta_S,dirsave,zeta_S2)
    moveTarget(dirbase,"example",dirtarget,dirsave2)