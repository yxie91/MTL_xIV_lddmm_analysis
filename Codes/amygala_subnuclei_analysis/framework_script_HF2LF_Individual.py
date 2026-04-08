'''
Author: Kaitlin Stouffer
Purpose:
    - single modality mapping (high field segmentations to low field segmentations)
    - no boundary estimation or density support weights used 
    - parameters are only rigid (A + tau + scale (isotropic)) and diffeomorphism
    - each structure mapped independently
    
Use:
    python3  HF2LF_individual aFileFull aFile tFileFull tFile savedir
'''
import os
import sys
from sys import path as sys_path
import scipy.io as sio
sys_path.append("..")
sys_path.append("../..")
import glob

import torch
import xmodmap
import scipy as sp
import numpy as np
from matplotlib import pyplot as plt

from xmodmap.preprocess.preprocess import resizeData
from xmodmap.io.getOutput import writeParticleVTK, writeVTK, getJacobian, getEntropy
from xmodmap.io.getInput import getFromFile

def saveAtlas(qx1,qw1,zeta_S,s,m,nu_S,zeta_S2,savedir):
    # rescale state
    D = resizeData(qx1,s,m)
    
    # compute geometrically deformed only
    nu_D = (torch.squeeze(qw1)[...,None])*zeta_S
    nu_D2= (torch.squeeze(qw1)[...,None])*zeta_S2
    jac = getJacobian(D,nu_S,nu_D)
    
    writeParticleVTK(D, nu_D, os.path.join(savedir,"atlas_nu_D.vtk"), norm=True, condense=False, featNames=None, sW=None)
    summary = {
        "D": D, # Deformed Atlas Particle Location
        "nu_D": nu_D, # Deformed Atalas Feature Density
        "nu_D2":nu_D2, # Deformed Atalas Feature Density in sub regions
        "jac": jac # Jacobian determinant (deformation metric)
    }
    torch.save(summary,os.path.join(savedir,"atlas_deformationSummary.pt"))
    imageVals = [
        np.sum(nu_D.cpu().numpy(),axis=-1),
        np.argmax(nu_D.cpu().numpy(),axis=-1)+1,
        np.argmax(nu_D2.cpu().numpy(),axis=-1)+1,
        np.squeeze(jac.cpu().numpy()),
    ]
    imageNames = [
        "Weight_AtlasFeatures",
        "MaxVal_AtlasFeatures",
        "MaxVal_AtlasFeatures_Subregion",
        "Jacobian",
    ]
    writeVTK(D, imageVals, imageNames, os.path.join(savedir,"atlas_deformationSummary.vtk"))
    return
                   
def saveTarget(Td, w_Td, zeta_T,w_T,s,m,suffix="",savedir=""):
    # rescale state
    Td = resizeData(Td,s,m)
    
    # compute deformed features
    nu_Td = torch.squeeze(w_Td)[...,None]*zeta_T
    
    jac = getJacobian(Td,w_T*zeta_T,nu_Td)
    
    writeParticleVTK(Td, nu_Td, os.path.join(savedir,"target_nu_Td" + suffix + ".vtk"), norm=True, condense=False, featNames=None, sW=None)
    summary = {
        "Td": Td,
        "nu_Td": nu_Td,
        "jac": jac
    }
    torch.save(summary,os.path.join(savedir,"target_deformationSummary" + suffix + ".pt"))
    imageVals = [
        np.sum(nu_Td.cpu().numpy(),axis=-1),
        np.argmax(nu_Td.cpu().numpy(),axis=-1)+1,
        np.squeeze(jac.cpu().numpy()),
        np.squeeze(getEntropy(nu_Td.cpu().numpy())),
    ]
    imageNames = [
        "Weight_TargetFeatures",
        "MaxVal_TargetFeatures",
        "Jacobian",
        "Entropy"
    ]
    writeVTK(Td, imageVals, imageNames, os.path.join(savedir,"target_deformationSummary" + suffix + ".vtk"))
    return

def saveOriginal(S,nu_S,T,nu_T,savedir):
    print("saving original with sizes")
    wS = torch.sum(nu_S,axis=-1)
    wT = torch.sum(nu_T,axis=-1)
    print(S.shape)
    print(nu_S.shape)
    print(T.shape)
    print(nu_T.shape)
    maxS = torch.argmax(nu_S,axis=-1)+1.0
    maxT = torch.argmax(nu_T,axis=-1)+1.0
    
    writeVTK(S,[wS.cpu().numpy(),maxS.cpu().numpy()],['Weight','Max_Region'],os.path.join(savedir,"originalAtlas.vtk"))
    writeVTK(T,[wT.cpu().numpy(),maxT.cpu().numpy()],['Weight','Max_Region'],os.path.join(savedir,"originalTarget.vtk"))
    return
def HF2LF_individual(aFileFull, aFile, tFileFull, tFile, savedir):
    # set random seed
    torch.manual_seed(0)
    torch.set_printoptions(precision=6)


    print(savedir)
    os.makedirs(savedir, exist_ok=True)
    print(torch.get_default_dtype)

    d = 3
    dimEff = 3
    sigmaRKHS = [0.2, 0.1,0.05]
    sigmaVar = [0.2, 0.05, 0.02]
    steps = 250
    beta = None
    res = 1.0
    kScale = torch.tensor(1.)
    extra = ""
    cA = 1.0
    cT = 1.0  # original is 0.5
    cS = 5.0
    single = True
    gamma = 0.01

    S,nu_S = getFromFile(aFile,featIndex=1)
    _,nu_S2 = getFromFile(aFile,featIndex=2)
    T,nu_T = getFromFile(tFile,featIndex=1)
    fullS,fullnu_S = getFromFile(aFileFull,featIndex=1)
    fullT,fullnu_T = getFromFile(tFileFull,featIndex=1)
    fullw_T = fullnu_T.sum(axis=-1)[...,None]
    fullzeta_T = fullnu_T / fullw_T
        
    labs = nu_T.shape[-1]  # in target
    labS = nu_S.shape[-1]  # template
    '''
    A180 = torch.eye(3)
    A180[0,0] = -1.0
    A180[1,1] = -1.0
    S = S@A180

    fullS = fullS@A180
    '''
    # center template and target around center of mass (challenging for reshooting though)
    comT = torch.mean(T,axis=0)
    comT = comT*0.0
    T = T - comT
    fullT = fullT - comT
    comS = torch.mean(S,axis=0)
    comS = comS*0.0
    S = S - comS
    fullS = fullS - comS

    #saveOriginal(fullS,fullnu_S,fullT,fullnu_T,savedir)
    saveOriginal(S,nu_S,T,nu_T,savedir)
    w_S2 = nu_S2.sum(axis=-1)[..., None]
    zeta_S2 = nu_S2 / torch.sum(nu_S2,axis=-1)[...,None]
    zeta_S2[torch.squeeze(w_S2 == 0), ...] = 0.0
    from xmodmap.preprocess.makePQ_legacy import makePQ
    (
        w_S,
        w_T,
        zeta_S,
        zeta_T,
        q0,
        p0,
        numS,
        Stilde,
        Ttilde,
        s,
        m,
        pi_STinit,
        lamb0,
    ) = makePQ(S, nu_S, T, nu_T)

    sm = {
        "s": s,
        "m": m,
        "cA": cA,
        "cT": cT,
        "cS": cS,
        "sigmaRKHS": sigmaRKHS,
        "d": d,
        "dimEff": dimEff,
        "single": single,
        "comT": comT,
        "comS": comS,
        "gamma": gamma,
        "steps": steps
    }

    torch.save(sm,os.path.join(savedir,"sm.pt"))
    fullTtilde = (fullT - m) * (1.0 / s)

    # Model Setup

    ## Varifold distance without boundaries
    dataloss = xmodmap.distance.LossVarifoldNorm(sigmaVar, w_T, zeta_T, Ttilde)   # Dream of : (sigmaVar, T, nu_T)
    dataloss.normalize_across_scale(Stilde, w_S, zeta_S, torch.eye(zeta_S.shape[-1])) # single modality with pi as identity 
    dataloss.weight = 1.

    ## non-rigid and affine deformations
    hamiltonian = xmodmap.deformation.Hamiltonian(sigmaRKHS, Stilde, cA=cA, cS=cS,  cT=cT, dimEff=dimEff, single=single)
    hamiltonian.weight = gamma
    shooting = xmodmap.deformation.Shooting(sigmaRKHS, Stilde, cA=cA, cS=cS,  cT=cT, dimEff=dimEff, single=single)


    # Optimization

    variable_init = {
        "px": torch.zeros_like(Stilde).requires_grad_(True),
        "pw": torch.zeros_like(w_S).requires_grad_(True),
        "qx": Stilde.clone().detach().requires_grad_(True),
        "qw": w_S.clone().detach().requires_grad_(True),
        "zeta_S": zeta_S,
    }

    variable_to_optimize = ["px", "pw"]

    precond = {
        "px": torch.rsqrt(kScale),
        "pw": torch.rsqrt(kScale) / dimEff / w_S,
    }

    loss = xmodmap.model.SingleModality(hamiltonian, shooting, dataloss)
    loss.init(variable_init, variable_to_optimize, precond=precond, savedir=savedir)
    #loss.resume(variable_init, os.path.join(savedir, 'checkpoint.pt'))
    loss.optimize(steps)

    # Saving
    precondVar = loss.get_variables_optimized()
    torch.save(precondVar,os.path.join(savedir,"precondVar.pt"))

    px1,pw1,qx1,qw1 = shooting(precondVar['px'], precondVar['pw'], precondVar['qx'], precondVar['qw'])[-1] # get Deformed Atlas


    shootingBack = xmodmap.deformation.ShootingBackwards(sigmaRKHS,Stilde,cA=cA,cS=cS,cT=cT, dimEff=dimEff, single=single)
    _,_,_,_,Td,wTd = shootingBack(px1, pw1, qx1, qw1, Ttilde, w_T)[-1] # get Deformed Target


    saveAtlas(qx1.detach(),qw1.detach(),zeta_S,s,m,nu_S,zeta_S2,savedir)
    saveTarget(Td.detach(),wTd.detach(),zeta_T,w_T,s,m,savedir=savedir)

    # shoot whole target back to confirm rotation alignment 
    _,_,_,_,fullTd,fullwTd = shootingBack(px1, pw1, qx1, qw1, fullTtilde, fullw_T)[-1]
    saveTarget(fullTd.detach(),fullwTd.detach(),fullzeta_T,fullw_T,s,m,suffix="_full",savedir=savedir)

    f = loss.print_log()
    f.savefig(os.path.join(savedir,"loss.png"),dpi=300)
    f = loss.print_log(logScale=True)
    f.savefig(os.path.join(savedir,"logloss.png"),dpi=300)
    return
if __name__ == '__main__':

    aFileFull = "Atlas_ShiftedAll/Atlases/example_meanAtlas.pt"
    aFile = "Atlas_ShiftedAll/Atlases/example_amygdala.npz"

    tFileFull = "Atlas_ShiftedAll/TargetsShifted/example.npz"
    tFile = "Atlas_ShiftedAll/TargetsShifted/example_amygdala.npz"
    
    savedir = "SubjectAtlas_Params2/example/amygdala"
    HF2LF_individual(aFileFull,aFile,tFileFull,tFile,savedir)