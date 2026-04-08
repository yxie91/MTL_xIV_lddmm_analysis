from pykeops.torch import Vi,Vj
from matplotlib import pyplot as plt
import numpy as np
import torch
import sys
sys.path.append("..")
sys.path.append("../..")
import os
import xmodmap

#sys.path.append("../xmodmap/")
#sys.path.append("../xmodmap/io/")
#import getOutput as gO
import xmodmap.io.getOutput as gO

def removeOutliers(S,thresh=0.500,zMin=None,zMax=0.5,K=5):
    '''
    remove points in S if they are above threshold away from 5 NN.
    
    Example on whole brain: thresh=0.5,zMin=None,zMax=0.5,K=5
    Example on half brain: thresh=0.5,zMin=,zMax=,K=5
    
    # take points based on zO first then on outs
    '''
    Snew = torch.clone(S)
    zO = torch.ones((Snew.shape[0])).type(torch.bool)
    if zMin is not None:
        x = torch.squeeze(Snew[:,-1] > zMin)
        zO = zO*x
    if zMax is not None:
        x = torch.squeeze(Snew[:,-1] < zMax)
        zO = zO*x
    
    Snew = Snew[zO,:]
    Si = Vi(Snew)
    Sj = Vj(Snew)
    D = Si.sqdist(Sj)
    Kmin = D.Kmin(K,dim=1)
    Kmin = torch.sqrt(Kmin)
    print("max of Kmin: ", torch.max(Kmin))
    Kmin = Kmin.sum(dim=1)
    print(Kmin)
    print("min of Kmin sum: ", torch.min(Kmin))
    outs = Kmin < thresh
    return zO, torch.squeeze(outs)
    
    
def computeSetDistance(S,T):
    '''
    Computes distance of each particle in D to its nearest neighbor in T.
    
    Args:
        S: coordinates of deformed source (or source)
        T: coordinates of target 
        
    Returns:
        list of distances equal to number of particles in D
        
    '''
    print("min and max of coordinates in computeSetDistance")
    print(torch.min(S,axis=0))
    print(torch.max(S,axis=0))
    print(torch.min(T,axis=0))
    print(torch.max(T,axis=0))
    D_i = Vi(S)
    T_j = Vj(T)
    
    D_ij = ((D_i - T_j) ** 2).sum(-1)  # symbolic matrix of squared distances
    nnD = D_ij.min(1, dim=1).sqrt()  # get nearest neighbor distance for each of S points
    
    return nnD

def plotBeforeAndAfterHistogram(distB, distA ,outputName, scaleA=False, weightsB=None, weightsA=None):
    '''
    Plots histogram of distances and computes mean and standard deviation.
    '''
    
    dB = distB.cpu().numpy()
    dA = distA.cpu().numpy()
    
    mi = min(np.min(dB),np.min(dA))
    ma = max(np.max(dB),np.max(dA))
    ma = max(np.quantile(dB,0.95),np.quantile(dA,0.95))
    if scaleA:
        ma = np.quantile(dA,0.95)
    
    mB = (np.mean(dB)).round(decimals=4)
    sB = (np.std(dB)).round(decimals=4)
    mA = (np.mean(dA)).round(decimals=4)
    sA = (np.std(dA)).round(decimals=4)
    meB = (np.median(dB)).round(decimals=4)
    meA = (np.median(dA)).round(decimals=4)
    tA = len(distA)
    tB = len(distB)
    
    labB = f'Mean {mB:.4f}, Std {sB:.4f}, Median: {meB:.4f}, TotalNum {tA}'
    labA = f'Mean {mA:.4f}, Std {sA:.4f}, Median: {meA:.4f}, Total Num {tB}'
    
    fig,ax = plt.subplots(2,1)
    hB,bB,_ = ax[0].hist(dB,range=(mi,ma),bins=25,label=labB,weights=weightsB)
    hA,bA,_ = ax[1].hist(dA,range=(mi,ma),bins=25,label=labA,weights=weightsA)
    bs = bB[-1] - bB[-2]
    ax[0].set_title("Distances Before Deformation")
    ax[1].set_title("Distances After Deformation")
    ax[0].set_xlabel(f"Distance (mm), bin size {bs}")
    ax[1].set_xlabel(f"Distance (mm), bin size {bs}")
    ax[0].legend()
    ax[1].legend()
    fig.savefig(outputName,dpi=300)
    print("hB: ", hB)
    print("hA: ", hA)
    print("hB norm: ", hB/np.sum(hB))
    print("hA norm: ", hA/np.sum(hA))
    
    fig,ax = plt.subplots(2,1)
    ax[0].hist(bB[:-1], bB, weights=hB/np.sum(hB),label=labB)
    ax[1].hist(bA[:-1], bA, weights=hA/np.sum(hA), label=labA)
    ax[0].set_title("Distances Before Deformation")
    ax[1].set_title("Distances After Deformation")
    ax[0].set_xlabel(f"Distance (mm), bin size {bs:.4f}")
    ax[1].set_xlabel(f"Distance (mm), bin size {bs:.4f}")
    ax[0].legend()
    ax[1].legend()
    fig.savefig(outputName.replace('.png','_normalized.png'),dpi=300)
    
    return

def plotSingle(distBList, titleList, outputName,binRange=None,binSize=0.01):
    '''
    Plots histogram of distances and computes mean and standard deviation.
    titleList should indicate names for all of distances (i.e. cell markers)
    '''
    
    fig,ax = plt.subplots(figsize=(12,3))
    fig0,ax0 = plt.subplots(figsize=(12,3))
    i = 0
    if binRange is None:
        mi = 10000.0
        ma = -100.0
        for distB in distBList:
            dB = distB.cpu().numpy()
            mi = min(np.min(dB),mi)
            ma = max(np.quantile(dB,0.95),ma)
            
    else:
        mi = binRange[0]
        ma = binRange[1]
    
    bins = np.arange(mi,ma,binSize)
    weights = []
    x = []
    labs = []
    for i in range(len(distBList)):
        dB = distBList[i].cpu().numpy()
        mB = (np.mean(dB)).round(decimals=4)
        sB = (np.std(dB)).round(decimals=4)
        meB = (np.median(dB)).round(decimals=4)
        tB = len(dB)
    
        labB = f'{titleList[i]}: Mean {mB:.3f}, Std {sB:.3f}, Median: {meB:.3f}, TotalNum {tB}'
        
        hB,bB,_ = ax0.hist(dB,bins=bins,label=labB)
        bs = bB[-1] - bB[-2]
        x.append(bB[:-1])
        weights.append(hB/np.sum(hB))
        labs.append(labB)
        ax.hist(bB[:-1], bB, weights=hB/np.sum(hB),label=labB,alpha=0.5)
    
    ax.set_title("Distances")
    ax.set_xlabel(f"Distance (mm)")
    ax.set_ylim(0,0.5)
    ax.set_ylabel('Percent of Cell Markers')
    ax.legend(fontsize='small')
    fig.savefig(outputName,dpi=300)
    f,ax = plt.subplots()
    ax.hist(x,bins,weights=weights,label=labs)
    ax.legend(fontsize='small')
    ax.set_ylim(0,0.5)
    ax.set_ylabel('Percent of Cell Markers')
    ax.set_title("Distances")
    ax.set_xlabel(f"Distance (mm)")
    f.savefig(outputName.replace('.png','_together.png'),dpi=300)
    
    
    return



def computeSetDistancesSubset(S,nu_S,featS,T,nu_T,featT,writeOut=True,plotNames=None):
    '''
    Computes set distance for each subset of source (S) and target (T) 
    based on associated feature values. For particles with distributed mass, weighted 
    
    Args:
        S = source locations (or deformed source locations)
        nu_S = full set of normalized feature values for source
        featS = list of feature dimensions
        T = target locations (eg Allen Atlas)
        nu_T = full set of feature values for target
        featT = list of feature dimensions
    
    Returns:
        min distance of each particle in set S to particle in set T
    '''
    
    assert len(featS) == len(featT), "Feature Associations Do Not Match: " + str(len(featS)) + " vs. " + str(len(featT))
    
    distances = []
    weights = []
    sTotalSet = False
    
    for i in range(len(featS)):

        featSi = featS[i]
        featTi = featT[i]
        
        inds = nu_S[:,featSi[0]] > 0
        sS = S[inds,:]
        snu_S = nu_S[inds,featSi[0]]
        
        # select subset in S
        for s in range(1,len(featSi)):
            inds = nu_S[:,featSi[s]] > 0
            sS = torch.cat((sS,S[inds,:]))
            snu_S = torch.cat((snu_S,nu_S[inds,featSi[s]]))
        
        # one Hot memory saving 
        if nu_T.shape[-1] == nu_T.shape[0] or nu_T.shape[-1] == 1:
            print("not one hot")
            inds = nu_T == featTi[0]
            sT = T[torch.squeeze(inds),:]
            
            for s in range(1,len(featTi)):
                inds = nu_T == featTi[s]
                sT = torch.cat((sT,T[torch.squeeze(inds),:]))
            snu_T = torch.ones((sT.shape[0],1))
        else:    
            inds = nu_T[:,featTi[0]] > 0
            sT = T[inds,:]
            snu_T = nu_T[inds,featTi[0]]
            # select subset in T
            for s in range(1,len(featTi)):
                inds = nu_T[:,featTi[s]] > 0
                sT = torch.cat((sT,T[inds,:]))
                snu_T = torch.cat((snu_T,nu_T[inds,featTi[s]]))
        if sS.shape[0] < 1 or sT.shape[0] < 1:
            print("not enough points")
            weights.append([torch.tensor(0.0)])
            distances.append([torch.tensor(0.0)])
            continue
        distances.append(computeSetDistance(sS,sT))
        weights.append(snu_S)
        if (writeOut):
            if plotNames is not None:
                t = torch.zeros((sS.shape[0]+sT.shape[0],1))
                t[0:sS.shape[0]] = 1.0
                gO.writeVTK(torch.cat((sS,sT)),[t],["Source"],plotNames[i] + '.vtk',polyData=None)
        
            if i == 0 or sTotalSet == False:
                Stotal = sS
                Ttotal = sT
                Sf = torch.zeros((sS.shape[0],1))+i
                Tf = torch.zeros((sT.shape[0],1))+i
                sTotalSet = True
            else:
                Stotal = torch.cat((Stotal,sS))
                Ttotal = torch.cat((Ttotal,sT))
                Sf = torch.cat((Sf,torch.zeros((sS.shape[0],1))+i))
                Tf = torch.cat((Tf,torch.zeros((sT.shape[0],1))+i))
    if (writeOut):
        gO.writeVTK(Stotal,[Sf],['cellMarkers'],plotNames[-1]+'_source.vtk',polyData=None)
        gO.writeVTK(Ttotal,[Tf],['cellMarkers'],plotNames[-1]+'_target.vtk',polyData=None)
        
    return distances,weights

def plotPi_ST(featS,precondFile,savedir,titles, featNames=None, norm=True):
    '''
    Plot histograms of values in PI_ST (normalized) for each of regions. Compiles sum of mass for each of the 
    regions considered to be the same.
    
    featNames should be list of feature names in target space 
    '''
    info = torch.load(precondFile)
    pi_ST = info['pi_ST']**2
    numTfeats = pi_ST.shape[-1]
    
    for i in range(len(featS)):
        feats = featS[i]
        prob = torch.zeros((1,numTfeats))
        for f in feats:
            prob += pi_ST[f,:]
        
        if norm:
            prob = prob / torch.sum(prob,axis=-1)
        print(prob.shape)
        print(prob)
        prob = torch.squeeze(prob)
        f,ax = plt.subplots()
        ax.bar(torch.arange(numTfeats).detach().cpu().numpy(),prob.detach().cpu().numpy())
        ax.set_title(titles[i] + " Estimated Distribution")
        if featNames is not None:
            ax.set_xticks(list(np.arange(numTfeats)))
            ax.set_xticklabels(featNames,fontsize=5,rotation='vertical')
        f.savefig(os.path.join(savedir,titles[i] + '_pi_ST.png'),dpi=300)
    return

def plotPi_STCompare(featS,pi_STs,savedir,titles, legLabels, featNames=None, norm=True):
    '''
    Plot histograms of values in PI_ST (normalized) for each of regions. Compiles sum of mass for each of the 
    regions considered to be the same.
    
    featNames should be list of feature names in target space 
    '''
    numTfeats = pi_STs[0].shape[-1]
    
    for i in range(len(featS)):
        feats = featS[i]
        fig,ax = plt.subplots()
        for j in range(len(pi_STs)):
            prob = torch.zeros((1,numTfeats))
            for f in feats:
                prob += pi_STs[j][f,:]
        
            if norm:
                prob = prob / torch.sum(prob,axis=-1)

            prob = torch.squeeze(prob)

            ax.bar(torch.arange(numTfeats).detach().cpu().numpy() + (j-1)*0.2,prob.detach().cpu().numpy(),label=legLabels[j],alpha=0.5,width=0.2)
            ax.set_title(titles[i] + " Estimated Distribution")
            
            if featNames is not None:
                ax.set_xticks(list(np.arange(numTfeats)))
                ax.set_xticklabels(featNames,fontsize=5,rotation='vertical')
            ax.legend()
            fig.savefig(os.path.join(savedir,titles[i] + '_pi_ST.png'),dpi=300)
    return

def plotPiSTMatrix(precondVarFile,normalize=False,vmin=None,vmax=None):
    '''
    pi_ST = distribution of size Slabels x Tlabels (assume torch object)
    '''
    p = torch.load(precondVarFile)
    zeta_S = p['zeta_S']
    inds = (torch.sum(zeta_S,axis=0) > 0).detach().cpu().numpy()
    pi_ST = (p['pi_ST']**2).detach().cpu().numpy()
    pi_STsub = pi_ST[inds,...]
    
    if normalize:
        pi_STsub = pi_STsub / np.sum(pi_STsub,axis=-1)[...,None]
        nstring = 'norm'
    else:
        nstring = ''
    
    f,ax = plt.subplots()
    #plt.axis('square')
    im = ax.imshow(pi_STsub,vmin=vmin,vmax=vmax)
    ax.set_aspect(1./ax.get_data_ratio(),adjustable='box')
    f.colorbar(im,ax=ax)
    f.savefig(precondVarFile.replace('precondVar.pt','pi_ST' + nstring + '.png'),dpi=300)
    return


def confirmCellAssignment(G,nu_Ci,C,nu_I,outname):
    '''
    Confirm Cell Assignment per Gene is the same as expected (e.g. coordinate systems align).
    
    Args:
        G = coordinates of genes
        nu_Ci = 0-based index of cell assignment for each gene 
        C = coordinates of cells
        nu_I = 0-based index of cell assignment for each cell
    Returns:
        Number of mismatches 
    '''
    
    gO.writeParticleVTK(G,torch.squeeze(nu_Ci),outname + 'genes.vtk')
    gO.writeParticleVTK(C,torch.squeeze(nu_I),outname + 'cells.vtk')
    
    D_i = Vi(G)
    T_j = Vj(C)
    
    D_ij = ((D_i - T_j) ** 2).sum(-1)  # symbolic matrix of squared distances
    nnIndex = D_ij.argKmin(1, dim=1)  # get nearest neighbor index 
    nnCi = nu_I[nnIndex.type(torch.LongTensor)]

    totMis = torch.squeeze(nu_Ci) != torch.squeeze(nnCi)
    print(f'Total number of mismatches is {torch.sum(totMis)}')
    print(f'Percent missed is {100.0*torch.sum(totMis)/totMis.shape[0]}')
    
    Gmissed = G[torch.squeeze(totMis),:] # coordinates of genes that missed
    Cids_missed = nu_Ci[torch.squeeze(totMis)] # true IDs assigned 
    Cids_missedFalse = nnCi[torch.squeeze(totMis)]
    Cfalse = C[nnIndex.type(torch.LongTensor)]
    Cfalse = Cfalse[torch.squeeze(totMis),:]
    Ctrue = torch.zeros((Gmissed.shape[0],3))
    for i in range(Ctrue.shape[0]):
        Ctrue[i,:] = C[torch.squeeze(nu_I == Cids_missed[i]),:]
      
    '''
    D_i = Vi(Gmissed.type(torch.FloatTensor))
    T_j = Vj(Ctrue.type(torch.FloatTensor))
    D_ij = ((D_i - T_j) ** 2).sum(-1)
    print(D_ij.shape)
    nnD = D_ij.min(1,dim=1).sqrt()
    '''
    
    nnD = torch.sqrt(((torch.squeeze(Gmissed)-torch.squeeze(Ctrue))**2).sum(dim=-1))
    
    f,ax = plt.subplots()
    ax.hist(torch.squeeze(nnD).cpu().numpy(),density=True)
    ax.set_title('Distance to correctly assigned cell (mm)')
    f.savefig(outname + '_true.png',dpi=300)
    
    '''
    T_j = Vj(Cfalse.type(torch.FloatTensor))
    D_ij = ((D_i - T_j) ** 2 ).sum(-1)
    nnD = D_ij.min(1,dim=1).sqrt()
    '''
    nnD = torch.sqrt(((torch.squeeze(Gmissed)-torch.squeeze(Cfalse))**2).sum(dim=-1))


    f,ax = plt.subplots()
    ax.hist(torch.squeeze(nnD).cpu().numpy(),density=True)
    ax.set_title('Distance to closest assigned cell (mm)')
    f.savefig(outname + '_closest.png',dpi=300)
    
    feats = torch.zeros((Gmissed.shape[0],2))
    feats[:,0] = torch.squeeze(Cids_missed)
    feats[:,1] = torch.squeeze(Cids_missedFalse)
    gO.writeParticleVTK(Gmissed,feats,outname + 'misassignedGenes.vtk',featNames=['True_Assign','False_Assign'])
    return

def KLdivergence(p1,p2):
    '''
    Compute the KL divergence between probability distribution 1 and probability distribution 2
    Here, p1 and p2 are assumed to be distributions over the same discrete space
    
    if p1,p2 have dimensions of 2, then the distributions are assumed to be the last dimension
    '''
    
    if torch.is_tensor(p1):
        p1n = p1.detach().cpu().numpy()
    else:
        p1n = p1
    if torch.is_tensor(p2):
        p2n = p2.detach().cpu().numpy()
    else:
        p2n = p2
        
    if p1n.shape[0] != p2n.shape[0]:
        p2n = p2n[None,...]
        print("p1n shape: ", p1n.shape)
        print("p2n shape: ", p2n.shape)
    
    # check for 0/0
    if np.min(p2n) == 0:
        p2n[p2n == 0] = 0.000001
        p2n = p2n / np.sum(p2n,axis=-1)[...,None]
    
    if np.min(p1n) == 0:
        p1n[p1n == 0] = 0.000001
        p1n = p1n / np.sum(p1n, axis=-1)[...,None]
    
    # check for needing to renormalize
    if np.sum(np.sum(p1n,axis=-1) != 1) > 0:
        p1n = p1n / np.sum(p1n, axis=-1)[...,None]
    
    if np.sum(np.sum(p2n,axis=-1) != 1) > 0:
        p2n = p2n / np.sum(p2n, axis=-1)[...,None]

    div = (p1n/p2n)
    k = p1n * np.log(div)
    kd = np.sum(k,axis=-1)
    
    return kd 
    
