from pykeops.torch import Vi,Vj

import torch
import numpy as np
import sys
sys.path.append('..')

from accuracyMeasures import KLdivergence
torch.set_default_dtype(torch.float32)



class EmpiricalDistributions():
    '''
    Compute empirical distribution of feature values or mass of data points at sample points.
    Example sample points could be grid points or atlas coordinates.
    '''
    def __init__(self, samplePoints, dataPoints, sampleFeatures, dataFeatures, sampleSupportRestriction=None):
        self.samplePoints = samplePoints
        self.dataPoints = dataPoints
        self.sampleFeatures = sampleFeatures
        self.dataFeatures = dataFeatures
        
        if sampleSupportRestriction is not None:
            self.sampleFeatures = self.sampleFeatures * torch.squeeze(sampleSupportRestriction)[...,None] 
            # remove those points with zeros
        
        wJ = torch.sum(self.sampleFeatures,axis=-1)
        inds = wJ > 0
        self.samplePoints = self.samplePoints[torch.squeeze(inds),...]
        self.sampleFeatures = self.sampleFeatures[torch.squeeze(inds),...]
        
    
    def computeWeightedDistribution(self, resampleFeatures):
        '''
        Computes weighted average of distribution for each feature type of sample points.
        Weights = 0 or 1 depending on indicator for particle being "in" region and in support of target given by sW (pulled back)
        '''
        pi_ST = torch.zeros((self.sampleFeatures.shape[-1],self.dataFeatures.shape[-1])) # estimated distributions
       
        wJ = torch.sum(self.sampleFeatures,axis=-1) # weights per particle
        
        for i in range(self.sampleFeatures.shape[-1]):
            inds = torch.squeeze(self.sampleFeatures[:,i])/torch.squeeze(wJ) > 0.5 # select points with majority of mass in region
            if torch.sum(inds) < 1:
                continue
            Z = torch.sum(inds*torch.squeeze(self.sampleFeatures[:,i]),axis=0) # total mass in region 
            pi_ST[i,:] = torch.sum(inds[...,None]*resampleFeatures,axis=0)/Z # total mass in each feature divided by total mass in region
        
        return pi_ST
    
    def computeTotalCounts(self,resampleFeatures):
        pi_ST = torch.zeros((self.sampleFeatures.shape[-1],self.dataFeatures.shape[-1]))
        wJ = torch.sum(self.sampleFeatures,axis=-1)
        for i in range(self.sampleFeatures.shape[-1]):
            inds = torch.squeeze(self.sampleFeatures[:,i])/torch.squeeze(wJ) > 0.5
            if torch.sum(inds) < 1:
                continue
            pi_ST[i,:] = torch.sum(inds[...,None]*resampleFeatures,axis=0) # total mass in each of target features for atlas points with majority in single region
        return pi_ST
    
    def computeVarianceWithinDistribution(self, resampleFeatures):
        '''
        Computes variance in probability distributions within regions of atlas.
        '''
        var_ST = torch.zeros((self.sampleFeatures.shape[-1],self.dataFeatures.shape[-1]))
        wJ = torch.sum(self.sampleFeatures,axis=-1)
        varParts = torch.zeros((resampleFeatures.shape[0],self.dataFeatures.shape[-1]))
        klParts = torch.zeros((resampleFeatures.shape[0],1))
        kl_ST = torch.zeros((self.sampleFeatures.shape[-1],1))
        totPartsUsed = 0
        regionsUsed = self.sampleFeatures.shape[-1]
        areaUsed = torch.zeros((self.sampleFeatures.shape[-1],1))
        
        for i in range(self.sampleFeatures.shape[-1]):
            # take only particles in atlas that have at least half of mass in 1 region (e.g. no boundary particles)
            inds = torch.squeeze(self.sampleFeatures[:,i]) / torch.squeeze(wJ) > 0.5
            if torch.sum(inds) < 1:
                regionsUsed -= 1
                continue
            else:
                totPartsUsed += torch.sum(inds)
            totInd = torch.arange(self.sampleFeatures.shape[0])
            totInd = totInd[inds]
            wJsub = wJ[inds]
            sub = resampleFeatures[inds,...] / torch.sum(resampleFeatures[inds,...],axis=-1)[...,None] # each probability distribution

            if sub.shape[0] < 2:
                continue
            isZero = torch.sum(resampleFeatures[inds,...],axis=-1) > 0
            totInd = totInd[isZero]
            wJsub = wJsub[isZero]
            if torch.sum(isZero) != sub.shape[0]:
                if (torch.sum(isZero) == 0):
                    continue
                sub = sub[isZero,...]
            
            m = torch.sum(sub,axis=0) / sub.shape[0] # mean mass per target feature across particles in atlas region
            
            var_ST[i,:] = torch.sum((sub-m[None,...])**2,axis=0)/(sub.shape[0] - 1) # variance in the mean distribution across particles
            varParts[totInd,...] = (sub-m[None,...])**2 # distance squared of actual particle distribution from mean
            
            kll = (KLdivergence(sub,m))
            kl_ST[i] = torch.tensor(np.mean(kll))
            klParts[totInd,...] = torch.tensor(kll)[...,None]
            print("total parts used: ", totPartsUsed)
            areaUsed[i] = torch.sum(wJsub)
            print("total mass used: ", torch.sum(wJsub))
        print("average particles per region is: " + str(totPartsUsed/regionsUsed))
        
        return var_ST, varParts, kl_ST, klParts, areaUsed
    
    def computeVarianceBetweenDistribution(self, resampleFeaturesList):
        '''
        Compute variance between different resampled distributions.
        To DO: just look at pi_ST and compute variance across pi_ST's?
        '''
            
        return
        
    def NNAssign(self,returnRev=False):
        '''
        Assign dataPoints to nearest neighbor in sample points.
        Compute Mass of feature values of data points at all sample points.
        
        Returns total mass of data features at each of sample points.
        '''
        
        D_i = Vi(self.dataPoints)
        T_j = Vj(self.samplePoints)
    
        D_ij = ((D_i - T_j) ** 2).sum(-1)  # symbolic matrix of squared distances
        
        if not returnRev:
            nnD = D_ij.argKmin(1, dim=1)  # get nearest neighbor index for each of S points
            resampleFeatures = torch.zeros((self.samplePoints.shape[0],self.dataFeatures.shape[-1]))
            nnDadd = torch.cat((torch.squeeze(nnD),torch.arange(self.samplePoints.shape[0])))
            for i in range(self.dataFeatures.shape[-1]):
                resampleFeatures[:,i] = torch.bincount(nnDadd,weights=torch.cat((self.dataFeatures[:,i],torch.ones(self.samplePoints.shape[0])))) - 1
        else:
            print(D_ij.shape)
            nnD = D_ij.argKmin(1,dim=1) # get index of sample point that is closest
            print(nnD.shape)
            print(self.dataPoints.shape)
            resampleFeatures = torch.zeros((self.dataPoints.shape[0],self.sampleFeatures.shape[-1]))
            resampleFeatures[torch.arange(self.dataPoints.shape[0]),:] = self.sampleFeatures[torch.squeeze(nnD),:]
        return resampleFeatures

    
    def GaussianAssign(self,sigma):
        '''
        Spread mass evenly of dataPoints with a Gaussian kernel of bandwith sigma (mm).
        
        Returns total mass of data features at each of sample points.
        '''
        
        D_i = Vi(self.dataPoints)
        T_j = Vj(self.samplePoints)
        
        D_ij = ((D_i - T_j) ** 2).sum(-1)
        K = (-D_ij / (2.0 * sigma * sigma)).exp() # unnormalized weights (Data x Sample)
        Knorm = K.sum(dim=1) # normalize for conservation of mass
        Knorm[torch.squeeze(Knorm > 0),...] = 1.0/Knorm[torch.squeeze(Knorm > 0),...]
        resampleFeatures = torch.zeros((self.samplePoints.shape[0],self.dataFeatures.shape[-1]))
        resampleFeatures = (Vi(self.dataFeatures*Knorm)*K).sum(dim=0)
        '''
        for i in range(self.dataFeatures.shape[-1]):
            resampleFeatures[:,i] = (self.dataFeatures[:,i][...,None]*K/Knorm).sum(axis=0)
        '''
        return resampleFeatures
        