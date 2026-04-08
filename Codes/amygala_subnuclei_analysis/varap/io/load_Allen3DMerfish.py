import glob
import numpy as np
import sys
from sys import path as sys_path
sys_path.append('../../')
sys_path.append('../utils/')
from varap.utils.subSample import *
from varap.io.load_utils import centerAndScale, selectFeatureSubset
from varap.io.writeOut import writeParticleVTK
import os
import pickle
import scipy as sp
from scipy import io


class Allen3DMerfishLoader:
    
    def __init__(self,rootDir,res,dimEff=None,numF=None,deltaF=False):
        '''
        rootDir = directory with XnuX.npz files for each slice
        res = [x,y,z] resolution; x = 0 if finest resolution is unknown
        numF = number of feature dimensions
        deltaF = true if features are encoded with index of feature dimension rather than size numF vector
        dimEff = effective dimension of data (2 if filenames represent slices, 3 if represent slabs)
        '''
        if ('.pkl' in rootDir):
            with open(rootDir, 'rb') as f:  # Python 3: open(..., 'rb')
                self.filenames, self.res, self.sizes,self.numFeatures,self.deltaF,self.dimEff,self.filenames_subsample = pickle.load(f)
        else:
            self.filenames = glob.glob(rootDir + '*Xnu*npz.npz')
            self.res = res # x,y,z resolution as list
            if numF is not None:
                self.numFeatures = numF
            else:
                self.numFeatures = None

            self.deltaF = deltaF
            self.sizes = None
            if dimEff is not None:
                self.dimEff = dimEff
            else:
                self.dimEff = None

            self.filenames_subsample = None
    
    def getSizes(self):
        '''
        Returns maximum size (particles) of slices and number of features
        
        Sets number of Features and sizes of slices 
        '''
        
        sizes = []
        uniqueF = []
        
        totS = 0
        
        for f in self.filenames:
            n = np.load(f)
            sizes.append(n[n.files[0]].shape[0])
            if (self.dimEff is None):
                zCoordFirst = n[n.files[0]][0,-1]
                if (np.sum(n[n.files[0]][:,-1] - zCoordFirst) == 0):
                    self.dimEff = 2
                else:
                    self.dimEff = 3
                print("set dimEff: ", self.dimEff)
                
            
            # assume discrete if only one value stored 
            if len(n[n.files[1]].shape) < 2 or n[n.files[1]].shape[1] == 1 or self.deltaF:
                uniqueF.append(np.unique(n[n.files[1]]))
            
            else:
                if self.numFeatures is None:
                    self.numFeatures = n[n.files[1]].shape[1]
                else:
                    print("Expected Features is ", self.numFeatures)
                    print("Features Read in Dataset is ", n[n.files[1]].shape[1])
        if self.numFeatures is None:
            self.numFeatures = len(np.unique(np.asarray(uniqueF)))
        
        self.sizes = sizes
        print("set sizes: ", self.sizes)
        print("set features: ", self.numFeatures)
        return max(self.sizes), self.numFeatures
    
    def getNumberOfFiles(self):
        return len(self.sizes)
    
    def getFilename(self,index):
        return self.filenames[index]
    
    def getFilename_subsample(self,index):
        return self.filenames_subsample[index]
    
    def getHighLowPair(self,index):
        if self.filenames_subsample is None:
            print("No initialization complete. Please call subsample on high resolution data.")
            return
        else:
            X,nuX = self.getSlice(index)
            info = np.load(self.filenames_subsample[index])
            Z = info[info.files[0]]
            nuZ = info[info.files[1]]
            return X,nuX,Z,nuZ
                
                
    def getSlice(self,index):
        '''
        returns the data corresponding to a single slice only 
        '''
        
        info = np.load(self.filenames[index])
        coordinates = info[info.files[0]]
        features = info[info.files[1]]
        return coordinates, features
    
    def subSampleRandom(self,outpath,resolution,overhead=0.1):
        '''
        subsample each of datasets per file in filenames and write in outpath 
        subsample will be done with given resolution
        
        two choices of sampling: random sampling for initialization or stratified sampling with uniform distribution over features
        '''
        if (not os.path.exists(outpath)):
            os.mkdir(outpath) 
        
        fs = []
        count = 0
        for i in range(len(self.filenames)):
            X,nuX = self.getSlice(i)
            Z,nuZ = makeRandomSubSample(X, nuX, resolution, self.numFeatures,C=1.2,dimEff=self.dimEff)
            if (overhead > 0.0):
                Z,nuZ = addOverhead(Z,nuZ,overhead=overhead)
            sn = outpath + self.filenames[i].split('/')[-1].replace('.npz','') + '_RS_o' + str(overhead) + '.npz'
            fs.append(sn)
            np.savez(sn,Z=Z,nu_Z=nuZ)
        self.filenames_subsample = fs
        print("starting number of particles, ", sum(self.sizes))
        print("target number of particles, ", count)
        return
    
    def subSampleStratified(self,outpath,resolution,alpha=0.75):
        
        if (not os.path.exists(outpath)):
            os.mkdir(outpath) 
        
        fs = []
        count = 0
        for i in range(len(self.filenames)):
            X,nuX = self.getSlice(i)
            Z,nuZ = makeStratifiedSubSample(X,nuX,resolution,self.numFeatures,alpha=alpha)
            nuZ = makeUniform(Z,nuZ)
            count += Z.shape[0]
            sn = outpath + self.filenames[i].split('/')[-1].replace('.npz','') + '_US.npz'
            fs.append(sn)
            np.savez(sn,Z=Z,nu_Z=nuZ)
        self.filenames_subsample = fs
        print("starting number of particles, ", sum(self.sizes))
        print("target number of particles, ", count)
        return
    
    def retrieveSubSampleStratified(self,outpath,resolution,alpha=0.75):
        fs = []
        for i in range(len(self.filenames)):
            sn = outpath + self.filenames[i].split('/')[-1].replace('.npz','') + '_US.npz'
            fs.append(sn)
        self.filenames_subsample = fs
        return
    
    def saveToPKL(self,outpath):
        with open(outpath, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([self.filenames, self.res, self.sizes, self.numFeatures,self.deltaF,self.dimEff,self.filenames_subsample], f)
        return

if __name__ == '__main__':
    fp = '/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/XnuX_Aligned/top50MI/initialHighAllCells.pkl'
    a = Allen3DMerfishLoader(fp,[0,0,0.100])
    #print("filenames are: ", len(a.filenames))
    #particles,features = a.getSizes()
    #print(a.sizes)
    #print(a.numFeatures)
    #print(sum(a.sizes))
    #a.saveToPKL(fp+'initialHighAllCells.pkl')
    #a.writeAll(fp+'initialHighAllCells')
    #print(a.sizes)
    #print(a.numFeatures)
    #print(sum(a.sizes))
    sigma = 0.2
    a.subSampleStratified(fp,sigma,alpha=0.75)
    #a.writeSubSampled()
    a.saveToPKL(fp+'initialHighLowAll_' + str(sigma) + '.pkl')
    '''
    sigma = 0.1
    a.subSampleStratified(fp,sigma,alpha=0.75)
    a.saveToPKL(fp+'initialHighLowAll_' + str(sigma) + '.pkl')
    sigma = 0.2
    a.subSampleStratified(fp,sigma,alpha=0.75)
    a.saveToPKL(fp+'initialHighLowAll_' + str(sigma) + '.pkl')
    '''
    #a.centerAndScaleAll()

    
    