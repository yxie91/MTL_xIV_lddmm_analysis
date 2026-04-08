import glob
import numpy as np
import sys
from sys import path as sys_path
sys_path.append('../../')
sys_path.append('../utils/')
from varap.utils.subSample import *
from varap.io.load_utils import centerAndScale
from varap.io.writeOut import writeParticleVTK
import os
import pickle
import scipy as sp
from scipy import io

class BarSeqGeneSubsetLoader:
    
    def __init__(self,rootDir,res,geneFeat='nu_G',geneInds=None,dimEff=None,featNames=None):
        '''
        rootDir = directory with XnuX.npz files for each slice
        res = [x,y,z] resolution; x = 0 if finest resolution is unknown
        numF = number of feature dimensions
        deltaF = true if features are encoded with index of feature dimension rather than size numF vector
        dimEff = effective dimension of data (2 if filenames represent slices, 3 if represent slabs)
        geneFeat = True indicates to use gene distribution as features, otherwise uses cell type
        '''
        if ('.pkl' in rootDir):
            with open(rootDir, 'rb') as f:  # Python 3: open(..., 'rb')
                self.filenames, self.res, self.sizes,self.numFeatures,self.geneInds,self.dimEff,self.filenames_subsample,self.featNames,self.fName = pickle.load(f)
                print("filenames: ", self.filenames)
                print("filenames_subsample: ", self.filenames_subsample)
        elif ('.npz' in rootDir):
            self.filenames = [rootDir]
            allFiles = np.load(rootDir)
            self.res = res
            self.fName = geneFeat
            self.numFeatures = allFiles[self.fName].shape[-1]
            self.geneInds = geneInds
            self.sizes = [allFiles[allFiles.files[0]].shape[0]]
            self.dimEff = 3
            self.filenames_subsample = None
            if featNames is not None:
                self.featNames = featNames
            else:
                self.featNames = None
        else:
            #self.filenames = glob.glob(rootDir + 'slice*centered*npz') # original
            #self.filenames = glob.glob(rootDir + 'slice*npz') # aligned high resolution
            self.filenames = glob.glob(rootDir + '*cellSlice*.npz') # aligned high resolution cells
            self.filenames = glob.glob(rootDir + '*allGeneSlice*.npz') # aligned all genes
            self.filenames = glob.glob(rootDir + '*cellGeneSlice*.npz') # aligned genes in cells 
            newlist = []
            for ff in self.filenames:
                if not 'US' in ff:
                    newlist.append(ff)
            self.filenames = newlist
            self.res = res # x,y,z resolution as list
            if numF is not None:
                self.numFeatures = numF
            else:
                self.numFeatures = None

            self.geneInds = geneInds
            self.sizes = None
            if dimEff is not None:
                self.dimEff = dimEff
            else:
                self.dimEff = None

            self.filenames_subsample = None
            
            if featNames is not None:
                self.featNames = featNames
            else:
                self.featNames = None
            self.fName = geneFeat # nu_G = genes, nu_T = cell types, nu_R = region types
    
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
            if len(n[self.fName].shape) < 2 or n[self.fName].shape[1] == 1 or self.deltaF:
                uniqueF.append(np.unique(n[self.fName]))
            
            else:
                if self.numFeatures is None:
                    self.numFeatures = n[self.fName].shape[1]
                else:
                    print("Expected Features is ", self.numFeatures)
                    print("Features Read in Dataset is ", n[self.fName].shape[1])
        if self.numFeatures is None:
            self.numFeatures = len(np.unique(np.asarray(uniqueF)))
        if self.geneInds is not None:
            self.numFeatures = len(self.geneInds)
        if self.featNames is None:
            self.featNames = []
            for f in range(self.numFeatures):
                self.featNames.append('Feat' + str(f))
        
        self.sizes = sizes
        print("set sizes: ", self.sizes)
        print("set features: ", self.numFeatures)
        return max(self.sizes), self.numFeatures
    
    def getNumberOfFiles(self):
        print("sizes are: ", self.sizes)
        print("total is: ", sum(self.sizes))
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
            nuZ = info[info.files[1]] # assume subsampled is just the single feature set
            return X,nuX,Z,nuZ
                
                
    def getSlice(self,index):
        '''
        returns the data corresponding to a single slice only 
        '''
        
        info = np.load(self.filenames[index])
        coordinates = info[info.files[0]]
        features = info[self.fName]
        return coordinates, features
    
    def centerAndScaleAll(self):
        keepZ = self.dimEff == 2
        newFilenames = []
        for f in self.filenames:
            newFilenames.append(centerAndScale(f,keepZ=keepZ))
        self.filenames = newFilenames
        return
    
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
            sn = outpath + self.filenames[i].split('/')[-1].replace('.npz','') + '_' + self.fName + '_US.npz'
            fs.append(sn)
            np.savez(sn,Z=Z,nu_Z=nuZ)
        self.filenames_subsample = fs
        print("starting number of particles, ", sum(self.sizes))
        print("target number of particles, ", count)
        return
    
    def retrieveSubSampleStratified(self,outpath,resolution,alpha=0.75):
        fs = []
        for i in range(len(self.filenames)):
            sn = outpath + self.filenames[i].split('/')[-1].replace('.npz','') + '_' + self.fName + '_US.npz'
            fs.append(sn)
        self.filenames_subsample = fs
        return
    
    def writeSubSampled(self):
        for f in self.filenames_subsample:
            writeParticleVTK(f,condense=False,featNames=self.featNames)
        return
    
    def saveToPKL(self,outpath):
        with open(outpath, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([self.filenames, self.res, self.sizes, self.numFeatures,self.deltaF,self.dimEff,self.filenames_subsample,self.featNames,self.fName], f)
        return
    
    def writeAll(self,outpath):
        X = np.zeros((sum(self.sizes),3))
        nuX = np.zeros((sum(self.sizes),self.numFeatures))
        
        c = 0
        for f in range(len(self.filenames)):
            x,nux = self.getSlice(f)
            X[c:c+self.sizes[f],...] = x
            nuX[c:c+self.sizes[f],...] = nux
            c += self.sizes[f]
        np.savez(outpath+'.npz',Z=X,nu_Z=nuX)
        writeParticleVTK(outpath+'.npz',condense=False,featNames=self.featNames)
        return
                         

if __name__ == '__main__':
    fp = '/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/SliceToSlice/BarSeqAligned/Whole_Brain_2023/sig0.25/Genes/'
    origDataFP = '/cis/home/kstouff4/Documents/SpatialTranscriptomics/BarSeq/Whole_Brain_2023/'
    pref = 'filt_neurons_D079_3L_goodsubset_CCFv2'
    pref = 'rolonies'
    #pref = 'filt_neurons-clust3'
    ct = sp.io.loadmat(origDataFP+pref+'_cellTypes.mat',appendmat=False)
    cellTypeNames = ct['cellTypes']
    ctList = []
    for i in range(len(cellTypeNames)):
        ctList.append(cellTypeNames[i][0][0])
    cellTypeNames = ctList
    
    gtList = []
    gt = sp.io.loadmat(origDataFP+pref+'_geneNames.mat',appendmat=False)
    geneTypeNames = gt['geneNames']
    for i in range(len(geneTypeNames)):
        gtList.append(geneTypeNames[i][0][0])
    geneTypeNames = gtList
    
    rtList = []
    rt = sp.io.loadmat(origDataFP + pref + '_regionTypes.mat',appendmat=False)
    regionTypeNames = rt['regionTypes']
    for i in range(len(regionTypeNames)):
        rtList.append(regionTypeNames[i][0][0])
    regionTypeNames = rtList
    
    a = BarSeqCellLoader(fp,[0.0,0.0,0.200],featNames=geneTypeNames,geneFeat='nu_G',dimEff=2)
    print("filenames are: ", a.filenames)
    #a.featNames = genes
    print("feature names are:", a.featNames)
    particles,features = a.getSizes()
    a.saveToPKL(fp+'initialHighCellGenes_nu_G.pkl')
    a.writeAll(fp+'initialHighCellGenes_nu_G')
    #print(a.sizes)
    #print(a.numFeatures)
    #print(sum(a.sizes))
    sigma = 0.1
    a.subSampleStratified(fp,sigma,alpha=0.75)
    #a.writeSubSampled()
    a.saveToPKL(fp+'initialHighLowCellGenes_nu_G.pkl')
    #a.centerAndScaleAll()
    
    