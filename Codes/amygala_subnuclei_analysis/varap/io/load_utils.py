'''
Class of generic functions for altering format of input data.
I/O: npz files
'''

# Author: Kaitlin Stouffer (kstouff4@jhmi.edu)

import numpy as np

def centerAndScale(filename,keepZ=False,s=0.001):
    '''
    Center data around center of Mass and then Scale coordinates
    '''
    data = np.load(filename,allow_pickle=True)
    X = data[data.files[0]]
    c = np.mean(X,axis=0)
    X = X - c
    if keepZ:
        z = c[-1]
        X[:,-1] = z
    X = X*s
    di = dict(data)
    di[data.files[0]] = X
    newName = filename.replace('.npz','_centeredAndScaled' + str(s) + '.npz')
    np.savez(newName,**di)
    return newName

def makeOneHot(nu_X):
    u,i = np.unique(nu_X,return_inverse=True)
    z = np.zeros((nu_X.shape[0],len(u)))
    z[np.arange(nu_X.shape[0]),i] = 1.0
    return z
                 
                 
def selectFeatureSubset(filename,featureName,featureIndices,savesuff):
    data = np.load(filename)
    X = data[data.files[0]]
    nu_X = data[featureName]
    
    # make one hot and select indices
    if nu_X.shape[-1] == X.shape[0] or nu_X.shape[-1] < len(featureIndices):
        nu_X = makeOneHot(nu_X)
    
    nu_Xs = nu_X[:,featureIndices]
    
    # remove zeros
    inds = np.squeeze(np.sum(nu_Xs,axis=-1) > 0)
    nu_Xss = nu_Xs[inds,:]
    Xs = X[inds,:]
    
    np.savez(filename.replace('.npz',savesuff + '.npz'),X=Xs,nu_X=nu_Xss)
    return