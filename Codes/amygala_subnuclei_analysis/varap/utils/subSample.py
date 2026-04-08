import numpy as np

def makeOneHot(nu,maxVal=673,zeroBased=True):
    nu1 = np.zeros((nu.shape[0],maxVal)).astype('float32')
    if (zeroBased):
        nu1[np.arange(nu.shape[0]),np.squeeze(nu).astype(int)] = 1
    else:
        nu1[np.arange(nu.shape[0]),np.squeeze(nu-1).astype(int)] = 1
    return nu1

def makeRandomSubSample(X, nu_X, sig, maxV,C=1.2,dimEff=3):
    # Input:
    # X is the initial data
    # N is the number of random points to subsample based on C
    # dimEff = effective dimensions (2 if wish to subsample according to number in square area rather than cubic volume)

    # Output 
    # Y is the output subsample
    if (dimEff == 3):
        volBB = np.prod(np.max(X,axis=0)-np.min(X,axis=0))
        N = np.round(C*volBB/(sig**3)).astype(int)
        print("N " + str(N) + " vs " + str(X.shape[0]))
    elif (dimEff == 2):
        volBB = np.prod(np.max(X[:,0:2],axis=0) - np.min(X[:,0:2],axis=0))
        N = np.round(C*volBB/(sig**2)).astype(int)
        
    N = min(X.shape[0],N)
    sub_ind = np.random.choice(X.shape[0],replace = False, size = N)
    print(sub_ind.shape)
    Z = X[sub_ind,:]
    nu_Z = nu_X[sub_ind,...] # start weights off as fraction of what you started with
    if (nu_Z.shape[-1] < maxV):
        nu_Z = makeOneHot(nu_Z,maxVal=maxV)*X.shape[0]/N
    else:
        # weigh nu_X with the total delta of mass missing
        xMass = np.sum(nu_X)
        zMass = np.sum(nu_Z)
        c = xMass/zMass
        nu_Z = c*nu_Z
    return Z, nu_Z

def addOverhead(Z,nu_Z,overhead=0.1):
    '''
    Add mass to all of features to execute optimization in full space of features
    '''
    nnu_Z = nu_Z + overhead
    nnu_Z = nnu_Z*(np.sum(nu_Z)/np.sum(nnu_Z))
    
    return nnu_Z

def makeUniform(Z,nu_Z):
    '''
    Replace feature values in nu_Z with mass distributed uniformly across all features
    '''
    nnu_Z = np.zeros_like(nu_Z) + 1.0/nu_Z.shape[-1]
    nnu_Z = nnu_Z*(np.sum(nu_Z,axis=-1)[...,None])
    return nnu_Z

def makeStratifiedSubSample(X,nuX,sig,maxV,alpha=0.75):
    '''
    Sample 1 point at random from a cube of size (sigma*alpha)^3
    Returns the feature values associated with the original point sampled, reweighted to reflect total weight in cube
    No points are returned for cubes with 0 total weight
    '''
    print("making stratified subsample")
    coords = X
    indOriginal = np.arange(nuX.shape[0])
    allInfo = np.stack((coords[:,0],coords[:,1],coords[:,2],indOriginal),axis=-1)
    np.random.shuffle(allInfo)
    coords = allInfo[:,0:3]
    nuX = nuX[allInfo[:,-1].astype(int),...]
    if nuX.shape[-1] < maxV or len(nuX.shape) < 2:
        nuXo = np.zeros((nuX.shape[0],2))
        nuXo = nuXo + 0.5
    else:
        nuXo = nuX
    

    coords_labels = np.floor((coords - np.floor(np.min(coords,axis=0)))/(sig*alpha)).astype(int) # minimum number of cubes in x and y
    totCubes = (np.max(coords_labels[:,0])+1)*(np.max(coords_labels[:,1])+1)*(np.max(coords_labels[:,2])+1)
        
    # Numpy version 
    uInds,sample,inv,counts = np.unique(coords_labels,return_index=True,return_inverse=True,return_counts=True,axis=0) # returns first occurrence of each cube 
    we = np.sum(nuXo,axis=-1)
    bins = np.arange(np.max(inv) + 2)
    bins = bins - 0.5
    bins = list(bins)
    hist,be = np.histogram(inv,bins=bins,weights=we,density=False) # mass per bin should indicate total weight 
    
    nuZ = nuX[np.squeeze(sample),...] # returns first occurrence of mRNA in cube 
    
    Z = coords[np.squeeze(sample),...]
    
    if (nuZ.shape[-1] < maxV or len(nuZ.shape) < 2):
        nu_Z = makeOneHot(nuZ,maxVal=maxV)*np.squeeze(hist)[...,None] #*counts[...,None] # weigh particles based on number of MRNA in each cube
    else:
        nu_Z = np.zeros_like(nuZ)
        temp = nuZ*np.squeeze(hist)[...,None]/np.sum(nuZ,axis=-1)[...,None]
        nu_Z[nuZ > 0] = temp[nuZ > 0]
    print("sume before :" + str(np.sum(nu_Z)))
    nu_Z = nu_Z*np.sum(nuXo)/np.sum(nu_Z)
    print("sum after :" + str(np.sum(nu_Z)))
    print("nu_Z shape, ", nu_Z.shape)
    print("nuZ sum original: " + str(np.sum(nuXo)))
    
    return Z,nu_Z
