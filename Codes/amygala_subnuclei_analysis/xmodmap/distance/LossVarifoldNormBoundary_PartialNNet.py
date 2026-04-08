import torch
import xmodmap

from xmodmap.distance.LossVarifoldNormBoundary import LossVarifoldNormBoundary
#from xmodmap.preprocess.BoundaryClassifer import BoundaryClassifier
from xmodmap.preprocess.preprocess import resizeData

#from xmodmap.distance.LossVarifoldNorm import GaussLinKernelSingle


class LossVarifoldNormBoundary_PartialNNet(LossVarifoldNormBoundary):
    '''
    Support weights reflect density of particles in target space and interior of set of coronal sections.
    
    densSigma = bandwith (for data in unit cube) of Gaussian kernel used to compute local density of particles
    in target space.
    '''

    def __init__(self, sigmaVar, w_T, zeta_T, Ttilde, nnetFile, s=1., m=0.):
        
        super().__init__(sigmaVar, w_T, zeta_T, Ttilde)
        self.nnet = xmodmap.preprocess.BoundaryClassifier(param_file=nnetFile)
        
        # rescaling parameters
        self.s = s
        self.m = m

        if self.w_T.shape[-1] != 1:
            self.w_T = self.w_T[..., None]

    def supportWeight(self, qx, lamb):
        sW = self.nnet.predict(1.0 / lamb, resizeData(qx, self.s, self.m))
        return sW[..., None]
    
    def __call__(self, sSx, sSw, zeta_S, pi_ST, lamb):
        return super().__call__(sSx, sSw, zeta_S, pi_ST, lamb)

