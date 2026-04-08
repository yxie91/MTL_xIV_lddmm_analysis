from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
import logging
import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import pyfftw
from skimage.io import imread
from skimage.transform import resize

pyfftw.config.NUM_THREADS = -1


from base.loggingUtils import setup_default_logging
from base.gridscalars import GridScalars
from base.topChange import TopChange
from base.imageExamples import Circle, TwoCircles, ThreeCircles
from base import loggingUtils
from base.fourierKernel import Kernel


loggingUtils.setup_default_logging('', stdOutput=True)

logging.info("Controlled shape evolution with topological change.\nThis work is part of Daniel Solano's Ph.D. dissertation")

T = 20
N = max(100, int(25 * np.sqrt(T)))
choice = 2
h0 = 0.025
TopCost = 10.

if choice == 1:
    ftemp = Circle(N=2*N+1)
    ftarg = TwoCircles(N=2*N+1)
    name = 'Circle_TwoCircles'
elif choice == 2:
    ftarg = ThreeCircles(N=2*N+1, radius=(.4, .3, .3), offset=(.5, 0, -.5, .5, .5))
    ftemp = TwoCircles(N=2*N+1)
    TopCost = 100.
    name = 'TwoCircles_ThreeCircles'
elif choice == 2.5:
    ftemp = ThreeCircles(N=2*N+1, radius=(.4, .3, .3), offset=(.5, 0, -.5, .5, .5))
    ftarg = TwoCircles(N=2*N+1)
    name = 'ThreeCircles_TwoCircles'
elif choice == 3:
    f0 = resize(imread('../TestData/Images/2D/TopChange/spock.png').astype(float), (2*N+1, 2*N+1))
    f1 = resize(imread('../TestData/Images/2D/TopChange/hand.png').astype(float), (2*N+1, 2*N+1))
    name = 'SpockToHand'
    ftemp = GridScalars(grid=f0)
    ftarg = GridScalars(grid=f1)
elif choice == 4:
    # h0 = 0.05
    TopCost = 100.
    f0 = resize(imread('../TestData/Images/2D/TopChange/bagel.png').astype(float), (2*N+1, 2*N+1))
    f1 = resize((imread('../TestData/Images/2D/TopChange/pretzel.png') > 0.5).astype(float), (2*N+1, 2*N+1))
    name = 'BagetToPretzel'
    ftemp = GridScalars(grid=f0)
    ftarg = GridScalars(grid=f1)


a_ = 0.01
N = ftemp.data.shape[0]//2
T = 30
h = h0/np.sqrt(T)
KparDiff = Kernel(name='laplacian', sigma=.05*N, order=3, dim=2, normalize=False,
                  fftw=True)
#h = .5

options = {
    'mode': 'normal',
    'KparDiff': KparDiff,
    'outputDir': '../Output/TopChange/'+name,
    'dim': 2,
    'pMargin': 2,
    'rMargin': a_,
    'Tsize': T,
    'TopCost': TopCost,
    'resol': h0,
    'sigmaError': 1e-3,
    'sigmaSmooth': h0,
    'endpointH1cost': 1,
    'endpointSigmaSmooth': h,
    'diffeoCondition': 100.,
    'diffeoWarmup': 0,
    'alternate': 0,
    'padWidth': 50,
    'pplot': True,
    'saveRate': 10,
    'maxIter': 2000
}

f = TopChange(Template=ftemp, Target=ftarg, options=options)

f.optimize()

plt.ioff()
plt.show()
