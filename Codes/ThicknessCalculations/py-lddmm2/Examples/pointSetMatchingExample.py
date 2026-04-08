from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
import os
import matplotlib
if 'DISPLAY' in os.environ:
    matplotlib.use('qt5Agg')
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from base import loggingUtils
from base.pointSets import PointSet
from base.kernelFunctions import Kernel
from base.pointSetExamples import TwoDiscs, Rbf
from base.surfaceExamples import Ellipse
from base.pointSetMatching import PointSetMatching
from base.secondOrderPointSetMatching import SecondOrderPointSetMatching
plt.ion()

# model = 'Gaussian'
dim = 3
true_dim = 2
N = 1000
secondOrder = True
# model = 'RBF'
model = 'roll'
#model = 'EllipsesWithLandmarks'
errorType = 'measure'


def getModel(model, N):
    res = {}
    res['sigmaKernel'] = 1.
    res['sigmaDist'] = 2.
    res['sigmaError'] = 100.
    res['errorType'] = 'measure'
    res['passenger'] = None
    res['timeStep'] = 0.1
    if model == 'flatNormal':
        dim = 3
        A = np.random.normal(0, 1, size=(dim, dim))
        R = expm(A - A.T)
        fv0 = np.zeros((N, dim))
        fv0[:, :true_dim] = np.random.normal(0, 1, size=(N, true_dim))
        fv0 /= np.sqrt((fv0**2).sum(axis=1))[:, None]
        res['fv0'] = PointSet(data=np.dot(fv0, R))
        A = np.random.normal(0, 1, size=(dim, dim))
        R = expm(A - A.T)
        fv1pts = np.zeros((N, dim))
        fv1pts[:, :true_dim] = np.random.normal(0, 1, size=(N, true_dim))
        fv1pts /= np.sqrt((fv1pts**2).sum(axis=1))[:, None]
        res['fv1'] = PointSet(data=np.dot(fv1pts, R))
    elif model == 'disc':
        res['fv0'] = TwoDiscs(largeRadius=10, smallRadius=4.5)
        res['fv1'] = TwoDiscs(largeRadius=12, smallRadius=3)
    elif model == 'roll':
        loops = 3
        res['timeStep'] = 0.05
        t = np.linspace(0, loops, N)
        dim = 3
        fv0 = np.zeros((N, dim))
        fv0[:, 0] = 2*np.pi*t
        res['fv1'] = PointSet(data=fv0)
        fv1 = np.zeros((N, dim))
        fv1[:, 0] = (2 - t/t[-1]) * np.cos(2*np.pi*t)
        fv1[:, 1] = (2 - t/t[-1]) * np.sin(2*np.pi*t)
        res['fv0'] = PointSet(data=fv1)
        res['sigmaKernel'] = 1. #[5., 1.]
        res['sigmaDist'] = [2., .5]
        res['sigmaError'] = 1.
        res['errorType'] = 'L2'
    elif model == 'roll_inverted':
        loops = 3
        res['timeStep'] = 0.05
        t = np.linspace(0, loops, N)
        dim = 2
        fv0 = np.zeros((N, dim))
        fv0[:, 0] = (2 - t/t[-1]) * np.cos(2*np.pi*t)
        fv0[:, 1] = (2 - t/t[-1]) * np.sin(2*np.pi*t)
        res['fv0'] = PointSet(data=fv0)
        fv1 = np.zeros((N, dim))
        fv1[:, 0] = -(2 - t/t[-1]) * np.cos(2*np.pi*t)
        fv1[:, 1] = (2 - t/t[-1]) * np.sin(2*np.pi*t)
        res['fv1'] = PointSet(data=fv1)
        res['sigmaKernel'] = 5. #[5., 1.]
        res['sigmaDist'] = [2., .5]
        res['sigmaError'] = 1.
        res['errorType'] = 'L2'
    elif model == 'RBF':
        res['fv0'] = Rbf(N, d=2, sigma=0.05)
        res['fv1'] = Rbf(N, d=2, sigma=0.05)
        res['sigmaKernel'] = .1
        res['sigmaDist'] = .2
        res['sigmaError'] = 0.001*N
    elif model == 'EllipsesWithLandmarks':
        s0 = Ellipse(withLandmarks=True)
        res['fv0'] = PointSet(data=s0.lmk)
        res['passenger'] = PointSet(data=s0.vertices)
        ell = Ellipse(radius=[1.5, 0.9, 3], withLandmarks=True)
        res['fv1'] = PointSet(data=ell.lmk)
        res['sigmaKernel'] = 1.0
        res['sigmaDist'] = 1.0
        res['sigmaError'] = .1

    return res

# R2, T2 = rigidRegistration((fv1pts, fv1.points))


res = getModel(model, N)
fv0 = res['fv0']
fv1 = res['fv1']
loggingUtils.setup_default_logging('../Output', stdOutput=True)

K1 = Kernel(name='gauss', sigma=res['sigmaKernel'])

# # sm = PointSetMatchingParam(timeStep=0.1, )
# sm.KparDiff.pk_dtype = 'float64'
# sm.KparDist.pk_dtype = 'float64'
options = {
    'outputDir': '../Output/pointSetMatchingTest/'+model,
    'timeStep': res['timeStep'],
    'mode': 'normal',
    'maxIter': 10000,
    'burnIn': 10,
    'epsInit': .1,
    'randomInit': True,
    'affine': 'none',
    'rotWeight': 100,
    'transWeight': 100.,
    'scaleWeight': 10.,
    'affineWeight': 100.,
    'KparDiff': K1,
    'KparDist': ('gauss', res['sigmaDist']),
    'sigmaError': res['sigmaError'],
    'errorType': res['errorType'],
    'unreducedResetRate': 50,
    'unreduced': False,
    'algorithm': 'bfgs',
    'unreducedWeight': 0.0,
    'passenger': res['passenger'],
    'pk_dtype': 'float32'
}
if secondOrder:
    f = SecondOrderPointSetMatching(Template=fv0, Target=fv1, options=options)
else:
    f = PointSetMatching(Template=fv0, Target=fv1, options=options)

if f.options['algorithm'] == 'sgd':
    f.set_sgd(100, 100, 100)
    f.sgdBurnIn = 25000
    f.maxIter = 50000
    f.saveRate = 100
    f.options['maxIter'] = 100000
    f.unreducedResetRate = 1000
    f.unreducedRecomputeWeightsMax = 500
    # f.param.sigmaError = 0.01
    f.sgdNormalization = 'sdev'
    f.sgdEpsInit = 0.1/fv0.vertices.shape[0]
else:
    f.saveRate = 10

f.optimizeMatching()
plt.ioff()
plt.show()

