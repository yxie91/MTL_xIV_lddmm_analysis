import logging
from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
from base.loggingUtils import setup_default_logging
from base.kernelFunctions import Kernel
from base import pointSets, pointSetTimeSeries as match
from base import surfaces
from base.timeSeriesExamples import Ellipses, Ellipse


setup_default_logging('../Output/Temp', fileName='info', stdOutput = True)
model = 'ellipses_landmarks'
sigma = 6.5
sigmaDist = 2.5
sigmaError = 0.5
outputDir = '../Output/Temp'
lmk = None
passenger = None
errorType = 'measure'

if model == 'atrophy':
    rdir = '../TestData/AtrophyLargeNoise/'
    fv0_ = surfaces.Surface(surf=rdir + 'baseline.vtk')
    fv0 = pointSets.PointSet(data=fv0_.vertices)
    fv1 = [fv0]
    for k in range(5):
        fv1_ = surfaces.Surface(surf=rdir + 'followUp' + str(2 * k + 1) + '.vtk')
        fv1 += [pointSets.PointSet(data=fv1_.vertices)]
    outputDir = '../Output/pointSetTimeSeriesAtrophy'
elif model == 'ellipses_unlabelled':
    fv0_ = Ellipse(withLandmarks=True)
    fv0 = pointSets.PointSet(data=fv0_.vertices, weights=1/fv0_.vertices.shape[0])
    ell = Ellipses(nsurf=10, a=[1.5, 5], b=[1, 0.3], c=[0.9, 3], withLandmarks=True)
    lmk = (fv0_.lmk, ell.lmk, 10.)
    fv1 = []
    for fv in ell.fv:
        fv1.append(pointSets.PointSet(data=fv.vertices, weights=1/fv.vertices.shape[0]))
    outputDir = '../Output/pointSetTimeSeriesEllipses_unlabelled'
    sigma = .5
    sigmaDist = .2
    sigmaError = .01
elif model == 'ellipses_landmarks':
    fv0_ = Ellipse(withLandmarks=True)
    fv0 = pointSets.PointSet(data=fv0_.lmk)
    ell = Ellipses(nsurf=10, a=[1.5, 5], b=[1, 0.3], c=[0.9, 3], withLandmarks=True)
    passenger =  pointSets.PointSet(data=fv0_.vertices)
    fv1 = []
    for fv in ell.lmk:
        fv1.append(pointSets.PointSet(data=fv))
    outputDir = '../Output/pointSetTimeSeriesEllipses_landmarks'
    sigma = 1.0
    sigmaDist = 1.0
    sigmaError = 1.0
    errorType = 'L2'
else:
    logging.error("unrecognized model ", model)
    exit()

K1 = Kernel(name='laplacian', sigma = sigma, order=4)
options = {
    'outputDir': outputDir,
    'mode': 'normal',
    'maxIter': 2000,
    'affine': 'none',
    'regWeight': 1.,
    'affineWeight': .1,
    'KparDiff': K1,
    'Landmarks': lmk,
    'passenger': passenger,
    'KparDist': ('gauss', sigmaDist),
    'sigmaError': sigmaError,
    'errorType': errorType,
    'algorithm': 'bfgs',
    'internalWeight': 0.1,
    'internalCost': 'elastic'
}


f = match.PointSetTimeMatching(Template=fv0, Target=fv1, options=options)

f.optimizeMatching()



