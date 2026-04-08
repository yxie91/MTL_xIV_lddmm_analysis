from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
from base.loggingUtils import setup_default_logging
from base.kernelFunctions import Kernel
from base import surfaceMatching
from base import surfaces as surfaces
from base.secondOrderSurfaceTimeMatching import SecondOrderSurfaceTimeMatching
from surfaceTimeSeriesSimulation import getModel


setup_default_logging('../Output/Temp', fileName='info', stdOutput = True)

model = 'ellipses'
res = getModel(model)

K1 = Kernel(name='laplacian', sigma = res['sigma'], order=4)
options = {
    'outputDir': res['outputDir'],
    'mode': 'normal',
    'maxIter': 2000,
    'affine': 'none',
    'regWeight': 1.,
    'Landmarks': res['lmk'],
    'affineWeight': .1,
    'KparDiff': K1,
    'KparDist': ('gauss', res['sigmaDist']),
    'sigmaError': res['sigmaError'],
    'errorType': res['errorType'],
    'typeRegression': 'spline2',
    'algorithm': 'bfgs',
    'internalWeight': 0.1,
    'internalCost': 'elastic'
}


# rdir = '../TestData/AtrophyLargeNoise/'
# fv0 = surfaces.Surface(surf=rdir + 'baseline.vtk')
# # fv1 = [fv0]
# fv1 = []
# for k in range(2):
#     fv1 += [surfaces.Surface(surf=rdir + 'followUp' + str(2 * k + 1) + '.vtk')]
# outputDir = '../Output/timeSeriesNoAtrophy'

# K1 = Kernel(name='gauss', sigma = 6.5, order=4)
## typeRegression = 'geodesic' or 'spline' or 'spline2' (which is 'geodesic' and 'spline' together)
# options = {
#     'outputDir': outputDir,
#     'mode': 'normal',
#     'maxIter': 2000,
#     'affine': 'none',
#     'regWeight': 1.,
#     'Landmarks': None,
#     'affineWeight': .1,
#     'KparDiff': K1,
#     'KparDist': ('gauss', 2.5),
#     'sigmaError': 0.5,
#     'errorType': 'current',
#     'algorithm': 'bfgs',
#     'typeRegression': 'spline2',
#     'internalWeight': 0.,
#     'saveRate': 10,
#     'internalCost': None
# }

f = SecondOrderSurfaceTimeMatching(Template=res['fv0'], Target=res['fv1'], options=options)
print(f.options['testGradient'])
f.optimizeMatching()



