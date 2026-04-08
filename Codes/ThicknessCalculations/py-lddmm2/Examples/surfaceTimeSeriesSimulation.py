import logging
from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
from base.loggingUtils import setup_default_logging
from base.kernelFunctions import Kernel
from base import surfaces as surfaces, surfaceTimeSeries as match
from base.timeSeriesExamples import Ellipses, Ellipse


def getModel(model):
    sigma = 6.5
    sigmaDist = 2.5
    sigmaError = 0.5
    outputDir = '../Output/Temp'
    errorType = 'varifold'
    res = {}

    if model == 'atrophy':
        rdir = '../TestData/AtrophyLargeNoise/'
        fv0 = surfaces.Surface(surf=rdir + 'baseline.vtk')
        fv1 = [fv0]
        for k in range(5):
            fv1 += [surfaces.Surface(surf=rdir + 'followUp' + str(2 * k + 1) + '.vtk')]
        outputDir = '../Output/timeSeriesNoAtrophy'
        lmk = None
    elif model == 'ellipses':
        fv0 = Ellipse(withLandmarks=True)
        ell = Ellipses(nsurf=10, a=[1.5, 5], b=[1, 0.3], c=[0.9, 3], withLandmarks=True)
        fv1 = ell.fv
        lmk = (fv0.lmk, ell.lmk, 1.)
        outputDir = '../Output/timeSeriesEllipses'
        sigma = 1.0
        sigmaDist = 1.0
        sigmaError = 1.0
    else:
        logging.error("unrecognized model ", model)
        return res

    res['outputDir'] = outputDir
    res['errorType'] = errorType
    res['lmk'] = lmk
    res['sigma'] = sigma
    res['sigmaDist'] = sigmaDist
    res['sigmaError'] = sigmaError
    res['fv0'] = fv0
    res['fv1'] = fv1

    return res


if __name__=="__main__":
    setup_default_logging('../Output/Temp', fileName='info', stdOutput = True)
    model = 'ellipses'
    res = getModel(model)

    K1 = Kernel(name='laplacian', sigma = res['sigma'], order=4)
    options = {
        'outputDir': res['outputDir'],
        'mode': 'normal',
        'maxIter': 2000,
        'affine': 'euclidean',
        'regWeight': 1.,
        'Landmarks': res['lmk'],
        'affineWeight': .1,
        'KparDiff': K1,
        'KparDist': ('gauss', res['sigmaDist']),
        'sigmaError': res['sigmaError'],
        'errorType': res['errorType'],
        'algorithm': 'bfgs',
        'internalWeight': 0.1,
        'internalCost': 'elastic'
    }


    f = match.SurfaceTimeMatching(Template=res['fv0'], Target=res['fv1'], options=options)

    f.optimizeMatching()



