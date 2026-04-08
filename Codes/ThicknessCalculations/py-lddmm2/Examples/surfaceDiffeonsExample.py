from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
from base.surfaces import Surface
from base import gaussianDiffeons as gd
from base import loggingUtils
from base.gaussianDiffeonsSurfaceMatching import SurfaceMatchingDiffeons
from base import examples


def compute(createsurfaces=True):
    if createsurfaces:
        fv1, fv2 = examples.bumps2(centers1=([-0.1, -0.3], [.3, 0.25],
                                             [-0.1, .3], [.25, -.3]),
                                   scale1=(.5, .5, .5, .5),
                                   weights1=(.3, .3, .3, .3),
                                   centers2=([.3, -.3], [-.3, .3],
                                             [.3, .3], [-.3, -.3]),
                                   scale2=(.3, .3, .3, .3),
                                   weights2=(.5, .5, .5, .5), d=25)

    else:
        fv1 = Surface(surf='../TestData/Diffeons/fv1Alt.vtk')
        fv2 = Surface(surf='../TestData/Diffeons/fv2Alt.vtk')

    r0 = 100./fv1.vertices.shape[0]
    withDiffeons = False

    options = {
        'timeStep': 0.1,
        'sigmaKernel': 5.,
        'sigmaError': 1.,
        'errorType': 'varifold',
        'algorithm': 'cg',
        'mode': 'normal',
        'subsampleTemplate': 1,
        'zeroVar': False,
        'subsampleTargetSize': 500,
        'maxIter': 10000,
        'affine': 'euclidean',
        'rotWeight': 1.,
        'transWeight': 1.,
        'scaleWeight': 10.,
        'affineWeight:': 100.
    }

    if withDiffeons:
        gdOpt = gd.gdOptimizer(surf=fv1, sigmaDist=.5, DiffeonEpsForNet=r0,
                               testGradient=True, maxIter=100)
        gdOpt.optimize()
        options['outputDir'] = '../Output/Diffeons/BallsAlt50_500_d'
        f = SurfaceMatchingDiffeons(Template=fv1, Target=fv2, options=options)
        options['Diffeons'] = (gdOpt.c0, gdOpt.S0, gdOpt.idx)
        options['DecimationTarget'] = 100
    else:
        options['outputDir'] = '../Output/Diffeons/Scale100_250_0'
        f = SurfaceMatchingDiffeons(Template=fv1, Target=fv2, options=options)

    loggingUtils.setup_default_logging(options['outputDir'], fileName='info',
                                       stdOutput=True)
    f.optimizeMatching()

    return f


if __name__ == "__main__":
    compute(True)
