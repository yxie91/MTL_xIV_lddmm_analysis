from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib
if 'DISPLAY' in os.environ:
    matplotlib.use('qt5Agg')
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pykeops
pykeops.clean_pykeops()
import logging
from base import loggingUtils
from base.meshes import Mesh, thick2D
from base.kernelFunctions import Kernel
from base.meshMatching import MeshMatching
from base.secondOrderMeshMatching import SecondOrderMeshMatching
from base.imageMatchingLDDMM import ImageMatching
from base.meshExamples import TwoBalls, TwoDiscs, MoGCircle, TwoEllipses, Rbf, Heart
plt.ion()

# model = 'GaussCenters'
#model = 'Heart'
# model = 'Ellipses'
# model = 'EllipsesTranslation'
model = 'Spheres'
secondOrder = False
shrink = False
shrinkRatio = 0.75
eulerian = False
s = 20

if secondOrder:
    typeCost = 'LDDMM'
    order = '_SO_'
    internalCost = None
else:
    typeCost = 'LDDMM'
    internalCost = 'elastic_energy'
    order = ''

if shrink:
    sh = f'_shrink_{shrinkRatio:.2}_'
else:
    sh = ''


def compute(model, thick=False):
    dirichletImage = False
    randomWeights = False
    if internalCost is None:
        sigmaKernel = 1.
        sigmaDist = 5.
        sigmaError = .1
        regweight = 1.
    else:
        sigmaKernel = .1
        sigmaDist = 5.
        sigmaError = .1
        regweight = 1.

    internalWeight = 10.
    delta = .25

    loggingUtils.setup_default_logging('../Output', stdOutput=True)
    if model == 'Circles':
        fv0 = TwoDiscs(largeRadius=10, smallRadius=4.5)
        fv1 = TwoDiscs(largeRadius=12, smallRadius=3)

        ftemp = fv0
        ftarg = fv1
        sigmaDist = 1.
        sigmaKernel = 1.
        dirichletImage = True
        randomWeights = False
    elif model == 'Ellipses':
        fv0 = TwoEllipses(Boundary_a=14, Boundary_b=6, smallRadius=0.25)
        fv1 = TwoEllipses(Boundary_a=12, Boundary_b=10, smallRadius=.4,
                          translation=[0.25, -0.1])
        sigmaDist = 1.
        sigmaKernel = 1.
        sigmaError = 0.01
        ftemp = fv0
        ftarg = fv1
        dirichletImage = True
        randomWeights = False
    elif model == 'EllipsesTranslation':
        fv0 = TwoEllipses(Boundary_a=12, Boundary_b=10, smallRadius=.3,
                          translation=[0.3, 0], volumeRatio=5000)
        fv1 = TwoEllipses(Boundary_a=12, Boundary_b=10, smallRadius=.3,
                          translation=[-0.3, 0], volumeRatio=5000)
        sigmaDist = [2., 1.]
        sigmaKernel = 1.
        sigmaError = .1
        ftemp = fv0
        ftarg = fv1
        dirichletImage = True
        randomWeights = True
    elif model == 'Spheres':
        ftemp = TwoBalls(largeRadius=10, smallRadius=4.5, maxh=0.25)
        ftarg = TwoBalls(largeRadius=15, smallRadius=3, maxh=0.25)
        sigmaKernel = 2.
        sigmaDist = 1.
        dirichletImage = True
        randomWeights = True
    elif model == 'SpheresSmall':
        ftemp = TwoBalls(largeRadius=10, smallRadius=4.5)
        ftarg = TwoBalls(largeRadius=15, smallRadius=3)
        logging.info(f'Number of faces in template: {ftemp.faces.shape[0]}')
        sigmaKernel = 2.
        sigmaDist = 1.
        dirichletImage = True
        randomWeights = True
    elif model == 'GaussCenters':
        sigmaKernel = 5.
        sigmaDist = 5.
        sigmaError = .5
        thick = False
        ftemp = MoGCircle(largeRadius=10, nregions=10, targetSize=1000,
                          volumeRatio=20000)
        centers = ftemp.GaussCenters + .5 * np.random.normal(0, 1, ftemp.GaussCenters.shape)
        ftarg = MoGCircle(largeRadius=12, centers=1.2*centers,
                          typeProb=ftemp.typeProb, alpha=ftemp.alpha,
                          targetSize=1000, volumeRatio=20000)
    elif model == 'Heart':
        ftarg = Heart(zoom=100, targetSize=1000)
        ftemp = Heart(p=1.5, scales=(1.1, 1.75), zoom=100, targetSize=1000)
        sigmaKernel = 5.
        internalWeight = 10
        sigmaError = 1.
        sigmaDist = 10
        regweight = 1.
    elif model == 'RBF':
        N = 100000
        thick = False
        ftemp = Rbf(N, d=2, sigma=0.05)
        ftarg = Rbf(N, d=2, sigma=0.05)
        sigmaKernel = .1
        sigmaDist = .2
        sigmaError = 1.
    else:
        return

    # Transform image
    if thick and ftemp.vertices.shape[1] == 2:
        ftemp = thick2D(ftemp, delta=delta)
        ftarg = thick2D(ftarg, delta=delta)

    epsilon = 0.5
    alpha = np.sqrt(1 - epsilon)
    beta = (np.sqrt(1 - epsilon + ftemp.image.shape[1] * epsilon) - alpha) / np.sqrt(ftemp.image.shape[1])
    ftemp.updateImage(alpha * ftemp.image + beta * ftemp.image.sum(axis=1)[:, None])
    ftarg.updateImage(alpha * ftarg.image + beta * ftarg.image.sum(axis=1)[:, None])

    s0 = 1e9
    if dirichletImage:
        image1 = np.zeros(ftemp.image.shape)
        print(image1.shape)
        for k in range(ftemp.faces.shape[0]):
            image1[k, :] = stats.dirichlet.rvs(ftemp.image[k, :] * s0)
        ftemp.updateImage(image1)

        image1 = np.zeros(ftarg.image.shape)
        for k in range(ftarg.faces.shape[0]):
            image1[k, :] = stats.dirichlet.rvs(ftarg.image[k, :] * s)
        ftarg.updateImage(image1)

    if randomWeights:
        sw = 10*s
        # w = np.zeros(ftemp.faces.shape[0])
        # for k in range(ftemp.faces.shape[0]):
        #     w[k] = stats.gamma.rvs(sw * ftemp.volumes[k]) / (sw * ftemp.volumes[k])
        # ftemp.updateWeights(w)
        w = np.zeros(ftarg.faces.shape[0])
        for k in range(ftarg.faces.shape[0]):
            w[k] = stats.gamma.rvs(sw * ftarg.face_weights[k], loc=0,
                                   scale=1/sw)
        ftarg.updateWeights(w)

    if eulerian:
        # Using standard image matching
        resolution = 0.1
        margin = 5
        imin = ftemp.vertices.min(axis=0) - margin
        imax = ftemp.vertices.max(axis=0) + margin
        imgTemp = ftemp.toImage(resolution=resolution, index=0, margin=5,
                                bounds=[imin, imax])
        imgTarg = ftarg.toImage(resolution=resolution, index=0, margin=5,
                                bounds=[imin, imax])
        sig = sigmaKernel / resolution
        sigDist = sigmaDist[0] / resolution
        logging.info(f"sigma for image matching: {sig:.2f}, {sigDist:.2f}")

        options = {
            'dim': ftarg.dim,
            'timeStep': 0.1,
            'algorithm': 'bfgs',
            'sigmaKernel': sig,
            'order': 3,
            'typeKernel': 'laplacian',
            'sigmaError': 5.,
            'rescaleFactor': 1.,
            'padWidth': 15,
            'affineAlign': None,
            'outputDir': '../Output/meshMatchingTestImageComparison/'+model+order+sh+f'_s_{s:.0f}',
            'mode': 'normal',
            'normalize': 255.,
            'regWeight': 1.,
            'sigmaSmooth': sigDist,
            'maxIter': 1000
        }
        f = ImageMatching(Template=imgTemp, Target=imgTarg, options=options)

        f.restartRate = 50
    else:
        K1 = Kernel(name='laplacian', sigma=sigmaKernel)
        options = {
            'outputDir': '../Output/meshMatchingResults/' + model + order + sh+f'_s_{s:.0f}_'+str(internalCost),
            'mode': 'normal',
            'maxIter': 2000,
            'affine': 'none',
            'rotWeight': 100,
            'transWeight': 10,
            'scaleWeight': 1.,
            'affineWeight': 100.,
            'KparDiff': K1,
            'KparDist': ('gauss', sigmaDist),
            'KparIm': ('euclidean', 1.),
            'sigmaError': sigmaError,
            'errorType': 'measure',
            'internalCost': internalCost,
            'internalWeight': internalWeight,
            'unreducedResetRate': 50,
            'unreduced': False,
            'algorithm': 'bfgs',
            'randomInit': True,
            'regWeight': regweight,
            'lame_mu': .1,
            'lame_lambda': .1,
            'pk_dtype': 'float32'
        }

        if shrink:
            ftemp = ftemp.shrinkTriangles(ratio=shrinkRatio)
            ftarg = ftarg.shrinkTriangles(ratio=shrinkRatio)

        if secondOrder:
            f = SecondOrderMeshMatching(Template=ftemp, Target=ftarg,
                                        options=options)
        else:
            f = MeshMatching(Template=ftemp, Target=ftarg, options=options)

        logging.info(f'Mesh matching {ftemp.vertices.shape[1]}-D, {ftemp.vertices.shape[0]} vertexes, {ftemp.faces.shape[0]} simplexes.')

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
            f.sgdEpsInit = .01 / ftemp.vertices.shape[0]
        else:
            f.saveRate = 10

    f.optimizeMatching()
    plt.ioff()
    plt.show()

    return f


compute(model, thick=True)
