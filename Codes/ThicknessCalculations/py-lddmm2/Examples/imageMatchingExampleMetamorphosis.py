from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
import matplotlib
matplotlib.use("QT5Agg")
import matplotlib.pyplot as plt
import pyfftw
import logging
from skimage.io import imread
from skimage.transform import resize
from copy import deepcopy
from base import loggingUtils
from base.gridscalars import GridScalars
from base.imageMatchingMetamorphosis import Metamorphosis
from base.imageExamples import Circle, TwoCircles, ThreeCircles

pyfftw.config.NUM_THREADS = -1

prior_run = False
loggingUtils.setup_default_logging('', stdOutput=True)
choice = 3
N = 100

logging.info(f'choice = {choice}')
if choice == 1:
    ftemp = '../TestData/Images/2D/faces/s23/5.pgm'
    ftarg = '../TestData/Images/2D/faces/s23/7.pgm'
    name = 'faces'
elif choice == 2:
    ftemp = Circle(N=2*N+1)
    ftemp.data *= 255
    ftarg = TwoCircles(N=2*N+1)
    ftarg.data *= 255
    name = 'Circle_TwoCircles'
elif choice == 3:
    ftarg = ThreeCircles(N=2*N+1, radius=(.4, .3, .3), offset=(.5, 0, -.5, .5, .5))
    ftemp = TwoCircles(N=2*N+1)
    name = 'TwoCircles_ThreeCircles'
    ftemp.data *= 255
    ftarg.data *= 255
elif choice == 4:
    f0 = resize(imread('../TestData/Images/2D/TopChange/bagel.png').astype(float), (2*N+1, 2*N+1))
    f1 = resize((imread('../TestData/Images/2D/TopChange/pretzel.png') > 0.5).astype(float), (2*N+1, 2*N+1))
    name = 'BagelToPretzel'
    ftemp = GridScalars(grid=f0)
    ftarg = GridScalars(grid=f1)
    ftemp.data *= 255
    ftarg.data *= 255

options = {
    'dim': 2,
    'timeStep': 1/30,
    'algorithm': 'bfgs',
    'sigmaKernel': 5,
    'orderKernel': 3,
    'typeKernel': 'laplacian',
    'sigmaError': 10.,
    'sigmaSmooth': 1.,
    'rescaleFactor': 1.,
    'padWidth': 50,
    'affineAlign': None,
    'outputDir': '../Output/imageMatchingExampleMeta/' + name,
    'mode': 'normal',
    'regWeight': 1.,
    'imgCoeff': 1.,
    'ZFreezeTime': 0,
    'LvFreezeTime': 5,
    'FreezeFrequency': 20,
    'maxIter': 2000
}
if prior_run:
    options['sigmaSmooth'] *= 4.
    options['maxIter'] = 500
    f = Metamorphosis(Template=ftemp, Target=ftarg, options=options)
    f.optimizeMatching()

    options['sigmaSmooth'] /= 2.
    options['maxIter'] = 500
    options['Lv'] = deepcopy(f.control['Lv'])
    f = Metamorphosis(Template=ftemp, Target=ftarg, options=options)
    f.optimizeMatching()

    options['sigmaSmooth'] /= 2.
    options['maxIter'] = 2000
    options['Lv'] = deepcopy(f.control['Lv'])

f = Metamorphosis(Template=ftemp, Target=ftarg, options=options)
f.optimizeMatching()

plt.ioff()
plt.show()

