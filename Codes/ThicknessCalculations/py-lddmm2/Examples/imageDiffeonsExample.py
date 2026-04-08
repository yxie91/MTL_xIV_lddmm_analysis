from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
import numpy as np
from scipy.ndimage import gaussian_filter
from base.gridscalars import GridScalars
from base.kernelFunctions import *
from base.gaussianDiffeonsImageMatching import ImageMatchingDiffeons
from base import loggingUtils


def compute(createImages=True):
    if createImages:
        [x, y] = np.mgrid[0:100, 0:100]/50.
        x = x-1
        y = y-1

        I1 = gaussian_filter(255*np.array(.06 - ((x)**2 + 1.5*y**2) > 0), 1)
        im1 = GridScalars(grid=I1, dim=2)

        I2 = gaussian_filter(255*np.array(.05 -
                             np.minimum((x-.2)**2 + 1.5*y**2,
                                        (x+.20)**2 + 1.5*y**2) > 0), 1)
        im2 = GridScalars(grid=I2, dim=2)
    else:
        path = '../TestData/Images/2D/'
        im1 = GridScalars(grid=path+'Heart/heart01.tif', dim=2)
        im2 = GridScalars(grid=path+'Heart/heart09.tif', dim=2)
        im1.data = gaussian_filter(im1.data, .5)
        im2.data = gaussian_filter(im2.data, .5)
        print(im2.data.max())

    options = {
        'timeStep': 0.05,
        'sigmaKernel': 5.,
        'sigmaError': 10.,
        'outputDir': '../Output/ImageDiffeons',
        'algorithm': 'bfgs',
        'mode': 'normal',
        'subsampleTemplate': 1,
        'zeroVar': False,
        'targetMargin': 0,
        'templateMargin': 0,
        'DecimationTarget': 5,
        'maxIter': 10000,
        'affine': 'none',
        'rotWeight': 1.,
        'transWeight': 1.,
        'scaleWeight': 10.,
        'affineWeight:': 100.
    }
    f = ImageMatchingDiffeons(Template=im1, Target=im2, options=options)
    f.optimizeMatching()
    return f


if __name__ == "__main__":
    loggingUtils.setup_default_logging('', stdOutput=True)
    compute(createImages=False)
