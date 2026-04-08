import numpy as np
import logging
from copy import deepcopy
import time

# Class running stochastic gradient descent
# opt is an optimizable class that must provide the following functions:
#   addProd(g0, step, g1): returns g0 + step * g1 for directions g0, g1
#   getGradientSGD(coeff) returns coeff * gradient; the result can be used as 'direction' in updateTry
#
# optional functions:
#   startOptim(): called before starting the optimization
#   startOfIteration(): called before each iteration
#   endOfIteration() called after each iteration
#   endOptim(): called once optimization is completed
#   gradCoeff: normalizaing coefficient for gradient.
#
# verb: for verbose printing
# epsInit: initial gradient step
# rate: SGD rate

def sgd(opt, verb = True, maxIter=1000, burnIn = 100, epsInit=1., rate = 0.01, normalization = 'sdev',
        TestGradient = False):
    if not hasattr(opt, 'addProd') or not hasattr(opt, 'getGradient'):
        logging.error('Error: required functions for SGD are not provided')
        return

    if hasattr(opt, 'startOptim'):
        opt.startOptim()

    if hasattr(opt, 'gradCoeff'):
        gradCoeff = opt.gradCoeff
    else:
        gradCoeff = 1.0

    logging.info(f'iteration 0: eps Init {epsInit}, normalization {normalization}')
    it = 1
    while it <= maxIter:
        startTime = time.process_time()
        if hasattr(opt, 'startOfIteration'):
            opt.startOfIteration()

        if TestGradient:
            grdTrue = opt.getGradient(gradCoeff, forceComplete=True)
            mgrd = dict()
            vgrd = dict()
            for key in grdTrue.keys():
                if grdTrue[key] is not None:
                    mgrd[key] = np.zeros(grdTrue[key].shape)
                    vgrd[key] = np.zeros(grdTrue[key].shape)
            nsim = 50
            for k in range(50):
                grd_ = opt.getGradient(gradCoeff)
                for key in grdTrue.keys():
                    if grdTrue[key] is not None:
                        mgrd[key] += grd_[key]
                        vgrd[key] += grd_[key]**2

            for key in grdTrue.keys():
                if grdTrue[key] is not None:
                    mgrd[key] /= nsim
                    vgrd[key] = np.sqrt(1e-10 + vgrd[key] / nsim - mgrd[key]**2)

            s = 'Test gradient SGD:'
            for key in grdTrue.keys():
                if grdTrue[key] is not None:
                    s += f'{key}: {np.sqrt(nsim)*(np.fabs(mgrd[key] - grdTrue[key])/vgrd[key]).mean():.2f}; '
            logging.info(s)

        grd = opt.getGradient(gradCoeff)
        if it == 1:
            meanGrd = deepcopy(grd)
            varGrd = dict()
            for k in grd.keys():
                # logging.info(k)
                if grd[k] is not None:
                    varGrd[k] = 1 + grd[k]**2
        else:
            for k in grd.keys():
                # logging.info(k)
                if grd[k] is not None:
                    meanGrd[k] += (grd[k] - meanGrd[k])/(it+1)
                    varGrd[k] += (grd[k]**2 - varGrd[k])/(it+1)

        grd_ = deepcopy(grd)
        if normalization == 'sdev':
            for k in grd.keys():
                if grd[k] is not None:
                    grd_[k] /= epsInit + np.sqrt(varGrd[k] + 1e-10)
                #grd_[k] /= epsInit + np.sqrt(varGrd[k] - meanGrd[k] ** 2 + 1e-10)
        elif normalization == 'var':
            for k in grd.keys():
                if grd[k] is not None:
                    grd_[k] /= epsInit + varGrd[k]

        eps = epsInit / (1 + rate*max(0, it - burnIn))
        opt.update(grd_, eps)
        if verb:
            gabs = 0
            mgabs = 0
            message = f'iteration {it}: '
            for k in grd.keys():
                if grd[k] is not None:
                    # logging.info(k)
                    gabs += np.fabs(grd[k]).max()
                    mgabs += np.fabs(meanGrd[k]).max()
                    message += k + f': {np.fabs(grd_[k]).max():.5f} {np.sqrt(varGrd[k] + 1e-10).max():.5f} '
                    #message += k + f': {np.fabs(grd_[k]).max():.5f} {np.sqrt(varGrd[k] - meanGrd[k] ** 2 + 1e-10).max():.5f} '
            logging.info(message + f' eps = {eps:.5f} time = {time.process_time() - startTime:.2f}')
        if hasattr(opt, 'endOfIteration'):
            opt.endOfIteration()
        it += 1

    if hasattr(opt, 'endOfProcedure'):
        opt.endOfProcedure()

    return opt
