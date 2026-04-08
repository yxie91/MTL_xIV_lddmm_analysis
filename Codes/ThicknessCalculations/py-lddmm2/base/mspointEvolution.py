import logging

import numpy as np
from numba import jit, prange
from . import gaussianDiffeons as gd
import numpy.linalg as LA
from . import affineBasis
import pdb

def mslandmarkPassengerEvolutionEuler(y0, xt, at, KparDiff, scales = [0,19], base_scales = [0, 19], affine=None, options=None, T=1.0):
    # KparDiff should be the full kernel.
    # Scales are the lambda's in x(t, lambda, `), can be any subset of the full scales.
    _options = {'withJacobian': False}
    if options is not None:
        for k in options.keys():
            _options[k] = options[k]

    withJacobian = _options['withJacobian']
    nfullscales = len(KparDiff)
    nbase_scales = len(base_scales)  # number of base scales
    # nbase_scales = len(xt)  # number of base scales
    dim = xt[0].shape[2]
    nsteps = at[0].shape[0]
    timeStep = T / nsteps
    nscales = len(scales)
    yt = []
    Jyt = []
    if isinstance(y0, list):    # y0 itself is a list
        for ii in range(nscales):
            yt.append(np.zeros((nsteps+1, y0[ii].shape[0], dim)))
            yt[ii][0] = y0[ii]
            if withJacobian:
                Jyt.append(np.zeros((nsteps+1, y0[ii].shape[0], 1)))
            else:
                Jyt = None
    else:
        # y0 will be repeated
        for ii in range(nscales):
            yt.append(np.zeros((nsteps+1, y0.shape[0], dim)))
            yt[ii][0] = y0
            if withJacobian:
                Jyt.append(np.zeros((nsteps+1, y0.shape[0], 1)))
            else:
                Jyt = None
    for t in range(nsteps):
        for ii in range(nscales):
            yt[ii][t+1] = yt[ii][t]
            if withJacobian:
                Jyt[ii][t+1] = Jyt[ii][t]
            for jj in range(nbase_scales):
                yt[ii][t+1] += timeStep * KparDiff[scales[ii]][base_scales[jj]].applyK(xt[jj][t], at[jj][t], firstVar=yt[ii][t])
                if withJacobian:
                    Jyt[ii][t+1] += timeStep * KparDiff[scales[ii]][base_scales[jj]].applyDivergence(xt[jj][t], at[jj][t],
                                                                                              firstVar=yt[ii][t])


    output = dict()
    output['yt'] = yt
    output['Jyt'] = Jyt

    return output


def mslandmarkPassengerEvolutionReverse(y0, xt, at, KparDiff, scales=[0,19], base_scales=[0,19], affine=None, options=None, T=1.0):
    _options = {'withJacobian': False}
    if options is not None:
        for k in options.keys():
            _options[k] = options[k]

    withJacobian = _options['withJacobian']

    # N = xt.shape[1]
    dim = xt[0].shape[2]
    nsteps = at[0].shape[0]
    nscales = len(scales)
    nbase_scales = len(base_scales)
    timeStep = T / nsteps

    # K = y0.shape[0]
    # yt = np.zeros((nscales, nsteps + 1, K, dim))
    yt = []
    Jyt = []
    if isinstance(y0, list):    # y0 itself is a list
        for ii in range(nscales):
            yt.append(np.zeros((nsteps+1, y0[ii].shape[0], dim)))
            yt[ii][0] = y0[ii]
            if withJacobian:
                Jyt.append(np.zeros((nsteps+1, y0[ii].shape[0], 1)))
            else:
                Jyt = None
    else:
        # y0 will be repeated
        for ii in range(nscales):
            yt.append(np.zeros((nsteps+1, y0.shape[0], dim)))
            yt[ii][0] = y0
            if withJacobian:
                Jyt.append(np.zeros((nsteps+1, y0.shape[0], 1)))
            else:
                Jyt = None

    nrep = 20
    for t in range(nsteps):
        for ii in range(nscales):
            t1 = nsteps-t-1
            ynew = yt[ii][t]
            yold = ynew
            Jnew = Jyt[ii][t]
            for k in range(nrep):
                for jj in range(nbase_scales):
                    ynew = yt[ii][t] - timeStep*KparDiff[scales[ii]][base_scales[jj]].applyK(xt[jj][t1], at[jj][t1], firstVar=ynew)
                    if withJacobian:
                        Jnew = Jyt[ii][t] - timeStep*KparDiff[scales[ii]][base_scales[jj]].applyDivergence(xt[jj][t1], at[jj][t1], firstVar=ynew)
                if np.abs(ynew - yold).max() < 1e-3:
                    break
                else:
                    if k == nrep - 1:
                        logging.info(f'Unable to compute exact inverse; error = {np.abs(ynew - yold).max():.04f}')
                    yold = ynew
            yt[ii][t+1] = ynew
            Jyt[ii][t+1] = Jnew
    output = dict()
    output['yt'] = yt
    output['Jyt'] = Jyt

    return output


def mslandmarkDirectEvolutionEuler(x0, at, KparDiff, affine=None, options=None, T=1.0):
    # Kernel is of size nbase_scale-by-nbase_scale
    _options = {'withJacobian': False, 'withNormals': None, 'withPointSet': None}
    if options is not None:
        for k in options.keys():
            _options[k] = options[k]

    withJacobian = _options['withJacobian']
    withNormals = _options['withNormals']
    withPointSet = _options['withPointSet']
    nscales = len(KparDiff)
    N = np.zeros(nscales).astype('int')
    dim = x0[0].shape[1]
    for ii in range(nscales):
        N[ii] = x0[ii].shape[0]
    nsteps = at[0].shape[0]
    timeStep = T / nsteps
    xt = []
    for ii in range(nscales):
        xt.append(np.zeros((nsteps+1, N[ii], dim)))
        xt[ii][0] = x0[ii]
    if withJacobian:
        Jt = []
        for ii in range(nscales):
            Jt.append(np.zeros((nsteps+1, N[ii], 1)))
    else:
        Jt = None

    if withPointSet is not None:
        # K = withPointSet.shape[0]
        yt = []
        for ii in range(nscales):
            yt.append(np.zeros((nsteps+1, withPointSet[ii].shape[0], dim)))
            y0 = withPointSet[ii]
            yt[ii][0] = y0
        if withJacobian:
            Jyt = []
            for ii in range(nscales):
                Jyt.append(np.zeros((nsteps+1, withPointSet[ii].shape[0],1)))
        else:
            Jyt = None
    else:
        yt = None
        Jyt = None

    if withNormals is not None:
        nt = []
        for ii in range(nscales):
            nt.append(np.zeros((nsteps+1, N, dim)))
            nt[ii][0] = withNormals[ii]
    else:
        nt = None
    for t in range(nsteps):
        for ii in range(nscales):
            xt[ii][t+1] = xt[ii][t]
            for jj in range(nscales):
                xt[ii][t+1] += timeStep*(KparDiff[ii][jj].applyK(xt[jj][t], at[jj][t], firstVar=xt[ii][t]))

            if withPointSet is not None:
                yt[ii][t+1] = yt[ii][t]
                for jj in range(nscales):
                    yt[ii][t+1] += timeStep * (KparDiff[ii][jj].applyK(xt[jj][t], at[jj][t], firstVar=yt[ii][t]))
                    ##
                    ##
                    # CHECK THE FORMULA
                    ##
                    ##
                    if withJacobian:
                        Jyt[ii][t+1] += timeStep * (KparDiff[ii][jj].applyDivergence(xt[jj][t], at[jj][t], firstVar=yt[ii][t]))

            if withJacobian:
                Jt[ii][t+1] = Jt[ii][t]
                for ll in range(nscales):
                    Jt[ii][t+1] += timeStep*KparDiff[ii][ll].applyDivergence(xt[ll][t], at[ll][t], firstVar=xt[ii][t])

            if withNormals is not None:
                for ii in range(nscales):
                    # nt[ii][t+1] = nt[ii][t] - timeStep
                    pass

    output = dict()
    output['xt'] = xt
    output['Jt'] = Jt
    output['yt'] = yt
    output['Jyt'] = Jyt
    output['nt'] = nt

    return output


def mslandmarkHamiltonianCovector(x0, at, px1, KparDiff, regweight, affine=None, extraTerm=None, T=1.0):
    nscales = len(KparDiff)
    N = np.zeros(nscales).astype('int')
    for ii in range(nscales):
        N[ii] = x0[ii].shape[0]
    dim = x0[0].shape[1]
    nsteps = at[0].shape[0]
    timeStep = T / nsteps

    st = mslandmarkDirectEvolutionEuler(x0, at, KparDiff, affine=affine, T=T)
    xt = st['xt']
    pxt = []
    px1_ = []
    for ii in range(nscales):
        pxt.append(np.zeros((nsteps+1, N[ii], dim)))
        px1_.append(np.zeros(pxt[ii].shape))

    if px1[0].ndim == 2:
        for ii in range(nscales):
            # px1_.append(np.zeros(pxt[ii].shape))
            # import pdb
            # pdb.set_trace()
            px1_[ii][-1] = px1[ii]
    else:
        px1_ = px1
    for ii in range(nscales):
        pxt[ii][nsteps] = px1_[ii][nsteps]
    for t in range(nsteps):
        for ii in range(nscales):
            zpx = np.zeros((pxt[ii].shape[1], pxt[ii].shape[2]))
            for jj in range(nscales):
                # import pdb
                # pdb.set_trace()
                if ii == jj:
                    zpx += KparDiff[ii][jj].applyDiffKT(xt[ii][nsteps-t-1], pxt[ii][nsteps-t],
                                                        at[jj][nsteps-t-1], regweight=regweight, lddmm=True)
                else:

                    zpx += KparDiff[ii][jj].applyDiffKT(xt[jj][nsteps-t-1], pxt[ii][nsteps-t], at[jj][nsteps-t-1],
                                                        firstVar=xt[ii][nsteps-t-1], regweight=regweight)
                    zpx += KparDiff[ii][jj].applyDiffKT(xt[jj][nsteps-t-1], at[ii][nsteps-t-1], pxt[jj][nsteps-t],
                                                        firstVar=xt[ii][nsteps-t-1], regweight=regweight)
                    zpx -= 2*KparDiff[ii][jj].applyDiffKT(xt[jj][nsteps-t-1], at[ii][nsteps-t-1], at[jj][nsteps-t-1],
                                                        firstVar=xt[ii][nsteps-t-1], regweight=regweight)
            pxt[ii][nsteps - t - 1] = pxt[ii][nsteps-t] + timeStep * zpx
            pxt[ii][nsteps - t - 1] += px1_[ii][nsteps - t - 1]
    return pxt, xt


# Computes gradient after covariant evolution for deformation cost a^TK(x,x) a
def mslandmarkHamiltonianGradient(x0, at, px1, KparDiff, regweight, getCovector=False, affine=None,
                                extraTerm=None, euclidean=False, T=1.0):
    (pxt, xt) = mslandmarkHamiltonianCovector(x0, at, px1, KparDiff, regweight, affine=affine, extraTerm=extraTerm, T=T)
    nscales = len(KparDiff)
    dat = []
    for ii in range(nscales):
        dat.append(np.zeros(at[ii].shape))
    timeStep = T / at[0].shape[0]
    nSteps = at[0].shape[0]
    if affine is not None:
        A = affine[0]
        dA = np.zeros(affine[0].shape)
        db = np.zeros(affine[1].shape)
    else:
        dA = None
        db = None
    for k in range(nSteps):
        for ii in range(nscales):
            for jj in range(nscales):
                if euclidean:
                    dat[ii][k] += KparDiff[ii][jj].applyK(xt[jj][k], (2*at[jj][k]-pxt[jj][k+1]),
                                                               firstVar=xt[ii][k])
            # if extraTerm is not None:
            #     Lv = extraTerm['grad'](x, KparDiff.applyK(x, a), variables='phi')['phi']
            #     dat[ss, k, :, :] += extraTerm['coeff'] * Lv
            #     if euclidean:
            #         dat[ss, k, :, :]  = KparDiff.applyK(x, dat[ss, k, :, :])

    res = dict()
    res['dat'] = dat
    res['xt'] = xt
    res['pxt'] = pxt
    res['dA'] = dA
    res['db'] = db
    return res
