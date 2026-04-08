import numpy as np
import logging
import pyfftw
from numba import prange, jit, int64
from multiprocessing import cpu_count
from .fourierKernel import Kernel


pyfftw.config.NUM_THREADS = cpu_count()


class Diffeomorphism:
    def __init__(self, shape, options):
        self.init(shape, options)

    def diffeoOptions(self, options):
        self.options = dict()
        if 'dim' not in options.keys():
            logging.error('Options must include a dimension; pursuing with dim=2 -- no guarantees')
            self.options['dim'] = 2
        self.options['timeStep'] = .1
        self.options['KparDiff'] = None
        self.options['sigmaKernel'] = 6.5
        self.options['orderKernel'] = -1
        self.options['kernelSize'] = 50
        self.options['typeKernel'] = 'gauss'
        self.options['resol'] = (1, 1, 1)
        self.options['kernelNormalization'] = 1.
        self.options['maskMargin'] = 1
        for k in options.keys():
            self.options[k] = options[k]

    def init(self, shape, options):
        self.diffeoOptions(options)
        self.dim = self.options['dim']
        self.imShape = shape
        self.vfShape = [self.dim] + list(shape)
        self.timeStep = self.options['timeStep']
        self.sigmaKernel = self.options['sigmaKernel']/(np.array(self.options['resol'])).min()
        self.typeKernel = self.options['typeKernel']
        self.kernelNormalization = 1.
        self.maskMargin = 1
        self.nbSemi = 0
        if np.isscalar(self.options['resol']):
            self.options['resol'] = (self.options['resol'],) * self.dim
        self.resol = self.options['resol']

        if self.options['KparDiff'] is None:
            self.KparDiff = Kernel(name=self.typeKernel, sigma=self.sigmaKernel,
                                   order=self.options['orderKernel'],
                                   dim=self.dim)
        else:
            self.KparDiff = self.options['KparDiff']
        self.KparDiff.init_fft(self.imShape)
        # self.param = param
        self.phi = np.zeros(self.vfShape)
        self.psi = np.zeros(self.vfShape)
        self._phi = np.zeros(self.vfShape)
        self._psi = np.zeros(self.vfShape)
        self.KShape = self.KparDiff.K.shape
        self.fftShape = ()
        for i in range(self.dim):
            self.fftShape += (self.KShape[i] + self.imShape[i] - 1,)
        ax = np.arange(self.dim, dtype=int)
        self.mask, self.dmask = makeMask(self.options['maskMargin'], self.imShape)
        self.fft_in = pyfftw.empty_aligned(self.fftShape,dtype='complex128')
        self.fft_out = pyfftw.empty_aligned(self.fftShape,dtype='complex128')
        self.ifft_out = pyfftw.empty_aligned(self.fftShape,dtype='complex128')
        self.fft_fwd = pyfftw.FFTW(self.fft_in, self.fft_out, axes=ax)
        self.fft_bwd = pyfftw.FFTW(self.fft_out, self.ifft_out, direction='FFTW_BACKWARD', axes=ax)
        self.periodic = False
        self.seqPadK = ()
        self.seqPadI = ()
        self.seqPadOne = ()

        for i in range(self.dim):
            # if not self.periodic:
            #     delta1 = self.KShape[i] - 1
            # else:
            #     delta1 = 0
            # if delta1 < self.imShape[i]:
            #     newShape[i] = imShape[i] + delta1
            # else:
            #     newShape[i] = 2 * delta1
            self.seqPadK += ((0, self.imShape[i] - 1),)
            self.seqPadI += ((self.KShape[i] // 2, self.KShape[i] // 2),)
            # seqPadI += ((0, K.shape[i] -1),)
            self.seqPadOne += ((self.KShape[i] - 1, 0),)
        newK = np.pad(self.KparDiff.K, self.seqPadK)
        self.fK = pyfftw.empty_aligned(self.fftShape, dtype='complex128')
        self.fft_in[...] = newK
        self.fK[...] = self.fft_fwd()

    def kernel_(self, Lv):
        res0 = np.zeros(Lv.shape)
        for i in range(Lv.shape[0]):
            self.fft_in[...] = np.pad(Lv[i,...]*self.mask[i,...], self.seqPadI)
            newOnes = np.pad(np.ones(self.imShape, dtype=bool), self.seqPadOne)
            fI = pyfftw.empty_aligned((self.fftShape), dtype='complex128')
            fI[...] = self.fft_fwd()
            self.fft_out[...] = self.fK * fI
            newRes = pyfftw.empty_aligned((self.fftShape))
            newRes[...] = np.real(self.fft_bwd())
            res0[i,...] = self.mask[i,...]*newRes[newOnes].reshape(self.imShape)
        return res0

    def kernel(self, Lv):
        return self.KparDiff.ApplyToVectorField(Lv, mask=self.mask)

    def initFlow(self):
        self._phi = idMesh(self.imShape)
        self._psi = idMesh(self.imShape)

    def updateFlow(self, Lv, dt):
        id = idMesh(self.imShape)
        v = self.kernel(Lv)
        res = (v*Lv).sum()
        semi = dt*v
        for jj in range(self.nbSemi):
            foo = id - semi/2
            semi = dt * multilinInterpVectorField(v, foo)
        foo = id - semi
        self._psi = multilinInterpVectorField(self._psi, foo)
        foo = id + semi
        self._phi = multilinInterpVectorField(foo, self._phi)
        # foo = id + dt*semi
        # self._phi = multilinInterpVectorField(self._phi, foo)
        # foo = id - dt * semi
        # self._psi = multilinInterpVectorField(foo, self._psi)

        return res

    def adjoint(self, v, w):
        gradv = differential(v, self.resol)
        gradw = differential(w, self.resol)
        Dvw = (gradv * w[None, ...]).sum(axis=1)
        Dwv = (gradw * v[None, ...]).sum(axis=1)
        return Dvw-Dwv

    def adjoint2(self, v, w):
        id = idMesh(v.shape[1:])
        foo1 = id + v
        foo2 = id - v
        gradv = differential(foo1, self.resol)
        foo = (gradv * w[None, ...]).sum(axis=1)
        res = multilinInterp(foo, foo2)
        return res

    def adjointStar(self, v, m):
        gradv = differential(v, self.resol)
        res = (gradv * m[:, None, ...]).sum(axis=0)
        for i in range(v.shape[0]):
            for j in range(v.shape[0]):
                foo = v[j,...] * m[i,...]
                foo1 = (np.roll(foo, -1, axis=j) - np.roll(foo, 1, axis=j))/(2*self.resol[j])
                res[i,...] += foo1
        return res

    def adjointStar2(self, v, m):
        id = idMesh(v.shape[1:])
        foo1 = id + v
        foo2 = id - v
        foo = np.zeros(m.shape)
        for i in range(v.shape[0]):
            foo[i,...] = multilinInterpDual(m[i,...], foo1)
        gradv = differential(foo2, self.resol)
        res = (gradv * foo[:, None, ...]).sum(axis=0)
        return res

    def big_adjoint(self, v, phi):
        dphi = inverseDifferential(phi, self.resol)
        Z = multilinInterp(v, phi)
        res = np.zeros(v.shape)
        for i in range(v.shape[0]):
            for j in range(v.shape[0]):
                res[i,...] += dphi[i,j,...] * Z[j,...]
        return res

    def big_adjointStar(self, m, phi):
        dphi = differential(phi, self.resol)
        jac = jacobianDeterminant(phi, self.resol)
        Z = multilinInterp(m, phi)
        res = np.zeros(m.shape)
        for i in range(m.shape[0]):
            for j in range(m.shape[0]):
                res[i,...] += dphi[j,i,...] * Z[j]
        return res * jac[None, ...]

    def GeodesicDiffeoEvolution(self, Lv, delta=1., accuracy=1., Tmax = 40, verb=True, nb_semi=3):
            v = self.kernel(Lv)
            M = delta * np.sqrt((v**2).sum(axis=0)).max()
            T = int(np.ceil(accuracy * M + 1))
            Lvt = Lv.copy()
            vt = v.copy()
            norm = np.sqrt( (vt*Lvt).sum())
                # foo = adjointStar(vt, Lvt)
                # foo2 = kernel.ApplyToVectorField(foo, mask=mask)
                # T = np.ceil(accuracy * (foo*foo2).sum() / (norm * norm + 1) + 1); * /
            if T > Tmax:
                T = Tmax

            id = idMesh(Lv.shape[1:])
            dt = delta / T
            if verb:
                logging.info(f'Evolution; T = {T:} {norm:.4f}')

            # self._phi = idMesh(self.imShape)
            # self._psi = idMesh(self.imShape)
            for t in range(T):
                semi = dt*vt
                for jj in range(nb_semi):
                    semi = dt*multilinInterpVectorField(vt, id-semi/2)

                if t > 0:
                    self._phi = multilinInterpVectorField(id+semi, self._phi)
                    self._psi = multilinInterpVectorField(self._psi, id-semi)
                else:
                    self._phi = id+semi
                    self._psi = id - semi

                Lvt = self.adjointStar2(semi, Lvt)
                vt = self.kernel(Lvt)
                norm2 = np.sqrt( (vt*Lvt).sum())

                if norm2 > 1e-6:
                    vt *= norm / norm2
                    Lvt *= norm / norm2

            return Lvt



def idMesh(shape, normalize=False, reverse = False):
    dim = len(shape)
    if dim==1:
        res = np.mgrid[0:shape[0]]
        if reverse:
            res = shape[0] - 1 - res
    elif dim == 2:
        x, y = np.mgrid[0:shape[0], 0:shape[1]]
        res = np.zeros((2, shape[0], shape[1]))
        if reverse:
            x = shape[0] - 1 - x
            y = shape[1] - 1 - y
        res[0, ...] = x
        res[1, ...] = y
    elif dim==3:
        x,y,z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        if reverse:
            x = shape[0] - 1 - x
            y = shape[1] - 1 - y
            z = shape[1] - 1 - z
        res = np.zeros((3,shape[0], shape[1], shape[2]))
        res[0,...] = x
        res[1,...] = y
        res[2,...] = z
    else:
        logging.error('No idMesh in dimension larger than 3')
        return
    if normalize:
        for i in range(res.shape[0]):
            res[i,...]/=shape[i]
    return res


def plateau(N,a,b, with_diff=False):
    t = np.linspace(0,1,N)
    f = np.ones(N)
    f[t<a] = 0
    J = np.logical_and(t>=a, t<b)
    u = (t[J] - a)/(b-a)
    f[J] = 3 * u**2 - 2* u**3
    J = np.logical_and(t>=1-b, t<=1-a)
    u = (1-a-t[J] )/(b-a)
    f[J] = 3 * u**2 - 2* u**3
    f[t>1-a] = 0

    if with_diff:
        df = np.zeros(N)
        J = np.logical_and(t >= a, t < b)
        u = (t[J] - a)/(b-a)
        df[J] = (6 * u - 6 * u**2) / (b-a)
        J = np.logical_and(t >= 1-b, t <= 1-a)
        u = (1-a-t[J] )/(b-a)
        df[J] = (-6 * u + 6 * u**2) / (b-a)
        df[t>1-a] = 0
    else:
        df = None

    if with_diff:
        return f, df
    else:
        return f


def makeMask(margin, S, Neumann=True, periodic = False):
    dim = len(S)
    delta = 0.05

    mask0 = np.ones((dim,) + tuple(S))
    dmask0 = np.zeros((dim,) + tuple(S))
    if periodic:
        return mask0, dmask0

    if Neumann:
        m = []
        dm = []
        for k in range(dim):
            u, du = plateau(S[k], margin / S[k], margin / S[k] + delta, with_diff=True)
            m.append(u)
            dm.append(du)

        # mask = m[0]
        # for k in range(1,dim):
        #     mask = mask[..., None] * m[k][None, ...]
        # for k in range(dim):
        #     mask0[k,...] = mask
        for k in range(dim):
            if k==0:
                mask = m[0]
                dmask = dm[0]
            else:
                mask = np.ones(S[0])
                dmask = m[0]
            for l in range(1,dim):
                if k == l:
                    mask = mask[..., None] * m[k][None, ...]
                    dmask = dmask[..., None] * dm[k][None, ...]
                else:
                    ml = np.ones(S[l])
                    mask = mask[..., None] * ml[None, ...]
                    dmask = dmask[..., None] * ml[None, ...]
            mask0[k, ...] = np.copy(mask)
            dmask0[k,...] = np.copy(dmask)
            #
            # if k != 0:
            #     mask = np.ones(S[0])
            #     dmask = np.zeros(S[0])
            # else:
            #     mask, dmask = plateau(S[0], margin / S[0], margin / S[0] + delta, with_diff = True)
            # for l in range(1, dim):
            #     if l != k:
            #         u = np.ones(S[l])
            #         du = np.zeros(S[l])
            #     else:
            #         u, du = plateau(S[k], margin / S[k], margin / S[k] + delta, with_diff = True)
            #     mask = mask[..., None] * u[None, ...]
            #     dmask = dmask[..., None] * du[None, ...]
            # mask0[k, ...] = mask
    else:
        #mask, dmask = plateau(S[0], margin / S[0], margin / S[0] + delta)
        m = []
        dm = []
        for k in range(dim):
            u, du = plateau(S[k], margin / S[k], margin / S[k] + delta, with_diff=True)
            m.append(u)
            dm.append(du)

        mask = m[0]
        for k in range(dim):
            mask = mask[..., None] * m[k][None, ...]
        for k in range(dim):
            mask0[k,...] = mask
        for k in range(dim):
            if k==0:
                dmask = dm[0]
            else:
                dmask = m[0]
            for l in range(1,dim):
                if k == l:
                    dmask = dmask[..., None] * dm[l][None, ...]
                else:
                    dmask = dmask[..., None] * m[l][None, ...]
            dmask0[k,...] = np.copy(dmask)

    return mask0, dmask0


@jit(nopython=True)
def interpolationMatrix(diffeo, targetShape):
    ## target shape has length ndim
    ## diffeo is either a dense vector field with ndim+1 dimensions or a list of ndim-dimensional points
    ## of size ndim x npoints
    ndim = len(targetShape)
    if ndim > 3:
        logging.error('interpolate only in dimensions 1 to 3')
        return


    tooLarge = diffeo.min() < 0
    for k in range(ndim):
        if diffeo[k, ...].max() > targetShape[k]-1:
            tooLarge = True
    if tooLarge:
        #print "min", diffeo.min()
        dfo = np.copy(diffeo)
        dfo = np.maximum(dfo, 0)
        for k in range(ndim):
            dfo[k, ...] = np.minimum(dfo[k, ...], targetShape[k]-1)
    else:
        dfo = np.copy(diffeo)

    ## only possible when diffeo is a vector field
    if diffeo.ndim > 2:
        dfo = np.reshape(dfo, (dfo.shape[0], np.prod(np.array(dfo.shape[1:]))))

    I = np.zeros(dfo.shape, dtype=int64)
    I[...] = np.floor(dfo)
    J = np.zeros(dfo.shape, dtype=int64)
    r = np.zeros(dfo.shape)
    for k in range(ndim):
        J[k, ...] = np.minimum(I[k, ...]+1, targetShape[k]-1)
        r[k, ...] = dfo[k, ...] - I[k, ...]
    #res = np.zeros(img.shape)
    res = np.zeros(dfo.shape[1])
    N = dfo.shape[1] * 2**ndim
    I0 = np.zeros(N, dtype = int64)
    J0 = np.zeros(N, dtype = int64)
    val = np.zeros(N)

    if ndim ==1:
        n = 0
        for k in range(I.shape[1]):
            I0[n, 0] = k
            J0[n, 0] = I[0,k]
            val[n] = (1-r[0, k])
            n += 1
            I0[n, 0] = k
            J0[n, 0] = J[0,k]
            val[n] = r[0, k]
            n += 1
    elif ndim==2:
        n = 0
        m1 = targetShape[1]
        for k in prange(I.shape[1]):
            I0[n] = k
            J0[n] = I[0,k] * m1 + I[1,k]
            val[n] = (1-r[1, k]) * (1-r[0, k])
            n += 1
            I0[n] = k
            J0[n] = J[0,k] * m1 + I[1,k]
            val[n] = (1-r[1, k]) * r[0, k]
            n += 1
            I0[n] = k
            J0[n] = I[0,k] * m1 + J[1,k]
            val[n] = r[1, k] * (1-r[0, k])
            n += 1
            I0[n] = k
            J0[n] = J[0,k] * m1 + J[1,k]
            val[n] = r[1, k] * r[0, k]
            n += 1
    elif ndim==3:
        n = 0
        m1 = targetShape[1] * targetShape[2]
        m2 = targetShape[2]
        for k in prange(I.shape[1]):
            I0[n] = k
            J0[n] = I[0,k] * m1 + I[1, k] * m2 + I[2, k]
            val[n] = (1-r[0,k]) * (1-r[1, k]) * (1-r[2, k])
            n += 1
            I0[n] = k
            J0[n] = J[0,k] * m1 + I[1, k] * m2 + I[2, k]
            val[n] = r[0,k] * (1-r[1, k]) * (1-r[2, k])
            n += 1
            I0[n] = k
            J0[n] = I[0,k] * m1 + J[1, k] * m2 + I[2, k]
            val[n] = (1-r[0,k]) * r[1, k] * (1-r[2, k])
            n += 1
            I0[n] = k
            J0[n] = I[0,k] * m1 + I[1, k] * m2 + J[2, k]
            val[n] = (1-r[0,k]) * (1-r[1, k]) * r[2, k]
            n += 1
            I0[n] = k
            J0[n] = J[0,k] * m1 + J[1, k] * m2 + I[2, k]
            val[n] = r[0,k] * r[1, k] * (1-r[2, k])
            n += 1
            I0[n] = k
            J0[n] = J[0,k] * m1 + I[1, k] * m2 + J[2, k]
            val[n] = r[0,k] * (1-r[1, k]) * r[2, k]
            n += 1
            I0[n] = k
            J0[n] = I[0,k] * m1 + J[1, k] * m2 + J[2, k]
            val[n] = (1-r[0,k]) * r[1, k] * r[2, k]
            n += 1
            I0[n] = k
            J0[n] = J[0,k] * m1 + J[1, k] * m2 + J[2, k]
            val[n] = r[0,k] * r[1, k] * r[2, k]
            n += 1

    return I0, J0, val


@jit(nopython=True)
def multilinInterp(img, diffeo):
    ndim = img.ndim
    targetShape = img.shape
    imgf = np.ravel(img)

    I, J, val = interpolationMatrix(diffeo, targetShape)
    if diffeo.ndim > 2:
        N = int(np.prod(np.array(diffeo.shape[1:])))
    else:
        N = diffeo.shape[1]
    res = np.zeros(N)
    for k in range(I.shape[0]):
        res[I[k]] += val[k] * imgf[J[k]]
    res = res.reshape(diffeo.shape[1:])

    return res


@jit(nopython=True)
def multilinInterpDual(img, diffeo):
    ndim = img.ndim
    targetShape = img.shape
    imgf = np.ravel(img)

    I, J, val = interpolationMatrix(diffeo, targetShape)
    if diffeo.ndim > 2:
        N = int(np.prod(np.array(diffeo.shape[1:])))
    else:
        N = diffeo.shape[1]
    res = np.zeros(N)
    for k in range(I.shape[0]):
        res[J[k]] += val[k] * imgf[I[k]]
    res = res.reshape(diffeo.shape[1:])

    return res


# multilinear interpolation
@jit(nopython=True, parallel=True)
def multilinInterp_(img, diffeo):
    int64 = "int64"
    ndim = img.ndim
    if ndim > 3:
        logging.error('interpolate only in dimensions 1 to 3')
        return

    #print diffeo.shape
    tooLarge = diffeo.min() < 0
    for k in range(img.ndim):
        if diffeo[k, ...].max() > img.shape[k]-1:
            tooLarge = True
    if tooLarge:
        #print "min", diffeo.min()
        dfo = np.copy(diffeo)
        dfo = np.maximum(dfo, 0)
        for k in range(img.ndim):
            dfo[k, ...] = np.minimum(dfo[k, ...], img.shape[k]-1)
    else:
        dfo = np.copy(diffeo)

    if diffeo.ndim > ndim:
        dfo = np.reshape(dfo, (dfo.shape[0], np.prod(np.array(dfo.shape[1:]))))
    I = np.zeros(dfo.shape, dtype=int64)
    I[...] = np.floor(dfo)
    J = np.zeros(dfo.shape, dtype=int64)
    r = np.zeros(dfo.shape)
    for k in range(ndim):
        J[k, ...] = np.minimum(I[k, ...]+1, img.shape[k]-1) 
        r[k, ...] = dfo[k, ...] - I[k, ...]
    #res = np.zeros(img.shape)
    res = np.zeros(dfo.shape[1])

    if ndim ==1:
        for k in range(I.shape[1]):
            res[k] = (1-r[0, k]) * img[I[0, k]] + r[0, k] * img[J[0, k]]
    elif ndim==2:
        for k in prange(I.shape[1]):
            res[k] = ((1-r[1, k]) * ((1-r[0, k]) * img[I[0, k], I[1,k]] + r[0, k] * img[J[0, k], I[1,k]])
                        + r[1, k] * ((1-r[0, k]) * img[I[0, k], J[1,k]] + r[0, k] * img[J[0, k], J[1,k]]))
    elif ndim==3:
        for k in prange(I.shape[1]):
            res[k] = ((1-r[2,k]) * ((1-r[1, k]) * ((1-r[0, k])
                                                          * img[I[0, k], I[1,k], I[2, k]]
                                                          + r[0, k] * img[J[0, k], I[1,k], I[2,k]])
                                    + r[1, k] * ((1-r[0, k])
                                                      * img[I[0, k], J[1,k], I[2, k]]
                                                      + r[0, k] * img[J[0, k], J[1,k], I[2,k]]))
                    + r[2,k] * ((1-r[1, k]) * ((1-r[0, k])
                                                         * img[I[0, k], I[1,k], J[2, k]]
                                                         + r[0, k] * img[J[0, k], I[1,k], J[2,k]])
                                + r[1, k] * ((1-r[0, k])
                                                  * img[I[0, k], J[1,k], J[2, k]]
                                                  + r[0, k] * img[J[0, k], J[1,k], J[2,k]])))
    else:
        print('interpolate only in dimensions 1 to 3')
        return

    res= res.reshape(diffeo.shape[1:])
    # if ndim ==1:
    #     for k in range(I.shape[1]):
    #         res[k] = (1-r[0, k]) * img[I[0, k]] + r[0, k] * img[J[0, k]]
    # elif ndim==2:
    #     for k in prange(I.shape[1]):
    #         for l in range(I.shape[2]):
    #             res[k,l] = ((1-r[1, k,l]) * ((1-r[0, k,l]) * img[I[0, k,l], I[1,k,l]] + r[0, k,l] * img[J[0, k,l], I[1,k,l]])
    #                     + r[1, k,l] * ((1-r[0, k,l]) * img[I[0, k,l], J[1,k,l]] + r[0, k,l] * img[J[0, k,l], J[1,k,l]]))
    # elif ndim==3:
    #     for k in prange(I.shape[1]):
    #         for l in range(I.shape[2]):
    #             for m in range(I.shape[3]):
    #                 res[k,l,m] = ((1-r[2,k,l, m]) * ((1-r[1, k,l, m]) * ((1-r[0, k,l, m])
    #                                                               * img[I[0, k,l, m], I[1,k,l, m], I[2, k,l, m]]
    #                                                               + r[0, k,l, m] * img[J[0, k,l, m], I[1,k,l, m], I[2,k,l, m]])
    #                                         + r[1, k,l, m] * ((1-r[0, k,l, m])
    #                                                           * img[I[0, k,l, m], J[1,k,l, m], I[2, k,l, m]]
    #                                                           + r[0, k,l, m] * img[J[0, k,l, m], J[1,k,l, m], I[2,k,l, m]]))
    #                         + r[2,k,l, m] * ((1-r[1, k,l, m]) * ((1-r[0, k,l, m])
    #                                                              * img[I[0, k,l, m], I[1,k,l, m], J[2, k,l, m]]
    #                                                              + r[0, k,l, m] * img[J[0, k,l, m], I[1,k,l, m], J[2,k,l, m]])
    #                                     + r[1, k,l, m] * ((1-r[0, k,l, m])
    #                                                       * img[I[0, k,l, m], J[1,k,l, m], J[2, k,l, m]]
    #                                                       + r[0, k,l, m] * img[J[0, k,l, m], J[1,k,l, m], J[2,k,l, m]])))
    # else:
    #     print('interpolate only in dimensions 1 to 3')
    #     return

    return res


# multilinear interpolation
@jit(nopython=True, parallel=True)
def multilinInterpDual_(img, diffeo):
    ndim = img.ndim
    if ndim > 3:
        print('interpolate only in dimensions 1 to 3')
        return

    #print diffeo.shape
    tooLarge = diffeo.min() < 0
    for k in range(img.ndim):
        if diffeo[k, ...].max() > img.shape[k]-1:
            tooLarge = True
    if tooLarge:
        #print "min", diffeo.min()
        dfo = np.copy(diffeo)
        dfo = np.maximum(dfo, 0)
        for k in range(img.ndim):
            dfo[k, ...] = np.minimum(dfo[k, ...], img.shape[k]-1)
    else:
        dfo = np.copy(diffeo)


    I = np.zeros(dfo.shape, dtype=int64)
    I[...] = np.floor(dfo)
    J = np.zeros(dfo.shape, dtype=int64)
    r = np.zeros(dfo.shape)
    for k in range(ndim):
        J[k, ...] = np.minimum(I[k, ...]+1, img.shape[k]-1)
        r[k, ...] = dfo[k, ...] - I[k, ...]

    res = np.zeros(img.shape)
    if ndim ==1:
        for k in range(I.shape[1]):
            res[I[0,k]] += (1-r[0, k]) * img[k]
            res[J[0,k]] += r[0, k] * img[k]
    elif ndim==2:
        for k in range(I.shape[1]):
            for l in range(I.shape[2]):
                res[I[0,k,l], I[1,k,l]] += (1-r[1, k,l]) * (1-r[0, k,l]) * img[k,l]
                res[J[0,k,l], I[1,k,l]] += (1-r[1, k,l]) * r[0, k,l] * img[k,l]
                res[I[0,k,l], J[1,k,l]] += (1-r[0, k,l]) * r[1, k,l] * img[k,l]
                res[J[0,k,l], J[1,k,l]] += r[1, k,l] * r[0, k,l] * img[k,l]
        # res[I[0,...], I[1,...]] += (1-r[1, ...]) * (1-r[0, ...]) * img
        # res[J[0,...], I[1,...]] += (1-r[1, ...]) * r[0, ...] * img
        # res[I[0,...], J[1,...]] += (1-r[0, ...]) * r[1, ...] * img
        # res[J[0,...], J[1,...]] += r[1, ...] * r[0, ...] * img
    elif ndim==3:
        for k in range(I.shape[1]):
            for l in range(I.shape[2]):
                for m in range(I.shape[3]):
                    res[I[0,k,l,m], I[1,k,l,m], I[2,k,l,m]] += (1-r[0, k,l,m]) * (1-r[1, k,l,m]) * (1-r[2, k,l,m]) * img[k,l,m]
                    res[J[0,k,l,m], I[1,k,l,m], I[2,k,l,m]] += r[0, k,l,m] * (1-r[1, k,l,m]) * (1-r[2, k,l,m]) * img[k,l,m]
                    res[I[0,k,l,m], J[1,k,l,m], I[2,k,l,m]] += (1-r[0, k,l,m]) * r[1, k,l,m] * (1-r[2, k,l,m]) * img[k,l,m]
                    res[I[0,k,l,m], I[1,k,l,m], J[2,k,l,m]] += (1-r[0, k,l,m]) * (1-r[1, k,l,m]) * r[2, k,l,m] * img[k,l,m]
                    res[J[0,k,l,m], J[1,k,l,m], I[2,k,l,m]] += r[0, k,l,m] * r[1, k,l,m] * (1-r[2, k,l,m]) * img[k,l,m]
                    res[J[0,k,l,m], I[1,k,l,m], J[2,k,l,m]] += r[0, k,l,m] * (1-r[1, k,l,m]) * r[2, k,l,m] * img[k,l,m]
                    res[I[0,k,l,m], J[1,k,l,m], J[2,k,l,m]] += (1-r[0, k,l,m]) * r[1, k,l,m] * r[2, k,l,m] * img[k,l,m]
                    res[J[0,k,l,m], J[1,k,l,m], J[2,k,l,m]] += r[0, k,l,m] * r[1, k,l,m] * r[2, k,l,m] * img[k,l,m]
    else:
        print('interpolate dual only in dimensions 1 to 3')
        return

    res = np.zeros(img.shape)
    if ndim ==1:
        for k in range(I.shape[1]):
            res[I[0,k]] += (1-r[0, k]) * img[k]
            res[J[0,k]] += r[0, k] * img[k]
    elif ndim==2:
        for k in prange(I.shape[1]):
            for l in range(I.shape[2]):
                res[I[0,k,l], I[1,k,l]] += (1-r[1, k,l]) * (1-r[0, k,l]) * img[k,l]
                res[J[0,k,l], I[1,k,l]] += (1-r[1, k,l]) * r[0, k,l] * img[k,l]
                res[I[0,k,l], J[1,k,l]] += (1-r[0, k,l]) * r[1, k,l] * img[k,l]
                res[J[0,k,l], J[1,k,l]] += r[1, k,l] * r[0, k,l] * img[k,l]
        # res[I[0,...], I[1,...]] += (1-r[1, ...]) * (1-r[0, ...]) * img
        # res[J[0,...], I[1,...]] += (1-r[1, ...]) * r[0, ...] * img
        # res[I[0,...], J[1,...]] += (1-r[0, ...]) * r[1, ...] * img
        # res[J[0,...], J[1,...]] += r[1, ...] * r[0, ...] * img
    elif ndim==3:
        for k in prange(I.shape[1]):
            for l in range(I.shape[2]):
                for m in range(I.shape[3]):
                    res[I[0,k,l,m], I[1,k,l,m], I[2,k,l,m]] += (1-r[0, k,l,m]) * (1-r[1, k,l,m]) * (1-r[2, k,l,m]) * img[k,l,m]
                    res[J[0,k,l,m], I[1,k,l,m], I[2,k,l,m]] += r[0, k,l,m] * (1-r[1, k,l,m]) * (1-r[2, k,l,m]) * img[k,l,m]
                    res[I[0,k,l,m], J[1,k,l,m], I[2,k,l,m]] += (1-r[0, k,l,m]) * r[1, k,l,m] * (1-r[2, k,l,m]) * img[k,l,m]
                    res[I[0,k,l,m], I[1,k,l,m], J[2,k,l,m]] += (1-r[0, k,l,m]) * (1-r[1, k,l,m]) * r[2, k,l,m] * img[k,l,m]
                    res[J[0,k,l,m], J[1,k,l,m], I[2,k,l,m]] += r[0, k,l,m] * r[1, k,l,m] * (1-r[2, k,l,m]) * img[k,l,m]
                    res[J[0,k,l,m], I[1,k,l,m], J[2,k,l,m]] += r[0, k,l,m] * (1-r[1, k,l,m]) * r[2, k,l,m] * img[k,l,m]
                    res[I[0,k,l,m], J[1,k,l,m], J[2,k,l,m]] += (1-r[0, k,l,m]) * r[1, k,l,m] * r[2, k,l,m] * img[k,l,m]
                    res[J[0,k,l,m], J[1,k,l,m], J[2,k,l,m]] += r[0, k,l,m] * r[1, k,l,m] * r[2, k,l,m] * img[k,l,m]
    else:
        print('interpolate dual only in dimensions 1 to 3')
        return
    return res


def multilinInterpVectorField(v, diffeo):
    res = np.zeros(v.shape)
    for k in range(v.shape[0]):
        res[k,...] = multilinInterp(v[k,...], diffeo)
    return res

@jit(nopython=True)
def multilinInterpGradient(img, diffeo):
    int64 = "int64"
    ndim = img.ndim
    if ndim > 3:
        print('interpolate only in dimensions 1 to 3')
        return

    #print diffeo.shape
    tooLarge = diffeo.min() < 0
    for k in range(img.ndim):
        if diffeo[k, ...].max() > img.shape[k]-1:
            tooLarge = True
    if tooLarge:
        dfo = np.copy(diffeo)
        dfo = np.maximum(dfo, 0)
        for k in range(img.ndim):
            dfo[k, ...] = np.minimum(dfo[k, ...], img.shape[k]-1)
    else:
        dfo = np.copy(diffeo)

    if diffeo.ndim > ndim:
        dfo = np.reshape(dfo, (dfo.shape[0], np.prod(np.array(dfo.shape[1:]))))

    I = np.zeros(dfo.shape, dtype=int64)
    I[...] = np.floor(dfo)
    J = np.zeros(dfo.shape, dtype=int64)
    r = np.zeros(dfo.shape)
    for k in range(ndim):
        J[k, ...] = np.minimum(I[k, ...]+1, img.shape[k]-1)
        r[k, ...] = dfo[k, ...] - I[k, ...]

#    if tooLarge:
#        print "too large"
#    print I.min(), I.max(), J.min(), J.max()

    if ndim ==1:
        res = np.zeros(I.shape)
        for k in range(I.shape[1]):
            res[0,k] = img[J[0, k]] - img[I[0, k]]
    elif ndim==2:
        res = np.zeros(I.shape)
        for k in prange(I.shape[1]):
            res[0,k] = ((1-r[1, k]) * (-img[I[0, k], I[1,k]] + img[J[0, k], I[1,k]])
                    + r[1, k] * (-img[I[0, k], J[1,k]] + img[J[0, k], J[1,k]]))
            res[1, k] = (- ((1-r[0, k]) * img[I[0, k], I[1,k]] + r[0, k] * img[J[0, k], I[1,k]])
                    + ((1-r[0, k]) * img[I[0, k], J[1,k]] + r[0, k] * img[J[0, k], J[1,k]]))
    elif ndim==3:
        #res = np.zeros(np.insert(img.shape, 0, 3))
        res = np.zeros(I.shape)
        for k in prange(I.shape[1]):
            # for l in range(I.shape[2]):
            #     for m in range(I.shape[3]):
            res[0,k] = ((1-r[2,k]) * ((1-r[1, k]) * (-img[I[0, k], I[1,k], I[2, k]] + img[J[0, k], I[1,k], I[2,k]])
                                    + r[1, k] * (-img[I[0, k], J[1,k], I[2, k]] + img[J[0, k], J[1,k], I[2,k]]))
                    + r[2,k] * ((1-r[1, k]) * (- img[I[0, k], I[1,k], J[2, k]] + img[J[0, k], I[1,k], J[2,k]])
                               + r[1, k] * (-img[I[0, k], J[1,k], J[2, k]] + img[J[0, k], J[1,k], J[2,k]])))
            res[1, k] = ((1-r[2,k]) * (-((1-r[0, k]) * img[I[0, k], I[1,k], I[2, k]] + r[0, k] * img[J[0, k], I[1,k], I[2,k]])
                                    + ((1-r[0, k]) * img[I[0, k], J[1,k], I[2, k]] + r[0, k] * img[J[0, k], J[1,k], I[2,k]]))
                    + r[2,k] * (-((1-r[0, k]) * img[I[0, k], I[1,k], J[2, k]] + r[0, k] * img[J[0, k], I[1,k], J[2,k]])
                                + ((1-r[0, k]) * img[I[0, k], J[1,k], J[2, k]] + r[0, k] * img[J[0, k], J[1,k], J[2,k]])))
            res[2, k] = (-((1-r[1, k]) * ((1-r[0, k]) * img[I[0, k], I[1,k], I[2, k]] + r[0, k] * img[J[0, k], I[1,k], I[2,k]])
                            + r[1, k] * ((1-r[0, k]) * img[I[0, k], J[1,k], I[2, k]] + r[0, k] * img[J[0, k], J[1,k], I[2,k]]))
                    + ((1-r[1, k]) * ((1-r[0, k]) * img[I[0, k], I[1,k], J[2, k]] + r[0, k] * img[J[0, k], I[1,k], J[2,k]])
                        + r[1, k] * ((1-r[0, k]) * img[I[0, k], J[1,k], J[2, k]] + r[0, k] * img[J[0, k], J[1,k], J[2,k]])))
    else:
        print('interpolate only in dimensions 1 to 3')
        return

    res = res.reshape(diffeo.shape)

    # if ndim ==1:
    #     res = np.zeros(I.shape)
    #     for k in range(I.shape[1]):
    #         res[0,k] = img[J[0, k]] - img[I[0, k]]
    # elif ndim==2:
    #     res = np.zeros(I.shape)
    #     for k in prange(I.shape[1]):
    #         for l in range(I.shape[2]):
    #             res[0,k,l] = ((1-r[1, k,l]) * (-img[I[0, k,l], I[1,k,l]] + img[J[0, k,l], I[1,k,l]])
    #                     + r[1, k,l] * (-img[I[0, k,l], J[1,k,l]] + img[J[0, k,l], J[1,k,l]]))
    #             res[1, k,l] = (- ((1-r[0, k,l]) * img[I[0, k,l], I[1,k,l]] + r[0, k,l] * img[J[0, k,l], I[1,k,l]])
    #                     + ((1-r[0, k,l]) * img[I[0, k,l], J[1,k,l]] + r[0, k,l] * img[J[0, k,l], J[1,k,l]]))
    # elif ndim==3:
    #     #res = np.zeros(np.insert(img.shape, 0, 3))
    #     res = np.zeros(I.shape)
    #     for k in prange(I.shape[1]):
    #         for l in range(I.shape[2]):
    #             for m in range(I.shape[3]):
    #                 res[0,k,l,m] = ((1-r[2,k,l,m]) * ((1-r[1, k,l,m]) * (-img[I[0, k,l,m], I[1,k,l,m], I[2, k,l,m]] + img[J[0, k,l,m], I[1,k,l,m], I[2,k,l,m]])
    #                                         + r[1, k,l,m] * (-img[I[0, k,l,m], J[1,k,l,m], I[2, k,l,m]] + img[J[0, k,l,m], J[1,k,l,m], I[2,k,l,m]]))
    #                         + r[2,k,l,m] * ((1-r[1, k,l,m]) * (- img[I[0, k,l,m], I[1,k,l,m], J[2, k,l,m]] + img[J[0, k,l,m], I[1,k,l,m], J[2,k,l,m]])
    #                                    + r[1, k,l,m] * (-img[I[0, k,l,m], J[1,k,l,m], J[2, k,l,m]] + img[J[0, k,l,m], J[1,k,l,m], J[2,k,l,m]])))
    #                 res[1, k,l,m] = ((1-r[2,k,l,m]) * (-((1-r[0, k,l,m]) * img[I[0, k,l,m], I[1,k,l,m], I[2, k,l,m]] + r[0, k,l,m] * img[J[0, k,l,m], I[1,k,l,m], I[2,k,l,m]])
    #                                         + ((1-r[0, k,l,m]) * img[I[0, k,l,m], J[1,k,l,m], I[2, k,l,m]] + r[0, k,l,m] * img[J[0, k,l,m], J[1,k,l,m], I[2,k,l,m]]))
    #                         + r[2,k,l,m] * (-((1-r[0, k,l,m]) * img[I[0, k,l,m], I[1,k,l,m], J[2, k,l,m]] + r[0, k,l,m] * img[J[0, k,l,m], I[1,k,l,m], J[2,k,l,m]])
    #                                     + ((1-r[0, k,l,m]) * img[I[0, k,l,m], J[1,k,l,m], J[2, k,l,m]] + r[0, k,l,m] * img[J[0, k,l,m], J[1,k,l,m], J[2,k,l,m]])))
    #                 res[2, k,l,m] = (-((1-r[1, k,l,m]) * ((1-r[0, k,l,m]) * img[I[0, k,l,m], I[1,k,l,m], I[2, k,l,m]] + r[0, k,l,m] * img[J[0, k,l,m], I[1,k,l,m], I[2,k,l,m]])
    #                                 + r[1, k,l,m] * ((1-r[0, k,l,m]) * img[I[0, k,l,m], J[1,k,l,m], I[2, k,l,m]] + r[0, k,l,m] * img[J[0, k,l,m], J[1,k,l,m], I[2,k,l,m]]))
    #                         + ((1-r[1, k,l,m]) * ((1-r[0, k,l,m]) * img[I[0, k,l,m], I[1,k,l,m], J[2, k,l,m]] + r[0, k,l,m] * img[J[0, k,l,m], I[1,k,l,m], J[2,k,l,m]])
    #                             + r[1, k,l,m] * ((1-r[0, k,l,m]) * img[I[0, k,l,m], J[1,k,l,m], J[2, k,l,m]] + r[0, k,l,m] * img[J[0, k,l,m], J[1,k,l,m], J[2,k,l,m]])))
    # else:
    #     print('interpolate only in dimensions 1 to 3')
    #     return
    #
    return res


def multilinInterpGradientVectorField(src, diffeo):
    res0 = np.zeros([src.shape[0]] + list(src.shape))
    for k in range(src.shape[0]):
        res0[k,...] = multilinInterpGradient(src[k,...], diffeo)
    return res0


# Computes gradient
#@jit(nopython=True, parallel=True)
def imageGradient(img, resol=None):
    res = None
    if img.ndim > 3:
        print('gradient only in dimensions 1 to 3')

    if img.ndim == 3:
        if resol is None:
            resol = (1.,1.,1.)
        res = np.zeros((3,img.shape[0], img.shape[1], img.shape[2]))
        res[0,1:img.shape[0]-1, :, :] = (img[2:img.shape[0], :, :] - img[0:img.shape[0]-2, :, :])/(2*resol[0])
        res[0,0, :, :] = (img[1, :, :] - img[0, :, :])/(resol[0])
        res[0,img.shape[0]-1, :, :] = (img[img.shape[0]-1, :, :] - img[img.shape[0]-2, :, :])/(resol[0])
        res[1,:, 1:img.shape[1]-1, :] = (img[:, 2:img.shape[1], :] - img[:, 0:img.shape[1]-2, :])/(2*resol[1])
        res[1,:, 0, :] = (img[:, 1, :] - img[:, 0, :])/(resol[1])
        res[1,:, img.shape[1]-1, :] = (img[:, img.shape[1]-1, :] - img[:, img.shape[1]-2, :])/(resol[1])
        res[2,:, :, 1:img.shape[2]-1] = (img[:, :, 2:img.shape[2]] - img[:, :, 0:img.shape[2]-2])/(2*resol[2])
        res[2,:, :, 0] = (img[:, :, 1] - img[:, :, 0])/(resol[2])
        res[2,:, :, img.shape[2]-1] = (img[:, :, img.shape[2]-1] - img[:, :, img.shape[2]-2])/(resol[2])
    elif img.ndim ==2:
        if resol is None:
            resol = np.array((1.,1.))
        res = np.zeros((2,img.shape[0], img.shape[1]))
        res[0,1:img.shape[0]-1, :] = (img[2:img.shape[0], :] - img[0:img.shape[0]-2, :])/(2*resol[0])
        res[0,0, :] = (img[1, :] - img[0, :])/(resol[0])
        res[0,img.shape[0]-1, :] = (img[img.shape[0]-1, :] - img[img.shape[0]-2, :])/(resol[0])
        res[1,:, 0] = (img[:, 1] - img[:, 0])/(resol[1])
        res[1,:, img.shape[1]-1] = (img[:, img.shape[1]-1] - img[:, img.shape[1]-2])/(resol[1])
        res[1,:, 1:img.shape[1]-1] = (img[:, 2:img.shape[1]] - img[:, 0:img.shape[1]-2])/(2*resol[1])
    elif img.ndim ==1:
        if resol is None:
            resol = 1
        res = np.zeros(img.shape[0])
        res[1:img.shape[0]-1] = (img[2:img.shape[0]] - img[0:img.shape[0]-2])/(2*resol)
        res[0] = (img[1] - img[0])/(resol)
        res[img.shape[0]-1] = (img[img.shape[0]-1] - img[img.shape[0]-2])/(resol)
    return res

# Computes Jacobian determinant
#@jit(nopython=True)
def jacobianDeterminant(diffeo, resol=(1.,1.,1.), periodic=False):
    if diffeo.ndim > 4:
        print('No jacobian in dimension larger than 3')
        return

    if diffeo.ndim == 4:
        if periodic == True:
            w = np.mgrid[0:diffeo.shape[1], 0:diffeo.shape[2], 0:diffeo.shape[3]]
            dw = diffeo-w
            for k in range(3):
                diffeo[k,:,:,:] -= np.rint(dw[k,:,:,:]/diffeo.shape[k+1])*diffeo.shape[k+1]
        grad = np.zeros([3,3,diffeo.shape[1], diffeo.shape[2], diffeo.shape[3]])
        grad[0,:,:,:,:] = imageGradient(np.squeeze(diffeo[0,:,:,:]), resol=resol)
        grad[1,:,:,:,:] = imageGradient(np.squeeze(diffeo[1,:,:,:]), resol=resol)
        grad[2,:,:,:,:] = imageGradient(np.squeeze(diffeo[2,:,:,:]), resol=resol)
        res = np.fabs(grad[0,0,:,:,:] * grad[1,1,:,:,:] * grad[2,2,:,:,:]
                        - grad[0,0,:,:,:] * grad[1,2,:,:,:] * grad[2,1,:,:,:]
                        - grad[0,1,:,:,:] * grad[1,0,:,:,:] * grad[2,2,:,:,:]
                        - grad[0,2,:,:,:] * grad[1,1,:,:,:] * grad[2,0,:,:,:]
                        + grad[0,1,:,:,:] * grad[1,2,:,:,:] * grad[2,0,:,:,:] 
                        + grad[0,2,:,:,:] * grad[1,0,:,:,:] * grad[2,1,:,:,:])
    elif diffeo.ndim == 3:
        if periodic == True:
            w = np.mgrid[0:diffeo.shape[1], 0:diffeo.shape[2]]
            dw = diffeo-w
            for k in range(2):
                diffeo[k,:,:] -= np.rint(dw[k,:,:]/diffeo.shape[k+1])*diffeo.shape[k+1]
        grad = np.zeros([2,2,diffeo.shape[1], diffeo.shape[2]])
        grad[0,:,:,:] = imageGradient(np.squeeze(diffeo[0,:,:]), resol=resol)
        grad[1,:,:,:] = imageGradient(np.squeeze(diffeo[1,:,:]), resol=resol)
        res = np.fabs(grad[0,0,:,:] * grad[1,1,:,:] - grad[0,1,:,:] * grad[1,0,:,:])
    else:
        if periodic == True:
            w = np.mgrid[0:diffeo.shape[0]]
            dw = diffeo-w
            diffeo -= np.rint(dw/diffeo.shape[0])*diffeo.shape[0]
        res =  np.fabs(imageGradient(np.squeeze(diffeo)), resol=resol)
    return res

# Computes differential
@jit(nopython=True)
def jacobianMatrix(diffeo, resol=(1.,1.,1.), periodic=False):
    if diffeo.ndim > 4:
        print('No jacobian in dimension larger than 3')
        return

    if diffeo.ndim == 4:
        if periodic == True:
            w = np.mgrid[0:diffeo.shape[1], 0:diffeo.shape[2], 0:diffeo.shape[3]]
            dw = diffeo-w
            for k in range(3):
                diffeo[k,:,:,:] -= np.rint(dw[k,:,:,:]/diffeo.shape[k+1])*diffeo.shape[k+1]
        grad = np.zeros([3,3,diffeo.shape[1], diffeo.shape[2], diffeo.shape[3]])
        grad[0,:,:,:,:] = imageGradient(np.squeeze(diffeo[0,:,:,:]), resol=resol)
        grad[1,:,:,:,:] = imageGradient(np.squeeze(diffeo[1,:,:,:]), resol=resol)
        grad[2,:,:,:,:] = imageGradient(np.squeeze(diffeo[2,:,:,:]), resol=resol)
    elif diffeo.ndim == 3:
        if periodic == True:
            w = np.mgrid[0:diffeo.shape[1], 0:diffeo.shape[2]]
            dw = diffeo-w
            for k in range(2):
                diffeo[k,:,:] -= np.rint(dw[k,:,:]/diffeo.shape[k+1])*diffeo.shape[k+1]
        grad = np.zeros([2,2,diffeo.shape[1], diffeo.shape[2]])
        grad[0,:,:,:] = imageGradient(np.squeeze(diffeo[0,:,:]), resol=resol)
        grad[1,:,:,:] = imageGradient(np.squeeze(diffeo[1,:,:]), resol=resol)
    else:
        if periodic == True:
            w = np.mgrid[0:diffeo.shape[0]]
            dw = diffeo-w
            diffeo -= np.rint(dw/diffeo.shape[0])*diffeo.shape[0]
        grad =  np.fabs(imageGradient(np.squeeze(diffeo)), resol=resol)
    return grad


def differential(v, resol):
    res = np.zeros([v.shape[0]] + list(v.shape))
    for t in range(v.shape[0]):
        res[t,...] = imageGradient(v[t,...], resol)
    return res

def differentialDual(v, resol):
    res = np.zeros([v.shape[0],] + list(v.shape))
    deter = resol.prod()
    for t in range(v.shape[0]):
        res[t, ...] = imageGradient(v[t,...], resol)/(resol[t]*deter)

def inverseDifferential(v, resol):
    res = np.zeros([v.shape[0],] + list(v.shape))
    grad = differential(v, resol)

    if v.shape[0] == 1:
        res = 1 / (grad + 0.00000000001)
    elif v.shape[0] == 2:
        jac = grad[0, 0, ...] * grad[1, 1, ...] - grad[1, 0, ...] * grad[0, 1, ...] + 0.00000000001
        res[0, 0, ...] = grad[1, 1, ...] / jac
        res[0, 1, ...] = - grad[0, 1, ...] / jac
        res[1, 0, ...] = - grad[1, 0, ...] / jac
        res[1, 1, ...] = grad[0, 0, ...] / jac
    elif v.shape[0] == 3:
        jac = grad[0, 0, :, :, :] * grad[1, 1, :, :, :] * grad[2, 2, :, :, :]\
                - grad[0, 0, :, :, :] * grad[1, 2, :, :, :] * grad[2, 1, :, :, :]\
                - grad[0, 1, :, :, :] * grad[1, 0, :, :, :] * grad[2, 2, :, :, :]\
                - grad[0, 2, :, :, :] * grad[1, 1, :, :, :] * grad[2, 0, :, :, :]\
                + grad[0, 1, :, :, :] * grad[1, 2, :, :, :] * grad[2, 0, :, :, :]\
                + grad[0, 2, :, :, :] * grad[1, 0, :, :, :] * grad[2, 1, :, :, :] + 1e-10
        res[0, 0, ...] = (grad[1, 1, ...] * grad[2, 2, ...] - grad[1, 2, ...] * grad[2, 1, ...]) / jac
        res[1, 1, ...] = (grad[0, 0, ...] * grad[2, 2, ...] - grad[0, 2, ...] * grad[2, 0, ...]) / jac
        res[2, 2, ...] = (grad[1, 1, ...] * grad[0, 0, ...] - grad[1, 0, ...] * grad[0, 1, ...]) / jac
        res[0, 1, ...] = -(grad[0, 1, ...] * grad[2, 2, ...] - grad[2, 1, ...] * grad[0, 2, ...]) / jac
        res[1, 0, ...] = -(grad[1, 0, ...] * grad[2, 2, ...] - grad[1, 2, ...] * grad[2, 0, ...]) / jac
        res[0, 2, ...] = (grad[0, 1, ...] * grad[1, 2, ...] - grad[0, 2, ...] * grad[1, 1, ...]) / jac
        res[2, 0, ...] = (grad[1, 0, ...] * grad[2, 1, ...] - grad[2, 0, ...] * grad[1, 1, ...]) / jac
        res[1, 2, ...] = -(grad[0, 0, ...] * grad[1, 2, ...] - grad[0, 2, ...] * grad[1, 0, ...]) / jac
        res[2, 1, ...] = -(grad[0, 0, ...] * grad[2, 1, ...] - grad[2, 0, ...] * grad[0, 1, ...]) / jac
    else:
        logging.error("no inverse in dimension higher than 3")
        return

    return res


def inverseMap(phi, resol, psi0=None):
    flag = 1
    d = phi.shape[1:]
    if psi0 is None:
        id = idMesh(d)
    psi = psi0.copy()
    id = idMesh(d)
    foo = id - multilinInterpVectorField(phi, psi)
    error = np.sqrt( (foo**2).sum()) / d.prod()
    for k in range(10):
        dpsi = inverseDifferential(foo, resol)
        psiTry = psi + dpsi
        foo = id - multilinInterpVectorField(phi, psi)
        errorTry = np.sqrt( (foo**2).sum()) / d.prod()
        logging.info(f"inversion error {error: 0.4f} {errorTry: 0.4f}")
        if errorTry < error:
            psi = psiTry
            error = errorTry
        else:
            break
    return psi


def laplacian(X, neumann=0):
    Xplus = np.zeros(np.array(X.shape) + 2)
    if X.ndim == 1:
        n1 = X.shape[0]
        Xplus[1:n1+1] = X
        if neumann:
            Xplus[0] = Xplus[1]
            Xplus[n1+1] = Xplus[n1]
        Y = Xplus[0:n1] + Xplus[2:n1+2] - 2*X
    elif X.ndim == 2:
        n1 = X.shape[0]
        n2 = X.shape[1]
        Xplus[1:n1+1, 1:n2+1] = X
        if neumann:
            Xplus[0,:] = Xplus[1,:]
            Xplus[:,0] = Xplus[:,1]
            Xplus[n1+1,:] = Xplus[n1,:]
            Xplus[:,n2+1] = Xplus[:,n2]
        Y = (Xplus[1:n1+1, 0:n2] + Xplus[1:n1+1, 2:n2+2] + 
            Xplus[0:n1, 1:n2+1] + Xplus[2:n1+2, 1:n2+1]) - 4*X
    elif X.ndim == 3:
        n1 = X.shape[0]
        n2 = X.shape[1]
        n3 = X.shape[2]
        Xplus[1:n1+1, 1:n2+1, 1:n3+1] = X
        if neumann:
            Xplus[0,:,:] = Xplus[1,:,:]
            Xplus[:,0,:] = Xplus[:,1,:]
            Xplus[:,:,0] = Xplus[:,:,1]
            Xplus[n1+1,:, :] = Xplus[n1,:, :]
            Xplus[:,n2+1, :] = Xplus[:,n2, :]
            Xplus[:,:,n3+1] = Xplus[:,:,n3]
        Y = (Xplus[0:n1, 1:n2+1, 1:n3+1] + Xplus[2:n1+2, 1:n2+1, 1:n3+1] + 
            Xplus[1:n1+1, 0:n2, 1:n3+1] + Xplus[1:n1+1, 2:n2+2, 1:n3+1] +
            Xplus[1:n1+1, 1:n2+1, 0:n3] + Xplus[1:n1+1, 1:n2+1, 2:n3+2]) - 6*X
    else:
        Y = None
        logging.error('Laplacian in dim less than 3 only')
        
    return Y



class DiffeoParam:
    def __init__(self, dim, timeStep=.1, KparDiff=None, sigmaKernel=6.5, order=-1,
                 kernelSize=50, typeKernel='gauss', resol=(1, 1, 1)):
        self.timeStep = timeStep
        self.sigmaKernel = sigmaKernel
        self.typeKernel = typeKernel
        self.kernelNormalization = 1.
        self.maskMargin = 1
        self.dim = dim
        self.resol = resol
        if KparDiff is None:
            self.KparDiff = Kernel(name=self.typeKernel, sigma=self.sigmaKernel,
                                   order=order, dim=dim)
        else:
            self.KparDiff = KparDiff
        self.KparDiff.init_fft(self.imShape)


