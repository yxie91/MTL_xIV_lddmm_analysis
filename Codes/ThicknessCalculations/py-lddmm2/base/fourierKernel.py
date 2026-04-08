import numpy as np
import logging
from scipy.signal import fftconvolve, convolve
import pyfftw
from multiprocessing import cpu_count
pyfftw.config.NUM_THREADS = cpu_count()


c_ = np.array([[1, 0, 0, 0, 0],
               [1, 1, 0, 0, 0],
               [1, 1, 1/3, 0, 0],
               [1, 1, 0.4, 1/15, 0],
               [1, 1, 3/7, 2/21, 1/105]])


c1_ = np.array([[0, 0, 0, 0],
                [1, 0, 0, 0],
                [1/3, 1/3, 0, 0],
                [1/5, 1/5, 1/15, 0],
                [1/7, 1/7, 2/35, 1/105]])


c2_ = np.array([[0, 0, 0],
               [0, 0, 0],
               [1/3, 0, 0],
               [1/15, 1/15, 0],
               [1/35, 1/35, 1/105]])


# Functions for kernels using Fourier transforms
def convolve_fftw(img, K, periodic=False, fK=None):
    fftShape = ()
    for i in range(K.ndim):
        if not periodic:
            delta1 = K.shape[i] - 1
        else:
            delta1 = 0
        if delta1 < img.shape[i]:
            fftShape += (img.shape[i] + delta1,)
        else:
            fftShape += (2 * delta1,)

    seqPadI = ()
    seqPadOne = ()
    ax = np.arange(img.ndim, dtype=int)
    for i in range(K.ndim):
        seqPadI += ((K.shape[i] // 2, K.shape[i] // 2),)
        # seqPadI += ((0, K.shape[i] -1),)
        seqPadOne += ((K.shape[i] - 1, 0),)

    fft_in = pyfftw.empty_aligned(fftShape, dtype='complex128')
    fft_out = pyfftw.empty_aligned(fftShape, dtype='complex128')
    ifft_out = pyfftw.empty_aligned(fftShape, dtype='complex128')
    fft_fwd = pyfftw.FFTW(fft_in, fft_out, axes=ax)
    fft_bwd = pyfftw.FFTW(fft_out, ifft_out, direction='FFTW_BACKWARD', axes=ax)

    if fK is None:
        seqPadK = ()
        for i in range(K.ndim):
            seqPadK += ((0, img.shape[i] - 1),)
        newK = np.pad(K, seqPadK)
        fK = pyfftw.empty_aligned(fftShape, dtype='complex128')
        fft_in[...] = newK
        fK[...] = fft_fwd()

    fft_in[...] = np.pad(img, seqPadI, mode='symmetric')
    newOnes = np.pad(np.ones(img.shape, dtype=bool), seqPadOne)
    fI = pyfftw.empty_aligned(fftShape, dtype='complex128')
    fI[...] = fft_fwd()
    fft_out[...] = fK * fI
    newRes = pyfftw.empty_aligned(fftShape)
    newRes[...] = np.real(fft_bwd())
    newRes = np.real(newRes[newOnes].reshape(img.shape))
    return newRes


def convolve_(img, K):
    # newres = fftconvolve(img, K, 'same')
    newres = convolve(img, K, 'same')
    return newres


class Kernel:
    def __init__(self, dim=3, sigma=1.0, order=1, name='gauss', normalize=False,
                 fftw=True):
        self.dim = dim
        self.sigma = sigma
        self.order = order
        self.fftw = fftw
        self.size = int(self.getSize(name, order, sigma))
        size = self.size
        logging.info(f'size kernel: {size:}')
        self.type = name
        self.fK = None
        self.fDiffK = (None,)*dim
        # self.ApplyToImage = self.ApplyToImage_fftw
        # self.convolve = convolve_fftw

        self.grid_ = self.getGrid()
        d2 = np.zeros(self.grid_[0].shape)
        for x in self.grid_:
            d2 += x**2
        d2 /= self.sigma**2
        d = np.sqrt(d2)
        self.d = d

        # If K = F(d), diffK = F'(d)/(d*sigma)
        if name == 'gauss':
            self.K = np.exp(-d2/2)
            self.diffK = -self.K/self.sigma**2
        elif name in ('laplacian', 'matern'):
            self.K = (c_[order, 0] + c_[order, 1] * d + c_[order, 2] * d2
                      + c_[order, 3] * d * d2 + c_[order, 4] * d2 * d2)*np.exp(-d)
            # self.diffK = -(c_[order, 0] + c_[order, 1] * d + c_[order, 2] * d2
            #                 + c_[order, 3] * d * d2 + c_[order, 4] * d2 * d2)/d \
            #              + (c_[order, 1] + 2*c_[order, 2] * d
            #                 + 3*c_[order, 3] * d2 + 4*c_[order, 4] * d * d2)/d
            self.diffK = -(c1_[order, 0] + c1_[order, 1] * d + c1_[order, 2] * d2
                           + c1_[order, 3] * d * d2) * np.exp(-d) / self.sigma**2
        else:
            logging.error('Unknown kernel')
            return

        if normalize:
            norm = self.K.sum()
            self.K /= norm
            self.diffK /= norm

    def init_fft(self, imShape):
        fftShape = ()
        for i in range(self.K.ndim):
            delta1 = self.K.shape[i] - 1
            if delta1 < imShape[i]:
                fftShape += (imShape[i] + delta1,)
            else:
                fftShape += (2 * delta1,)
        seqPadK = ()
        for i in range(self.K.ndim):
            delta1 = self.K.shape[i] - 1
            if delta1 < imShape[i]:
                seqPadK += ((0, imShape[i] - 1),)
            else:
                seqPadK += ((0, delta1-1),)

        seqPadI = ()
        seqPadOne = ()
        for i in range(self.K.ndim):
            delta1 = self.K.shape[i] - 1
            if delta1 < imShape[i]:
                seqPadI += ((self.K.shape[i] // 2, self.K.shape[i] // 2),)
                seqPadOne += ((self.K.shape[i] - 1, 0),)
            else:
                seqPadI += ((self.K.shape[i] - imShape[i]//2 - 1,
                             self.K.shape[i] - imShape[i] // 2 - 1),)
                seqPadOne += ((fftShape[i] - imShape[i], 0),)
        self.seqPadI = seqPadI
        self.seqPadOne = seqPadOne

        logging.info(f'{fftShape} {seqPadK}')
        newK = np.pad(self.K, seqPadK)
        self.fK = pyfftw.empty_aligned(fftShape, dtype='complex128')
        self.fft_in = pyfftw.empty_aligned(fftShape, dtype='complex128')
        self.fft_out = pyfftw.empty_aligned(fftShape, dtype='complex128')
        self.ifft_out = pyfftw.empty_aligned(fftShape, dtype='complex128')
        ax = np.arange(len(imShape), dtype=int)
        self.fft_fwd = pyfftw.FFTW(self.fft_in, self.fft_out, axes=ax, threads=16)
        self.fft_bwd = pyfftw.FFTW(self.fft_out, self.ifft_out, 
                                   direction='FFTW_BACKWARD', axes=ax, threads=16)

        self.fft_in[...] = newK
        self.fK[...] = self.fft_fwd()
        self.fftShape = fftShape

        self.fDiffK = ()
        for i in range(self.dim):
            newK = np.pad(self.diffK*self.grid_[i], seqPadK)
            self.fDiffK += (pyfftw.empty_aligned(fftShape, dtype='complex128'),)
            self.fft_in[...] = newK
            #fft_fwd = pyfftw.FFTW(self.fft_in, self.fft_out, axes=ax)
            self.fDiffK[i][...] = self.fft_fwd()
        self.imShape = imShape

    def convolve_fftw(self, img, diff=-1):
        #logging.info('fftw')
        self.fft_in[...] = np.pad(img, self.seqPadI, mode='symmetric')
        newOnes = np.pad(np.ones(img.shape, dtype=bool), self.seqPadOne)
        fI = pyfftw.empty_aligned(self.fftShape, dtype='complex128')
        fI[...] = self.fft_fwd()
        if diff < 0:
            self.fft_out[...] = self.fK * fI
        else:
            self.fft_out[...] = self.fDiffK[diff] * fI
        newRes = pyfftw.empty_aligned(self.fftShape)
        newRes[...] = np.real(self.fft_bwd())
        newRes = np.real(newRes[newOnes].reshape(img.shape))
        return newRes

    def convolve(self, img, K=None, diff=-1):
        if self.fftw:
            return self.convolve_fftw(img, diff=diff)
        else:
            return convolve_(img, K)

    def getGrid(self):
        size = 2*self.size+1
        if self.dim == 3:
            grid = np.mgrid[0:size, 0:size, 0:size]
        elif self.dim == 2:
            grid = np.mgrid[0:size, 0:size]
        elif self.dim == 1:
            grid = np.mgrid[0:size]
        else:
            grid = np.array(0)
            logging.error('Kernels in dimensions three or less')
        return grid - self.size

    def getSize(self, name, order, sigma, thresh=1e-2):
        t = np.linspace(0, 100, 10000)
        if name == 'gauss':
            K = np.exp(-t**2/2)
        if name in ('laplacian', 'matern'):
            K = (c_[order, 0] + c_[order, 1] * t + c_[order, 2] * t**2
                 + c_[order, 3] * t**3 + c_[order, 4] * t**4)*np.exp(-t)
        i = np.argmax(K.sum() - np.cumsum(K) < thresh*K.sum())
        return np.ceil(sigma*t[i])

    def ApplyToImage(self, img, mask=None):
        if mask is None:
            mask = np.ones(img.shape)
        res = mask * self.convolve(img*mask, self.K)
        return np.real(res)

    def ApplyDiffToImage(self, img, mask=None, dmask=None):
        if mask is None:
            mask = np.ones(img.shape)
            dmask = np.zeros((img.ndim,) + img.shape)
        else:
            if dmask is None:
                logging.warning('Kernel operations: missing mask differential')
                dmask = np.zeros((img.ndim,) + img.shape)
        res = np.zeros((img.ndim,) + img.shape)
        for i in range(img.ndim):
            res[i] = mask * self.convolve(img*mask, self.diffK*self.grid_[i],
                                          diff=i)\
                     + dmask[i] * self.convolve(img*mask, self.K)
        return np.real(res)

    def ApplyToVectorField(self, Lv, mask=None):
        if mask is None:
            mask = np.ones(Lv.shape)
        res = np.zeros(Lv.shape)
        for i in range(self.dim):
            res[i, ...] = mask[i, ...] * self.convolve(Lv[i, ...]*mask[i, ...],
                                                       self.K)
        return res

    def ApplyDivToVectorField(self, Lv, mask=None, dmask=None):
        # mask is a vector field whose ith coordinate only depends on the ith variable
        # dmask[i] is the derivative of mask[i] w.r.t. the ith coordinate
        if mask is None:
            mask = np.ones(Lv.shape)
            dmask = np.zeros(Lv.shape)
        else:
            if dmask is None:
                logging.warning('Kernel operations: missing mask differential')
                dmask = np.zeros(Lv.shape)

        res = np.zeros(Lv.shape[1:])
        for i in range(self.dim):
            res += mask[i, ...] * self.convolve(Lv[i, ...]*mask[i, ...],
                                                self.diffK*self.grid_[i],
                                                diff=i)
            res += dmask[i, ...] * self.convolve(Lv[i, ...]*mask[i, ...],
                                                 self.K)
        return res


