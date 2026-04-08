import numpy as np
from copy import deepcopy
import logging
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import time
from .gridscalars import GridScalars, saveImage
from .imageMatchingBase import ImageMatchingBase
from .diffeo import Kernel, multilinInterpVectorField, imageGradient, idMesh, multilinInterp, multilinInterpDual
from .bfgs import bfgs


class Control(dict):
    def __init__(self, u=None, Lv=None):
        super(Control, self).__init__()
        self['u'] = u
        self['Lv'] = Lv


class TopChange(ImageMatchingBase):
    def __init__(self, Template, Target, options=None):
        super(TopChange, self).__init__(Template=Template, Target=Target, 
                                        options=options)
        self.dt = 1/self.Tsize
        a_ = self.options['rMargin']
        p = 2 + self.options['pMargin'] + 2 * a_
        r = (2 * p) / (p - 2) + a_
        self.p = p
        self.r = r
        self.rstar = self.r / (self.r - 1)

        self.KparDiff.init_fft(self.im0.data.shape)
        self.diffusionKernel.init_fft(self.im0.data.shape)
        self.endpointKernel.init_fft(self.im0.data.shape)
        self.diffeoOnly = False
        self.topOnly = False


        self.TopCost = self.options['TopCost']
        self.endpointH1cost = self.options['endpointH1cost']
        self.sigmaDiffusion = self.options['sigmaSmooth'] * np.sqrt(self.dt)/self.h
        self.C = 1/self.options['sigmaError']**2
        self.a_ = 0.1
        self.mu_ = 0.1
        self.TOL = 1e-10
        self.width = 100
        self.inflection = 0.5
        self.iMargin = self.a_/2
        self.im1.data = self.Evolve({'u': np.zeros(self.control['u'].shape),
                                     'Lv': np.zeros(self.control['Lv'].shape)},
                                    im0=self.im1.data)[-1]

        # test0 = self.diffusionKernel.ApplyToImage_scipy(self.im0.data)
        # test1 = self.diffusionKernel.ApplyToImage_fftw(self.im0.data)
        # logging.info(f'Test fftw: {np.abs(test0-test1).max():.4f}')
        self.initial_plot()

    def set_parameters(self):
        super(ImageMatchingBase, self).set_parameters()
        self.originalTemplate = deepcopy(self.im0)
        self.originalTarget = deepcopy(self.im1)
        self.imHalfWidth = self.im0.data.shape[0] // 2
        self.h = 1/self.imHalfWidth
        self.hk = self.h * np.sqrt(self.options['diffeoCondition'])
        # self.hk = 1e-4
        # self.h = 2/self.imHalfWidth
        self.diffusionKernel = Kernel(name='gauss',
                                      sigma=self.options['sigmaSmooth']/(self.h*np.sqrt(self.Tsize)),
                                      dim=self.options['dim'], normalize=True)
        self.endpointKernel = Kernel(name='gauss',
                                     sigma=self.options['endpointSigmaSmooth']/self.h,
                                     dim=self.options['dim'], normalize=True)

    def initialize_variables(self):
        self.Tsize = self.options['Tsize']
        self.imDef = GridScalars(grid=self.im0)

        self.vfShape = [self.Tsize, self.dim] + list(self.im0.data.shape)
        self.topShape = [self.Tsize] + list(self.im0.data.shape)
        self.control = Control()
        self.controlTry = Control()
        self.v = np.zeros(self.vfShape)
        self.control['Lv'] = np.zeros(self.vfShape)
        self.control['u'] = np.zeros(self.topShape)
        self.controlTry['u'] = np.zeros(self.topShape)

    def getDefaultOptions(self):
        options = super().getDefaultOptions()
        options['rMargin'] = 0.01
        options['pMargin'] = 0.5
        options['TopCost'] = 1e3
        options['diffeoCondition'] = 1.
        options['endpointH1cost'] = 1.0
        options['endpointSigmaSmooth'] = 1.
        options['diffeoWarmup'] = 0
        options['alternate'] = 0
        # options['fidelityWeight'] = 0.
        return options

    def initial_save(self):
        if len(self.im0.data.shape) == 3:
            ext = '.vtk'
        else:
            ext = ''
        saveImage(255*self.originalTemplate.data,
                  self.outputDir + '/OriginalTemplate' + ext)
        saveImage(255*self.originalTarget.data,
                  self.outputDir + '/OriginalTarget' + ext)
        saveImage(255*self.im0.data,
                  self.outputDir + '/Template' + ext)
        saveImage(255*self.im1.data, self.outputDir + '/Target' + ext)
        saveImage(self.KparDiff.K, self.outputDir + '/Kernel' + ext,
                  normalize=True)
        saveImage(self.diffusionKernel.K,
                  self.outputDir + '/DiffusionKernel' + ext,
                  normalize=True)
        saveImage(self.mask.min(axis=0), self.outputDir + '/Mask' + ext,
                  normalize=True)

    def kernel(self, Lv):
        return self.KparDiff.ApplyToVectorField(Lv, self.mask) * self.hk**2

    def kernelDiv(self, Lv):
        return self.KparDiff.ApplyDivToVectorField(Lv, self.mask, self.dmask) * self.hk**2

    def dataTerm(self, IDef_, var=None):
        if var is None or 'I1' not in var:
            I1 = self.im1.data
        else:
            I1 = var['I1']
        diff = IDef_ - I1
        q = diff
        # q = self.endpointKernel.ApplyToImage(diff)
        dq = self.endpointKernel.ApplyDiffToImage(diff)
        normdq = np.sqrt((dq**2 + 1e-10).sum(axis=0))
        D = self.endpointH1cost*(normdq**self.rstar)
        L = np.abs(q)**self.rstar
        return self.C * (self.h**2) * (L+D).sum()

    def endPointGradient(self, endPoint=None):
        if endPoint is None:
            endPoint = self.imDef.data
        diff = endPoint - self.im1.data
        q = diff
        # q = self.endpointKernel.ApplyToImage(diff)
        dq = self.endpointKernel.ApplyDiffToImage(diff)
        normdq = np.sqrt((dq**2 + 1e-10).sum(axis=0))
        sdq = dq / normdq[None, ...]
        dq = -self.endpointH1cost*self.endpointKernel.ApplyDivToVectorField((normdq**(self.rstar - 1))*sdq)
        Lq = (np.abs(q)**(self.rstar - 1))*np.sign(q)
        # Lq = self.endpointKernel.ApplyToImage(Lq)
        Uq = self.C * (self.h**2)*self.rstar*(Lq+dq)
        return Uq

    def testEndpointGradient(self):
        dff = np.random.normal(size=self.imDef.data.shape)
        c = []
        eps0 = 1e-10
        for eps in [-eps0, eps0]:
            ff = deepcopy(self.imDef)
            ff.data += eps*dff
            c.append(self.dataTerm(ff.data))
        grd = self.endPointGradient()
        logging.info("test endpoint gradient: {0:.5f} {1:.5f}".format((c[1]-c[0])/(2*eps0), (grd*dff).sum()) )

    def testGradientFun(self, obj, grd, gradCoeff, opt=None, dotProduct=None):
        self.testEndpointGradient()
        self.test_controlGradient(self.control)
        dirfoo0 = self.randomDir()
        if self.diffeoOnly:
            dirfoo0['u'] *= 0
        else:
            dirfoo0['u'] /= np.sqrt(max((grd['u'] ** 2).sum(), 1e-10))
        if self.topOnly:
            dirfoo0['Lv'] *= 0
        else:
            dirfoo0['Lv'] /= np.sqrt(max((grd['Lv'] ** 2).sum(), 1e-10))
        dirfoo = deepcopy(dirfoo0)
        epsfoo = 1e-5
        objfoo1 = self.updateTry(dirfoo, epsfoo, obj - 1e10)
        [grdfoo] = self.dotProduct(grd, [dirfoo])
        objfoo2 = self.updateTry(dirfoo, -epsfoo, obj - 1e10)
        logging.info('Test Gradient: %.6f %.6f' % ((objfoo1 - objfoo2) / (2.0 * epsfoo), -grdfoo * gradCoeff))

        if not self.diffeoOnly:
            dirfoo['Lv'] *= 0
            objfoo1 = self.updateTry(dirfoo, epsfoo, obj - 1e10)
            [grdfoo] = self.dotProduct(grd, [dirfoo])
            objfoo2 = self.updateTry(dirfoo, -epsfoo, obj - 1e10)
            logging.info('Test Gradient (u): %.6f %.6f' % ((objfoo1 - objfoo2) / (2.0 * epsfoo), -grdfoo * gradCoeff))

        if not self.topOnly:
            dirfoo = deepcopy(dirfoo0)
            dirfoo['u'] *= 0
            objfoo1 = self.updateTry(dirfoo, epsfoo, obj - 1e10)
            [grdfoo] = self.dotProduct(grd, [dirfoo])
            objfoo2 = self.updateTry(dirfoo, -epsfoo, obj - 1e10)
            logging.info('Test Gradient (Lv): %.6f %.6f' % ((objfoo1 - objfoo2) / (2.0 * epsfoo), -grdfoo * gradCoeff))

    def ControlNorm(self, control, verb=True):
        u = control['u']
        Lv = control['Lv']
        u_r = np.zeros([self.Tsize, 1])
        vm = np.zeros([self.Tsize, 1])
        Velos = 1
        for k in range(self.Tsize):
            U = u[k, :, :]
            Lvk = Lv[k, :, :, :]
            u_r[k] = (np.abs(U)**self.r).sum()
            vk = self.kernel(Lvk)
            vm[k] = np.maximum((Lvk*vk).sum(), self.TOL)

        Rtop = (self.h ** (4/self.r)) * ((u_r**(self.p / self.r)).sum() * self.dt)**(2/self.p)
        Rdiff = (self.hk ** 2) * ((vm ** (self.p / 2)).sum()*self.dt)**(2/self.p)
        if verb:
            logging.info(f'Rtop: {Rtop:.4f}, Rdiff: {Rdiff:.4f}')

        R = self.TopCost * Rtop + Velos * Rdiff
        return R

    def ControlGradient(self, control):
        u = control['u']
        Lv = control['Lv']
        Ru = np.zeros(u.shape)
        Rv = np.zeros(Lv.shape)
        ur = np.zeros([self.Tsize, 1])
        vm = np.zeros([self.Tsize, 1])
        Velos = 1
        for k in range(self.Tsize):
            uk = u[k, :, :]
            Lvk = Lv[k, :, :, :]
            ur[k] = max((np.abs(uk)**self.r).sum(), 1e-16)
            Ru[k, :, :] = (ur[k]**(self.p/self.r-1)) * (np.abs(uk)**(self.r-2)) * uk
            vk = self.kernel(Lvk)
            lvv = (Lvk*vk).sum()
            if lvv < 0:
                logging.warning(f'Negative inner product vector field {lvv}')
            vm[k] = np.maximum(lvv, self.TOL)
            Rv[k, :, :, :] = (vm[k]**(self.p/2 - 1)) * vk

        #Ru *= self.dt
        #Rv *= self.dt
        res = {}
        res['u'] = 2 * self.TopCost * self.h**(4/self.r) \
            * max((ur**(self.p/self.r)).sum()*self.dt, 1e-16)**(2/self.p-1) * Ru
        res['Lv'] = 2 * Velos * (self.hk**2)\
            * max((vm**(self.p/2)).sum()*self.dt, 1e-16)**(2/self.p-1)*Rv
        return res

    def test_controlGradient(self, control):
        c2 = deepcopy(control)
        cng = self.ControlGradient(c2)
        eps = 1e-4
        du = np.random.normal(0, 1, size=c2['u'].shape) / np.sqrt(max((cng['u'] ** 2).sum(), 1e-10))
        dLv = np.random.normal(0, 1, size=c2['Lv'].shape) / np.sqrt(max((cng['Lv'] ** 2).sum(), 1e-10))
        c2['u'] = control['u'] + eps * du
        c2['Lv'] = control['Lv'] + eps * dLv
        cn1 = self.ControlNorm(c2, verb=True)
        c2['u'] = control['u'] - eps * du
        c2['Lv'] = control['Lv'] - eps * dLv
        cn2 = self.ControlNorm(c2)
        dc = Control()
        dc['u'] = du
        dc['Lv'] = dLv
        #check = (cng['u'] * du).sum() + (cng['Lv'] * dLv).sum()
        check = self.dotProduct(cng, [dc])[0]
        logging.info(f'test control norm: {(cn1-cn2)/(2*eps):.6f} {check:.6f}')

    def objectiveFun(self, control=None):
        if control is None:
            control = self.control

        R = self.ControlNorm(control)
        q = self.Evolve(control)
        q_final = q[-1]
        end = self.dataTerm(q_final)
        return R + end

    # def g_(self, x):
    #     g = 1/2 + self.mu_ * np.log(np.maximum(x+self.a_, self.TOL)
    #                                 / np.maximum(1+self.a_-x, self.TOL))
    #     return x
    #
    #
    # def dg_(self, x):
    #     return np.ones(x.shape)
    #
    # def ddg_(self, x):
    #     return np.zeros(x.shape)
    #
    # def ginv_(self, y):
    #     return y
    #
    # def dginv_(self, y):
    #     return np.ones(y.shape)

    def g_(self, x):
        g = 1/2 + self.mu_ * np.log(np.maximum(x+self.a_, self.TOL)
                                    / np.maximum(1+self.a_-x, self.TOL))
        return g
    
    def dg_(self, x):
        dg = (x+self.a_ >= self.TOL)*self.mu_/np.maximum(x+self.a_, self.TOL)\
             +(1+self.a_-x >= self.TOL)*self.mu_/np.maximum(1+self.a_-x, self.TOL)
        return dg

    def ddg_(self, x):
        ddg = - self.mu_/np.maximum(x+self.a_, self.TOL)**2 + self.mu_/np.maximum(1+ self.a_-x, self.TOL)**2
        return ddg

    def ginv_(self, y):
        z = np.maximum((y-1/2)/self.mu_, -20)
        ginv = (1+2*self.a_)/(1 + np.exp(-z)) - self.a_
        return ginv

    def dginv_(self, y):
        z = np.maximum((y - 1 / 2) / self.mu_, -20)
        dginv= ((1+2*self.a_)/(1+np.exp(-z))**2) * np.exp(-z)/self.mu_
        return dginv

    def reaction(self, q):
        C = (self.width) * (q ** 3) * ((1 - q) ** 3) * (q - self.inflection)
        return C

    def dreaction(self, q):
        C = 3 * (q ** 2) * ((1 - q) ** 3) * (q - self.inflection)\
            - 3 * (q ** 2) * ((1 - q) ** 2) * (q - self.inflection) \
            + (q ** 3) * ((1 - q) ** 3)
        return self.width*C

    def Evolve(self, control, im0=None):
        state = np.zeros([self.Tsize+1, self.imShape[0], self.imShape[1]])
        if im0 is None:
            state[0, :, :] = self.im0.data
        else:
            state[0, :, :] = im0

        eps = 1e-16*self.dt
        id = idMesh(self.imShape)
        for k in range(self.Tsize):
            qk = state[k, :, :]
            uk = control['u'][k, :, :]
            Lvk = control['Lv'][k, :, :, :]
            dfk = self.diffusionKernel.ApplyDiffToImage(qk)
            normdfk = np.sqrt((dfk**2).sum(axis=0) + eps)
            fk = self.diffusionKernel.ApplyToImage(qk)
            vk = self.kernel(Lvk)

            ck = self.reaction(fk)
            nuk = uk * normdfk + ck
            Im = multilinInterp(fk, id - self.dt * vk)
            fk2 = np.minimum(np.maximum(fk, -self.iMargin), 1+self.iMargin)
            muk = self.g_(Im) + self.dg_(fk2)*nuk*self.dt
            #muk = np.minimum(np.maximum(muk, -self.a_/2), 1.0+self.a_/2)
            state[k+1, :, :] = self.ginv_(muk)
        return state

    def Evolve_(self, control, im0=None):
        state = np.zeros([self.Tsize+1, self.imShape[0], self.imShape[1]])
        if im0 is None:
            state[0, :, :] = self.im0.data
        else:
            state[0, :, :] = im0

        eps = 1e-16*self.dt
        for k in range(self.Tsize):
            qk = state[k, :, :]
            uk = control['u'][k, :, :]
            Lvk = control['Lv'][k, :, :, :]
            dfk = self.diffusionKernel.ApplyDiffToImage(qk)
            normdfk = np.sqrt((dfk**2).sum(axis=0) + eps)
            fk = self.diffusionKernel.ApplyToImage(qk)
            vk = self.kernel(Lvk)

            ck = self.reaction(fk)
            nuk = uk * normdfk - (vk*dfk).sum(axis=0) + ck
            Im = deepcopy(fk)
            Im = np.minimum(np.maximum(Im, -self.iMargin), 1+self.iMargin)
            muk = self.g_(Im) + self.dg_(Im)*nuk*self.dt
            #muk = np.minimum(np.maximum(muk, -self.a_/2), 1.0+self.a_/2)
            state[k+1, :, :] = self.ginv_(muk)
        return state

    def EvolveStep(self, uk, Lvk, qk):
        eps = 1e-16*self.dt
        id = idMesh(self.imShape)
        dfk = self.diffusionKernel.ApplyDiffToImage(qk)
        normdfk = np.sqrt((dfk ** 2).sum(axis=0) + eps)
        fk = self.diffusionKernel.ApplyToImage(qk)
        vk = self.kernel(Lvk)

        ck = self.reaction(fk)
        nuk = uk * normdfk + ck
        Im = multilinInterp(fk, id - self.dt * vk)
#        fk2 = np.minimum(np.maximum(fk, -self.iMargin), 1 + self.iMargin)
        muk = self.g_(Im) + self.dg_(fk) * nuk * self.dt
        #muk = np.minimum(np.maximum(muk, -self.a_/2), 1.0+self.a_/2)
        return self.ginv_(muk)

    def testHStep(self, pk, uk, Lvk, qk):
        dqk = np.random.normal(0, 1, qk.shape)
        eps = 1e-8
        id = idMesh(self.imShape)
        c1 = (pk*self.EvolveStep(uk, Lvk, qk+eps*dqk)).sum()
        c2 = (pk*self.EvolveStep(uk, Lvk, qk-eps*dqk)).sum()

        dfk = self.diffusionKernel.ApplyDiffToImage(qk)
        fk = self.diffusionKernel.ApplyToImage(qk)
        ck = self.reaction(fk)
        dck = self.dreaction(fk)
#        fk = np.maximum(-self.iMargin, np.minimum(fk, 1 + self.iMargin))
        vk = self.kernel(Lvk)
        normdfk = ((dfk ** 2).sum(axis=0) + eps) ** (1 / 2)

        psik = id - self.dt * vk
        Imk = multilinInterp(fk, psik)
        nuk = uk * normdfk + ck
        muk = self.g_(Imk) + self.dt * self.dg_(fk) * nuk

        digk = self.dginv_(muk)
        rkpsi = self.dg_(Imk) * digk
        rk = self.dg_(fk) * digk
        drk = self.ddg_(fk) * digk

        vv = multilinInterpDual(rkpsi * pk, psik)
        uu = (rk * pk)[None, ...] * ((uk / normdfk)[None, ...] * dfk)
        dpk = self.diffusionKernel.ApplyToImage(vv + self.dt * (drk * nuk + rk * dck) * pk) \
               - self.dt * self.diffusionKernel.ApplyDivToVectorField(uu)

        check = (dpk*dqk).sum()
        logging.info(f'Test H step: {(c1-c2)/(2*eps):.6f}  {check: .6f}')

    def EvolveStep_(self, uk, Lvk, qk):
        eps = 1e-16*self.dt
        dfk = self.diffusionKernel.ApplyDiffToImage(qk)
        normdfk = np.sqrt((dfk ** 2).sum(axis=0) + eps)
        fk = self.diffusionKernel.ApplyToImage(qk)
        vk = self.kernel(Lvk)

        ck = self.reaction(fk)
        nuk = uk * normdfk - (vk * dfk).sum(axis=0) + ck
        Im = deepcopy(fk)
        Im = np.minimum(np.maximum(Im, -self.iMargin), 1 + self.iMargin)
        muk = self.g_(Im) + self.dg_(Im) * nuk * self.dt
        #muk = np.minimum(np.maximum(muk, -self.a_/2), 1.0+self.a_/2)
        return self.ginv_(muk)

    def testHStep_(self, pk, uk, Lvk, qk):
        dqk = np.random.normal(0, 1, qk.shape)
        eps = 1e-8
        id = idMesh(self.imShape)
        c1 = (pk*self.EvolveStep(uk, Lvk, qk+eps*dqk)).sum()
        c2 = (pk*self.EvolveStep(uk, Lvk, qk-eps*dqk)).sum()

        dfk = self.diffusionKernel.ApplyDiffToImage(qk)
        fk = self.diffusionKernel.ApplyToImage(qk)
        ck = self.reaction(fk)
        dck = self.dreaction(fk)
        fk = np.maximum(-self.iMargin, np.minimum(fk, 1 + self.iMargin))
        vk = self.kernel(Lvk)
        normdfk = ((dfk ** 2).sum(axis=0) + eps) ** (1 / 2)

        nuk = uk * normdfk - (vk * dfk).sum(axis=0) + ck
        muk = self.g_(fk) + self.dt * self.dg_(fk) * nuk
        digk = self.dginv_(muk)
        rk = self.dg_(fk) * digk
        drk = self.ddg_(fk) * digk

        vv = (rk * pk)[None, ...] * ((uk / normdfk)[None, ...] * dfk - vk)
        dpk = self.diffusionKernel.ApplyToImage((rk + self.dt * drk * nuk
                                                + self.dt * rk * dck) * pk) \
             - self.dt * self.diffusionKernel.ApplyDivToVectorField(vv)
        check = (dpk*dqk).sum()
        logging.info(f'Test H step: {(c1-c2)/(2*eps):.6f}  {check: .6f}')

    def Backwards(self, p_final, state, control):
        PullLam = Control(u=np.zeros(control['u'].shape),
                          Lv=np.zeros(control['Lv'].shape))
        p = np.zeros(state.shape)
        p[-1] = deepcopy(p_final)
        eps = 1e-16*self.dt
        id = idMesh(self.imShape)

        for k in range(self.Tsize-1, -1, -1):
            qk = state[k]
            uk = control['u'][k]
            Lvk = control['Lv'][k]

            # if self.options['mode'] == 'debug':
            #     self.testHStep(p[k+1], uk, Lvk, qk)
            #     self.testHStep_(p[k+1], uk, Lvk, qk)

            # Eikonal Time Step
            dfk = self.diffusionKernel.ApplyDiffToImage(qk)
            fk = self.diffusionKernel.ApplyToImage(qk)
            ck = self.reaction(fk)
            dck = self.dreaction(fk)
            fk = np.maximum(-self.iMargin, np.minimum(fk, 1+self.iMargin))
            vk = self.kernel(Lvk)
            normdfk = ((dfk**2).sum(axis=0)+eps)**(1/2)

            psik = id - self.dt * vk
            Imk = multilinInterp(fk, psik)
            nuk = uk * normdfk + ck
            muk = self.g_(Imk) + self.dt * self.dg_(fk)*nuk
            digk = self.dginv_(muk)
            rkpsi = self.dg_(Imk) * digk
            rk = self.dg_(fk) * digk
            drk = self.ddg_(fk) * digk

            PullLam['u'][k, :, :] = rk * normdfk * p[k+1]
            dvk = self.diffusionKernel.ApplyDiffToImage(qk)
            dvk = multilinInterpVectorField(dvk, psik)
            PullLam['Lv'][k] = - self.kernel((rkpsi * p[k+1])[None, ...] * dvk)

            vv = multilinInterpDual(rkpsi * p[k+1], psik)
            uu = (rk * p[k+1])[None, ...] * ((uk/normdfk)[None, ...] * dfk)
            p[k] = self.diffusionKernel.ApplyToImage(vv + self.dt * (drk * nuk + rk * dck) * p[k+1])\
                - self.dt * self.diffusionKernel.ApplyDivToVectorField(uu)
        PullLam['p'] = p
        return PullLam

    def Backwards_(self, p_final, state, control):
        PullLam = Control(u=np.zeros(control['u'].shape),
                          Lv=np.zeros(control['Lv'].shape))
        p = np.zeros(state.shape)
        p[-1] = deepcopy(p_final)
        eps = 1e-16*self.dt

        for k in range(self.Tsize-1, -1, -1):
            qk = state[k]
            uk = control['u'][k]
            Lvk = control['Lv'][k]

            if self.options['mode'] == 'debug':
                self.testHStep(p[k+1], uk, Lvk, qk)

            # Eikonal Time Step
            dfk = self.diffusionKernel.ApplyDiffToImage(qk)
            fk = self.diffusionKernel.ApplyToImage(qk)
            ck = self.reaction(fk)
            dck = self.dreaction(fk)
            fk = np.maximum(-self.iMargin, np.minimum(fk, 1+self.iMargin))
            vk = self.kernel(Lvk)

            normdfk = ((dfk**2).sum(axis=0)+eps)**(1/2)

            nuk = uk * normdfk - (vk*dfk).sum(axis=0) + ck
            muk = self.g_(fk) + self.dt * self.dg_(fk)*nuk
            #muk = np.maximum(muk, self.TOL)
            # hk = self.ginv_(muk)
            # hk = np.maximum(-self.iMargin, np.minimum(hk, 1+self.iMargin))
            digk = self.dginv_(muk)
            rk = self.dg_(fk) * digk
            drk = self.ddg_(fk) * digk

            PullLam['u'][k, :, :] = rk * normdfk * p[k+1]
            PullLam['Lv'][k, :, :, :] \
                = - self.kernel((rk * p[k+1])[None, ...] * dfk)

            vv = (rk * p[k+1])[None, ...] * ((uk/normdfk)[None, ...] * dfk - vk)
            p[k] = self.diffusionKernel.ApplyToImage((rk + self.dt * drk * nuk
                                                    + self.dt * rk * dck) * p[k+1])\
                - self.dt * self.diffusionKernel.ApplyDivToVectorField(vv)
        PullLam['p'] = p
        return PullLam

    def Flow(self, m):
        [T, dim, N, M] = m.shape
        id = idMesh(self.imShape)
        Diff = np.zeros([T+1, dim, N, M])
        Diff[0, :, :, :] = id
        temp = id
        for k in range(T):
            mk = m[k, :, :, :]
            vk = self.kernel(mk)
            v_circ = multilinInterpVectorField(vk, temp)
            temp += self.dt*v_circ
            Diff[k+1, :, :, :] = temp
        return Diff

    def FlowImage(self, m, q0):
        [T, dim, N, M] = m.shape
        id = idMesh(self.imShape)
        img = np.zeros([T+1, N, M])
        img[0] = q0
        for k in range(T):
            vk = self.kernel(m[k])
            img[k+1] = multilinInterp(img[k], id - self.dt * vk)
        return img

    def logJacobian(self, m):
        [T, dim, N, M] = m.shape
        jac = np.zeros([T+1, N, M])
        temp = idMesh(self.imShape)
        for k in range(T):
            mk = m[k, :, :, :]
            vk = self.kernel(mk)
            v_circ = multilinInterpVectorField(vk, temp)
            divv = self.kernelDiv(mk)
            djac = multilinInterp(divv, temp)
            temp += self.dt*v_circ
            jac[k+1] = jac[k] + self.dt*djac
        return jac

    def getGradient(self, coeff=1.0, update=None):
        if update is None:
            control = self.control
        else:
            Lv = self.control['Lv'] - update[1]*update[0]['Lv']
            u = self.control['u'] - update[1]*update[0]['u']
            control = {'Lv': Lv, 'u': u}

        q = self.Evolve(control)

        q_final = q[-1, :, :]
        p_final = -self.endPointGradient(q_final)
        PullLam = self.Backwards(p_final, q, control)

        R = self.ControlGradient(control)
        Eu = R['u'] - PullLam['u']
        Ev = R['Lv'] - PullLam['Lv']
        if self.diffeoOnly:
            Eu *= 0
        if self.topOnly:
            Ev *= 0

        res = {'u': Eu, 'Lv': Ev}
        return res

    def getVariable(self):
        return self.control

    def updateTry(self, dir, eps, objRef=None):
        controlTry = Control()
        controlTry['Lv'] = self.control['Lv'] - eps * dir['Lv']
        controlTry['u'] = self.control['u'] - eps * dir['u']
        objTry = self.objectiveFun(controlTry)
        if np.isnan(objTry):
            logging.info('Warning: nan in updateTry')
            return 1e500

        if (objRef is None) or (objTry < objRef):
            self.controlTry = controlTry
            self.objTry = objTry

        return objTry

    def randomDir(self):
        dirfoo = Control()
        dirfoo['Lv'] = np.random.normal(size=self.control['Lv'].shape)
        dirfoo['u'] = np.random.normal(size=self.control['u'].shape)
        return dirfoo

    def dotProduct_euclidean(self, g1, g2):
        res = np.zeros(len(g2))
        Lv = g1['Lv']
        u = g1['u']
        ll = 0
        for gr in g2:
            ggOldLv = gr['Lv']
            ggOldu = gr['u']
            res[ll] = ((ggOldLv*Lv).sum() + (ggOldu*u).sum())*self.dt
            ll = ll + 1
        return res

    def dotProduct_Riemannian(self, g1, g2):
        return self.dotProduct_euclidean(g1, g2)

    def acceptVarTry(self):
        self.obj = self.objTry
        self.control = deepcopy(self.controlTry)

    def snapshot(self, state=None):
        if state is None:
            state = self.state
        state_u = self.Evolve({'u': self.control['u'], 'Lv': np.zeros(self.control['Lv'].shape)})
        state_Lv = self.FlowImage(self.control['Lv'], self.im0.data)
        fig = plt.figure(3)
        k0 = self.options['padWidth']
        plt.clf()
        plt.subplot(2, 3, 1)
        plt.imshow(self.im0.data[k0:-k0, k0:-k0])
        plt.axis('off')
        plt.title('Template')
        plt.subplot(2, 3, 2)
        plt.imshow(self.im1.data[k0:-k0, k0:-k0])
        plt.axis('off')
        plt.title('Target')
        plt.subplot(2, 3, 3)
        plt.imshow(self.imDef.data[k0:-k0, k0:-k0])
        plt.contour(self.imDef.data[k0:-k0, k0:-k0], [0.5], colors='red')
        plt.axis('off')
        plt.title('Evolution')
        plt.subplot(2, 3, 4)
        plt.imshow(state_u[-1][k0:-k0, k0:-k0])
        plt.contour(state_u[-1][k0:-k0, k0:-k0], [0.5], colors='red')
        plt.title('topology only')
        plt.axis('off')
        plt.subplot(2, 3, 5)
        plt.imshow(state_Lv[-1][k0:-k0, k0:-k0])
        plt.contour(state_Lv[-1][k0:-k0, k0:-k0], [0.5], colors='red')
        plt.title('diffeo only')
        plt.axis('off')
        jac0 = self.logJacobian(self.control['Lv'])
        jac = jac0[-1][k0:-k0, k0:-k0]
        # phi = self.Flow(self.control['Lv'])[-1]
        # x = np.ravel(phi[0][0:-1:sk,0:-1:sk])
        # y = np.ravel(phi[1][0:-1:sk,0:-1:sk])
        plt.subplot(2, 3, 6)
        M = np.abs(jac).max()
        plt.imshow(jac, cmap='seismic', vmin=-M, vmax=M)
        plt.axis('off')
        plt.title('log Jacobian')
        plt.colorbar(shrink=0.5)
        #plt.matshow(jac, cmap='seismic')
        # J = np.nonzero(np.ravel(im > 0.5))
        # plt.scatter(y[J], x[J], c='b', marker='.')
        # J = np.nonzero(np.ravel(im <= 0.5))
        # plt.scatter(y[J], x[J], c='r', marker='.')
        # plt.axis('equal')

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(0.5)
        plt.savefig(self.outputDir + '/summary.png')
        jac = jac0[-1][k0:-k0, k0:-k0]
        M = jac.max()
        m = jac.min()
        jac0 = 255 * (jac0-m) / (M-m)
        for k in range(state.shape[0]):
            saveImage(255*state[k][k0:-k0, k0:-k0], self.outputDir + f'/evolution{k}')
            saveImage(255*state_u[k][k0:-k0, k0:-k0], self.outputDir + f'/evolution_u_only_{k}')
            saveImage(255*state_Lv[k][k0:-k0, k0:-k0], self.outputDir + f'/evolution_Lv_only_{k}')
            saveImage(jac0[k][k0:-k0, k0:-k0], self.outputDir + f'/evolution_logJacobian{k}',
                      normalize=False)

    def initial_plot(self):
        plt.figure(1)
        plt.clf()
        plt.subplot(2, 3, 1)
        plt.imshow(self.KparDiff.K)
        plt.subplot(2, 3, 2)
        plt.imshow(self.endpointKernel.K)
        plt.subplot(2, 3, 3)
        plt.imshow(self.diffusionKernel.K)
        plt.subplot(2, 3, 5)
        plt.imshow(self.im0.data)
        plt.subplot(2, 3, 6)
        plt.imshow(self.im1.data)
        plt.ion()
        plt.show()
        plt.pause(0.001)

    def plot_current(self):
        fig = plt.figure(2)
        k0 = self.options['padWidth']
        plt.clf()
        plt.subplot(1, 3, 1)
        plt.imshow(self.im0.data[k0:-k0, k0:-k0])
        plt.subplot(1, 3, 2)
        plt.imshow(self.im1.data[k0:-k0, k0:-k0])
        plt.subplot(1, 3, 3)
        plt.imshow(self.imDef.data[k0:-k0, k0:-k0])
        plt.contour(self.imDef.data[k0:-k0, k0:-k0], [0.5], colors='red')
        plt.axis('off')
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(0.5)
        # plt.show()
        # plt.pause(0.001)

    def saveAnimation(self):
        fig = plt.figure(10)
        k0 = self.options['padWidth']
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Euclidean LDDMM')
        writer = FFMpegWriter(fps=5, metadata=metadata)
        with writer.saving(fig, self.outputDir + '/animation' + ".mp4", 100):
            for t in range(self.state.shape[0]):
                plt.clf()
                plt.imshow(self.state[t][k0:-k0, k0:-k0])
                plt.contour(self.state[t][k0:-k0, k0:-k0], [0.5], colors='red')
                plt.axis('off')
                writer.grab_frame()

    def startOfIteration(self):
        super().startOfIteration()
        self.diffeoOnly = False
        self.topOnly = False
        if self.iter < self.options['diffeoWarmup']:
            self.diffeoOnly = True
        else:
            if self.options['alternate'] > 0:
                if (self.iter // self.options['alternate']) % 2 == 0:
                    self.diffeoOnly = True
                else:
                    self.topOnly = True
                if self.iter % self.options['alternate'] == 0:
                    self.reset = True
            elif self.options['diffeoWarmup'] > 0 and  self.iter == self.options['diffeoWarmup']:
                self.diffeoOnly = False
                self.reset = True

    def endOfIteration(self, forceSave=False):
        self.iter += 1
        self.state = self.Evolve(self.control, self.im0.data)
        self.imDef.data = self.state[-1]
        if self.iter % self.saveRate == 0:
            self.snapshot()
            self.saveAnimation()
        else:
            self.plot_current()

    def endOfProcedure(self):
        self.endOfIteration()

    def optimize(self):
        grd = self.getGradient(coeff=self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = max(0.001, np.sqrt(grd2) / 10000)
        self.epsMax = 5.
        logging.info('Gradient lower bound: %f' % (self.gradEps))
        bfgs(self, verb=self.options['verb'], maxIter=self.options['maxIter'],
             TestGradient=self.options['testGradient'], epsInit=1.,
             lineSearch=self.options['lineSearch'], memory=10)
