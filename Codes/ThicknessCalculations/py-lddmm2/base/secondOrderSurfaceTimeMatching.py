import logging
import numpy as np
import numpy.linalg as la
from copy import deepcopy
from . import surfaces
from .pointSets import PointSet
from .surfaceTimeSeries import SurfaceTimeMatching
from .secondOrderPointSetMatching import SecondOrderPointSetMatching
from . import pointEvolution as evol
from .affineBasis import getExponential, gradExponential
from .surfaceDistances import L2Norm0


class Control(dict):
    def __init__(self):
        super().__init__()
        self['a0'] = None
        self['rhot'] = None
        self['Afft'] = None
        self['a00'] = None
        self['rhot0'] = None
        self['Afft0'] = None

class State(dict):
    def __init__(self):
        super().__init__()
        self['xt0'] = None
        self['Jt0'] = None
        self['xt'] = None
        self['Jt'] = None


class SecondOrderSurfaceTimeMatching(SurfaceTimeMatching, SecondOrderPointSetMatching):
    def __init__(self, Template=None, Target=None, options=None):
        # self.rescaleTemplate = rescaleTemplate
        super().__init__(Template=Template, Target=Target, options=options)


    def getDefaultOptions(self):
        options = super().getDefaultOptions()
        options['typeRegression'] = False
        options['controlWeight'] = 1.0
        options['initialMomentum'] = None
        options['KparDiff0'] = None
        return options


    def setDotProduct(self, unreduced=False):
        self.euclideanGradient = True
        self.dotProduct = self.dotProduct_euclidean

    def set_parameters(self):
        super().set_parameters()
        if self.options['affine']=='euclidean' or self.options['affine']=='translation':
            self.saveCorrected = True
        else:
            self.saveCorrected = False

        self.typeRegressionSave = self.options['typeRegression']

        if self.options['affine'] != 'none':
            self.options['typeRegression'] = 'affine'

        if self.options['KparDiff0'] is None:
            self.options['KparDiff0'] = self.options['KparDiff']

        self.isActive = dict()
        if self.options['typeRegression'] == 'spline':
            self.isActive['a00'] = False
            self.isActive['rhot0'] = True
            self.isActive['a0'] = False
            self.isActive['rhot'] = True
        elif self.options['typeRegression'] == 'geodesic':
            self.isActive['a00'] = True
            self.isActive['rhot0'] = False
            self.isActive['a0'] = True
            self.isActive['rhot'] = False
        elif self.options['typeRegression'] == "affine":
            self.isActive['a00'] = False
            self.isActive['rhot0'] = False
            self.isActive['a0'] = False
            self.isActive['rhot'] = False
        else:
            self.isActive['a00'] = True
            self.isActive['rhot0'] = True
            self.isActive['a0'] = True
            self.isActive['rhot'] = True

        if self.affineDim > 0:
            self.isActive['Afft0'] = True
            self.isActive['Afft'] = True
        else:
            self.isActive['Afft0'] = False
            self.isActive['Afft'] = False

    def initialize_variables(self):
        self.x0 = self.fv0.vertices
        self.fvDef = []
        for k in range(self.nTarg):
            self.fvDef.append(surfaces.Surface(surf=self.fv0))
        self.def_lmk = []
        if self.match_landmarks:
            for k in range(self.nTarg):
                self.def_lmk.append(PointSet(data=self.tmpl_lmk))
        self.nvert = self.fvInit.vertices.shape[0]
        if self.match_landmarks:
            self.x0 = np.concatenate((self.fvInit.vertices, self.tmpl_lmk.vertices), axis=0)
            self.nlmk = self.tmpl_lmk.vertices.shape[0]
        else:
            self.x0 = np.copy(self.fvInit.vertices)
            self.nlmk = 0

        self.npt = self.x0.shape[0]


        if self.options['times'] is None:
            self.times = np.arange(1, self.nTarg+1)
        else:
            self.times = np.array(self.options['times'])

        self.Tsize0 = int(round(1./self.options['timeStep']))
        self.Tsize = int(round(self.times[-1]/self.options['timeStep']))
        self.jumpIndex = np.round(self.times/self.options['timeStep']).astype(int)
        #print self.jumpIndex
        self.isjump = np.zeros(self.Tsize+1, dtype=bool)
        for k in self.jumpIndex:
            self.isjump[k] = True

        self.control['rhot0'] = np.zeros([self.Tsize0, self.x0.shape[0], self.x0.shape[1]])
        self.control['rhot'] = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        if self.options['initialMomentum']==None:
            self.state['xt0'] = np.tile(self.x0, [self.Tsize0+1, 1, 1])
            self.state['xt'] = np.tile(self.x0, [self.Tsize+1, 1, 1])
            self.control['a00'] = np.zeros([self.x0.shape[0], self.x0.shape[1]])
            self.control['a0'] = np.zeros([self.x0.shape[0], self.x0.shape[1]])
            self.state['at'] = np.tile(self.control['a0'], [self.Tsize+1, 1, 1])
            self.state['at0'] = np.tile(self.control['a00'], [self.Tsize0+1, 1, 1])
        else:
            self.control['a00'] = self.options['initialMomentum'][0]
            self.control['a0'] = self.options['initialMomentum'][1]
            self.state = self.solveStateEquation()

        self.control['Afft0'] = np.zeros([self.Tsize0, self.affineDim])
        self.control['Afft'] = np.zeros([self.Tsize, self.affineDim])
        #self.v = np.zeros([self.Tsize+1, self.npt, self.dim])
        self.controlTry = deepcopy(self.control)
        self.stateTry = deepcopy(self.state)



    def solveStateEquation(self, control= None, init_state = None, kernel = None, options=None):
        if control is None:
            control = self.control
        if init_state is None:
            init_state = self.x0

        A0 = self.affB.getTransforms(control['Afft0'])
        A = self.affB.getTransforms(control['Afft'])

        st0 = evol.secondOrderEvolution(init_state, control['a00'], self.options['KparDiff0'], self.options['timeStep'],
                                        affine=A0, withSpline=control['rhot0'], options=options)
        st = evol.secondOrderEvolution(st0['xt'][-1, ...], control['a0'], self.options['KparDiff'],
                                        self.options['timeStep'], affine=A, withSpline=control['rhot'],
                                       options=options)
        st['xt0'] = st0['xt']
        st['at0'] = st0['at']
        st['Jt0'] = st0['Jt']
        return  st


    def fix_orientation(self, fv1=None):
        if fv1 is None:
            fv1 = self.fv1

        self.fv0.getEdges()
        self.closed = self.fv0.bdry.max() == 0
        v0 = self.fv0.surfVolume()
        if self.closed:
            if self.options['errorType'] == 'L2Norm' and v0 < 0:
                self.fv0.flipFaces()
                v0 = -v0
            if v0 < 0:
                self.fv0ori = -1
            else:
                self.fv0ori = 1
        else:
            self.fv0ori = 1
        if fv1:
            self.fv1ori = []
            for f1 in fv1:
                f1.getEdges()
                closed = self.closed and f1.bdry.max() == 0
                if closed:
                    v1 = f1.surfVolume()
                    if v0*v1 < 0:
                        f1.flipFaces()
                        v1 = -v1
                        if v1 < 0:
                            self.fv1ori.append(-1)
                        else:
                            self.fv1ori.append(1)
            else:
                self.fv0ori = 1
                self.fv1ori = 1
        else:
            self.fv1ori = 1
        #self.fv0Fine = surfaces.Surface(surf=self.fv0)
        logging.info('orientation: {0:d}'.format(self.fv0ori))



    def  objectiveFunDef(self, control, var = None, withTrajectory = False, withJacobian=False, display=False):
        if var is None or 'Init' not in var:
            x0 = self.x0
        else:
            x0 = var['Init'][0]
            
        a00 = control['a00']
        a0 = control['a0']
        Afft0 = control['Afft0']
        rhot0 = control['rhot0']
        Afft = control['Afft']
        rhot = control['rhot']
            
        timeStep = self.options['timeStep']
        st = self.solveStateEquation(control=control, options={'withJacobian':withJacobian})

        xt = st['xt']
        obj00 = 0.5 * (a00 * self.options['KparDiff0'].applyK(x0,a00)).sum() 
        obj0 = 0.5 * (a0 * self.options['KparDiff'].applyK(xt[0,...],a0)).sum()
        obj10 = 0
        obj1 = 0
        obj20 = 0
        obj2 = 0
        for t in range(self.Tsize0):
            rho = np.squeeze(rhot0[t, :, :])            
            obj10 += timeStep* self.options['controlWeight'] * (rho**2).sum()/2
            if self.affineDim > 0:
                obj20 +=  timeStep * (self.affineWeight.reshape(Afft0[t].shape) * Afft0[t]**2).sum()/2
        for t in range(self.Tsize):
            rho = np.squeeze(rhot[t, :, :])            
            obj1 += timeStep* self.options['controlWeight'] * (rho**2).sum()/2
            if self.affineDim > 0:
                obj2 +=  timeStep * (self.affineWeight.reshape(Afft[t].shape) * Afft[t]**2).sum()/2
            #print xt.sum(), at.sum(), obj
        obj = obj1+obj2+obj0+obj10+obj20+obj00
        if self.options['mode'] == 'debug':
            logging.info(f'deformation terms: init {obj00:.4f}, rho {obj10:.4f}, aff {obj20:.4f}; init {obj0:.4f}, rho {obj1:.4f}, aff {obj2:.4f}')

        return obj, st


    # def objectiveFun(self):
    #     if self.obj == None:
    #         self.obj, self.state = self.objectiveFunDef(self.control)
    #         self.obj0 = 0
    #         for k in range(self.nTarg):
    #             if self.options['errorType'] == 'L2Norm':
    #                 self.obj0 += L2Norm0(self.fv1[k]) / (self.options['sigmaError'] ** 2)
    #             else:
    #                 self.obj0 += self.fun_obj0(self.fv1[k]) / (self.options['sigmaError']**2)
    #             self.fvDef[k].updateVertices(self.state['xt'][self.jumpIndex[k], :, :])
    #         self.obj += self.obj0 + self.dataTerm(self.fvDef)
    #     return self.obj

    def getVariable(self):
        return self.control
    
    def updateTry(self, dr, eps, objRef=None):
        controlTry = Control()
        for k in dr.keys():
            if self.isActive[k]:
                if dr[k] is not None:
                    # print(k, self.control[k].shape, dr[k].shape)
                    controlTry[k] = self.control[k] - eps * dr[k]
            else:
                controlTry[k] = self.control[k]

        objTry, stTry = self.objectiveFunDef(controlTry,  withTrajectory=True)
        objTry += self.obj0

        ff = [] 
        for k in range(self.nTarg):
            ff.append(surfaces.Surface(surf=self.fvDef[k]))
            ff[k].updateVertices(np.squeeze(stTry['xt'][self.jumpIndex[k], :, :]))

        if self.match_landmarks:
            pp = []
            for k in range(self.nTarg):
                pp.append(PointSet(data=self.def_lmk[k]))
                pp[k].updateVertices(np.squeeze(stTry['xt'][self.jumpIndex[k], self.nvert:, :]))
        else:
            pp = None

        objTry += self.dataTerm(ff, {'lmk_def':pp})
        if np.isnan(objTry):
            print('Warning: nan in updateTry')
            return 1e500

        if (objRef == None) or (objTry < objRef):
            self.controlTry = controlTry
            self.objTry = objTry
            self.stateTry = stTry
            #print 'objTry=',objTry, dir.diff.sum()

        return objTry


    def secondOrderCovector(self, x00, control, px1, pa1,  affine = (None, None)):
        rhot0 = control['rhot0']
        rhot = control['rhot']

        N = x00.shape[0]
        A = self.affB.getTransforms(control['Afft'])
        A0 = self.affB.getTransforms(control['Afft0'])
        dim = x00.shape[1]

        st = self.solveStateEquation(control=control, init_state=x00)
        xt0 = st['xt0']
        at0 = st['at0']
        xt = st['xt']
        at = st['at']

        pxt, pat, _, _ = evol.secondOrderCovector(xt[0, :, :], at[0, :, :], px1, pa1, self.options['KparDiff'],
                                                  self.options['timeStep'], affine=A, withSpline=rhot,
                                                  isjump=self.isjump, forwardState=(xt, at))

        px01 = pxt[0,:,:] - self.options['KparDiff'].applyDiffKT(self.x0, control['a0'], control['a0'])
        pa01 = np.zeros([N, dim])
        pxt0, pat0, _, _ = evol.secondOrderCovector(x00, at0[0, :, :], px01, pa01, self.options['KparDiff0'],
                                                    self.options['timeStep'], affine=A0, withSpline=rhot0,
                                                    forwardState=(xt0, at0))

        return [[pxt0, pat0, xt0, at0], [pxt, pat, xt, at]]

    # Computes gradient after covariant evolution for deformation cost a^TK(x,x) a
    def secondOrderGradient(self, x00, control, px1, pa1, controlWeight=1.0):
        a00 = control['a00'] 
        rhot0 = control['rhot0']
        a0 = control['a0']
        rhot = control['rhot']
        foo = self.secondOrderCovector(x00, control, px1, pa1)
        (pxt0, pat0, xt0, at0) = foo[0]
        A = self.affB.getTransforms(control['Afft'])
        A0 = self.affB.getTransforms(control['Afft0'])

        if A0 is not None:
            dA0 = np.zeros(A0[0].shape)
            db0 = np.zeros(A0[1].shape)
        else:
            dA0 = None
            db0 = None

        if A is not None:
            dA = np.zeros(A[0].shape)
            db = np.zeros(A[1].shape)
        else:
            dA = None
            db = None

        Tsize0 = self.Tsize0
        timeStep = self.options['timeStep']
        drhot0 = np.zeros(rhot0.shape)
        KparDiff = self.options['KparDiff0']
        if A0 is not None:
            for t in range(Tsize0):
                x = xt0[t, :, :]
                a = at0[t, :, :]
                rho = rhot0[t, :, :]
                px = pxt0[t+1, :, :]
                pa = pat0[t+1, :, :]
                zx = x + timeStep * KparDiff.applyK(x, a)
                za = a + timeStep * (-KparDiff.applyDiffKT(x, a, a) + rho)
                U = getExponential(timeStep * A0[0][t])
                #U = np.eye(dim) + timeStep * affine[0][k]
                Ui = la.inv(U)
                pa = np.dot(pa, Ui.T)
                za = np.dot(za, Ui)
                dA0[t,...] =  (gradExponential(timeStep*A0[0][t], px, zx)
                                - gradExponential(timeStep*A0[0][t], za, pa))
                drhot0[t,...] = rho*controlWeight - pa
            db0 = pxt0[1:Tsize0+1,...].sum(axis=1)
        else:
            for t in range(Tsize0):
                rho = rhot0[t, :, :]
                pa = pat0[t+1, :, :]
                drhot0[t,...] = rho*controlWeight - pa

        da00 = KparDiff.applyK(x00, a00) - pat0[0,...]
        
        
        (pxt, pat, xt, at) = foo[1]
        Tsize = self.Tsize
        drhot = np.zeros(rhot.shape)
        KparDiff = self.options['KparDiff']
        if A is not None:
            for t in range(Tsize):
                x = xt[t, :, :]
                a = at[t, :, :]
                rho = rhot[t, :, :]
                px = pxt[t+1, :, :]
                pa = pat[t+1, :, :]
                zx = x + timeStep * KparDiff.applyK(x, a)
                za = a + timeStep * (-KparDiff.applyDiffKT(x, a, a) + rho)
                U = getExponential(timeStep * A[0][t])
                Ui = la.inv(U)
                pa = np.dot(pa, Ui.T)
                za = np.dot(za, Ui)
                dA[t,...] =  (gradExponential(timeStep*A[0][t], px, zx)
                                - gradExponential(timeStep*A[0][t], za, pa))
                drhot[t,...] = rho*controlWeight - pa
            db = pxt[1:Tsize+1,...].sum(axis=1)
        else:
            for t in range(Tsize):
                rho = rhot[t, :, :]
                pa = pat[t+1, :, :]
                drhot[t,...] = rho*controlWeight - pa

        da0 = KparDiff.applyK(xt[0,...], a0) - pat[0,...]
        res0 = dict()
        res0['da0'] = da00
        res0['drhot'] = drhot0
        res0['dA'] = dA0
        res0['db'] = db0
        res0['xt'] = xt0
        res0['at'] = at0
        res0['pxt'] = pxt0
        res0['pat'] = pat0

        res = dict()
        res['da0'] = da0
        res['drhot'] = drhot
        res['dA'] = dA
        res['db'] = db
        res['xt'] = xt
        res['at'] = at
        res['pxt'] = pxt
        res['pat'] = pat

        return res0, res

    def getGradient(self, coeff=1.0, update=None):
        A = None
        A0 = None
        #logging.info('Computing gradient')
        if update is None:
            control = self.control
            if self.match_landmarks:
                endPoint = (self.fvDef, self.def_lmk)
            else:
                endPoint = self.fvDef
        else:
            eps = update[1]
            dr = update[0]
            control = Control()
            for k in dr.keys():
                if self.isActive[k]:
                    if dr[k] is not None:
                        control[k] = self.control[k] - eps * dr[k]
                else:
                    control[k] = self.control[k]
            st = self.solveStateEquation(control=control)
            endPoint = []
            xt = st['xt']
            if self.match_landmarks:
                endPoint0 = []
                endPoint1 = []
                for k in range(self.nTarg):
                    endPoint0.append(surfaces.Surface(surf=self.fv0))
                    endPoint0[k].updateVertices(xt[self.jumpIndex[k], :self.nvert, :])
                    endPoint1.append(PointSet(data=xt[self.jumpIndex[k], self.nvert:, :]))
                endPoint = (endPoint0, endPoint1)
            else:
                endPoint = []
                for k in range(self.nTarg):
                    fvDef = surfaces.Surface(surf=self.fv0)
                    fvDef.updateVertices(xt[self.jumpIndex[k], :, :])
                    endPoint.append(fvDef)


        px1 = self.endPointGradient(endPoint=endPoint)
        pa1 = []
        for k in range(self.nTarg):
            pa1.append(np.zeros(self.control['a0'].shape))

        foo0, foo1 = self.secondOrderGradient(self.x0, control, px1, pa1,
                                              controlWeight=self.options['controlWeight'])
        grd = Control()

        if self.options['typeRegression'] == 'spline':
            grd['rhot'] = foo1['drhot']/(coeff)
            grd['rhot0'] = foo0['drhot']/(coeff)
            #grd.rhot = foo[1]/(coeff*self.rhot.shape[0])
        elif self.options['typeRegression'] == 'geodesic':
            grd['a0'] = foo1['da0'] / coeff
            grd['a00'] = foo0['da0'] / coeff
        elif self.options['typeRegression'] == 'affine':
            pass
        else:
            grd['rhot'] = foo1['drhot']/(coeff)
            grd['rhot0'] = foo0['drhot']/(coeff)
            grd['a0'] = foo1['da0'] / coeff
            grd['a00'] = foo0['da0'] / coeff

        if self.affineDim > 0 and self.iter < self.affBurnIn:
            dim2 = self.dim ** 2
            grd['Afft'] = np.zeros(self.control['Afft'].shape)
            grd['Afft0'] = np.zeros(self.control['Afft0'].shape)
            dA0 = foo0['dA']
            db0 = foo0['db']
            dA = foo1['dA']
            db = foo1['db']
            grd['Afft0'] = np.multiply(self.affineWeight.reshape([1, self.affineDim]), self.control['Afft0'])
            for t in range(self.Tsize0):
               dAff = np.dot(self.affineBasis.T, np.vstack([dA0[t].reshape([dim2,1]), db0[t].reshape([self.dim, 1])]))
               grd['Afft0'][t] -=  dAff.reshape(grd['Afft0'][t].shape)
            grd['Afft0'] *= self.options['timeStep']/(self.coeffAff*coeff)
            grd['Afft'] = np.multiply(self.affineWeight.reshape([1, self.affineDim]), self.control['Afft'])
            for t in range(self.Tsize):
               dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
               grd['Afft'][t] -=  dAff.reshape(grd['Afft'][t].shape)
            grd['Afft'] *= self.options['timeStep']/(self.coeffAff*coeff)
        
        #print (grd.a00**2).sum(),(grd.rhot0**2).sum(),(grd.aff0**2).sum(),(grd.a0**2).sum(),(grd.rhot**2).sum(),(grd.aff**2).sum() 
        return grd



    def randomDir(self):
        dirfoo = Control()
        if self.options['typeRegression'] == 'spline':
            dirfoo['a0'] = np.zeros([self.npt, self.dim])
            dirfoo['rhot'] = np.random.randn(self.Tsize, self.npt, self.dim)
            dirfoo['a00'] = np.zeros([self.npt, self.dim])
            dirfoo['rhot0'] = np.random.randn(self.Tsize0, self.npt, self.dim)
        elif self.options['typeRegression'] == 'geodesic':
            dirfoo['a0'] = np.random.randn(self.npt, self.dim)
            dirfoo['rhot'] = np.zeros([self.Tsize, self.npt, self.dim])
            dirfoo['a00'] = np.random.randn(self.npt, self.dim)
            dirfoo['rhot0'] = np.zeros([self.Tsize0, self.npt, self.dim])
        elif self.options['typeRegression'] == 'affine':
            dirfoo['a0'] = np.zeros([self.npt, self.dim])
            dirfoo['rhot'] = np.zeros([self.Tsize, self.npt, self.dim])
            dirfoo['a00'] = np.zeros([self.npt, self.dim])
            dirfoo['rhot0'] = np.zeros([self.Tsize0, self.npt, self.dim])
        else:
            dirfoo['a0'] = np.random.randn(self.npt, self.dim)
            dirfoo['rhot'] = np.random.randn(self.Tsize, self.npt, self.dim)
            dirfoo['a00'] = np.random.randn(self.npt, self.dim)
            dirfoo['rhot0'] = np.random.randn(self.Tsize0, self.npt, self.dim)


        if self.iter < self.affBurnIn:
            dirfoo['Afft'] = np.random.randn(self.Tsize, self.affineDim)
            dirfoo['Afft0'] = np.random.randn(self.Tsize0, self.affineDim)
        else:
            dirfoo['Afft'] = np.zeros((self.Tsize, self.affineDim))
            dirfoo['Afft0'] = np.zeros((self.Tsize0, self.affineDim))
        return dirfoo

    def dotProduct_euclidean(self, g1, g2):
        res = np.zeros(len(g2))
        gg = g1['rhot']
        gga = g1['a0']
        uu = g1['Afft']
        ll = 0
        for gr in g2:
            res[ll] = 0
            if gg is not None:
                res[ll] += (gr['rhot']*gg).sum()*self.options['timeStep']
            if gga is not None:
                res[ll] += (gr['a0'] * gga).sum()
            if uu is not None:
                res[ll] += (uu * gr['Afft']).sum() * self.coeffAff
            ll = ll+1

        gg = g1['rhot0']
        gga = g1['a00']
        uu = g1['Afft0']
        ll = 0
        for gr in g2:
            if gg is not None:
                res[ll] += (gr['rhot0']*gg).sum()*self.options['timeStep']
            if gga is not None:
                res[ll] += (gr['a00'] * gga).sum()
            if uu is not None:
                res[ll] += (uu * gr['Afft0']).sum() * self.coeffAff
            ll = ll+1

        return res


    def endOfIteration(self, forceSave = False):
        self.iter += 1
        if self.iter >= self.affBurnIn:
            self.options['typeRegression'] = self.typeRegressionSave
            self.affine = 'none'
            #self.coeffAff = self.coeffAff2
        nvert = self.nvert
        if (self.iter % self.saveRate == 0):
            logging.info('Saving surfaces...')
            obj1, self.state = self.objectiveFunDef(self.control, withTrajectory=True, withJacobian=True,
                                                    display=self.options['verb'])
            for k in range(self.nTarg):
                self.fvDef[k].updateVertices(self.state['xt'][self.jumpIndex[k], :nvert, :])

            if self.saveCorrected:
                f = surfaces.Surface(surf=self.fv0)
                X0 = self.affB.integrateFlow(self.control['Afft0'])
                X = self.affB.integrateFlow(self.control['Afft'])
                displ = np.zeros(nvert)
                at0Corr = np.zeros(self.state['at0'].shape)
                if self.match_landmarks:
                    p = PointSet(data=self.tmpl_lmk)
                else:
                    p = None
                for t in range(self.Tsize0+1):
                    R0 = X0[0][t,...]
                    U0 = la.inv(R0)
                    b0 = X0[1][t,...]
                    yyt = np.dot(self.state['xt0'][t,...] - b0, U0.T)
                    zt = np.dot(self.state['xt0'][t,...] - b0, U0.T)
                    if t < self.Tsize0:
                        a = np.dot(self.state['at0'][t,...], R0)
                        at0Corr[t,...] = a
                        vt = self.options['KparDiff0'].applyK(yyt, a, firstVar=zt)
                        vt = np.dot(vt, U0.T)
                    f.updateVertices(zt[:nvert, :])
                    if self.match_landmarks:
                        p.updateVertices(yyt[self.nvert:, :])
                        p.saveVTK(self.outputDir + '/' + self.options['saveFile']+'Corrected'+str(t) + '_lmk.vtk')
                    vf = surfaces.vtkFields()
                    vf.scalars.append('Jacobian')
                    vf.scalars.append(displ)
                    vf.scalars.append('displacement')
                    vf.scalars.append(displ)
                    vf.vectors.append('velocity')
                    vf.vectors.append(vt[:nvert, :])
                    f.saveVTK2(self.options['outputDir'] +'/'+self.options['saveFile']+'Corrected'+str(t)+'.vtk', vf)

                dt = 1.0 /self.Tsize0
                atCorr = np.zeros(self.state['at'].shape)
                for t in range(self.Tsize+1):
                    U = la.inv(X[0][t,...])
                    yyt = np.dot(self.state['xt'][t,...] - b0, U0.T)
                    yyt = np.dot(yyt - X[1][t, ...], U.T)
                    # zt = np.dot(self.state['xt'][t,...] - b0, U0.T)
                    # zt = np.dot(zt - X[1][t, ...], U.T)
                    if t < self.Tsize:
                        a = np.dot(self.state['at'][t,...], R0)
                        a = np.dot(a, X[0][t,...])
                        atCorr[t,...] = a
                        # vt = self.options['KparDiff'].applyK(yyt, a, firstVar=zt)
                        vt = self.options['KparDiff'].applyK(yyt, a)
                        vt = np.dot(vt, U.T)
                    f.updateVertices(yyt[:nvert, :])
                    if self.match_landmarks:
                        p.updateVertices(yyt[self.nvert:, :])
                        p.saveVTK(self.outputDir + '/' + self.options['saveFile']+'Corrected'+str(t) + '_lmk.vtk')
                    vf = surfaces.vtkFields()
                    vf.scalars.append('Jacobian')
                    vf.scalars.append(self.state['Jt'][t, :nvert])
                    vf.scalars.append('displacement')
                    vf.scalars.append(displ)
                    vf.vectors.append('velocity')
                    vf.vectors.append(vt[:nvert, :])
                    nu = self.fv0ori*f.computeVertexNormals()
                    displ += dt * (vt[:nvert, :]*nu).sum(axis=1)
                    f.saveVTK2(self.options['outputDir'] +'/'+self.options['saveFile']+'Corrected'+str(t+self.Tsize0)+'.vtk', vf)


                for k,fv in enumerate(self.fv1):
                    f = surfaces.Surface(surf=fv)
                    U = la.inv(X[0][self.jumpIndex[k]])
                    yyt = np.dot(f.vertices - b0, U0.T)
                    yyt = np.dot(yyt - X[1][self.jumpIndex[k], ...], U.T)
                    f.updateVertices(yyt[:nvert, :])
                    f.saveVTK(self.options['outputDir'] +'/Target'+str(k)+'Corrected.vtk')
            
            fvDef = surfaces.Surface(surf=self.fv0)
            AV0 = fvDef.computeVertexArea()
            nu = self.fv0ori*self.fv0.computeVertexNormals()
            #v = self.v[0,...]
            displ = np.zeros(nvert)
            # dt = 1.0 /self.Tsize
            v = self.options['KparDiff0'].applyK(self.x0, self.state['at0'][0,...])
            for kk in range(self.Tsize0+1):
                fvDef.updateVertices(np.squeeze(self.state['xt0'][kk, :nvert, :]))
                AV = fvDef.computeVertexArea()
                AV = AV[0]/AV0[0]
                vf = surfaces.vtkFields()
                vf.scalars.append('Jacobian')
                vf.scalars.append(self.state['Jt0'][kk, :nvert, 0])
                vf.scalars.append('Jacobian_T')
                vf.scalars.append(AV)
                vf.scalars.append('Jacobian_N')
                vf.scalars.append(self.state['Jt0'][kk, :nvert, 0]/AV)
                vf.scalars.append('displacement')
                vf.scalars.append(displ)
                if kk < self.Tsize:
                    v = self.options['KparDiff0'].applyK(self.state['xt0'][kk,...], self.state['at0'][kk,...])

                vf.vectors.append('velocity')
                vf.vectors.append(np.copy(v[:nvert, :]))
                fvDef.saveVTK2(self.options['outputDir'] +'/'+ self.options['saveFile']+str(kk)+'.vtk', vf)
                if self.match_landmarks:
                    pp = PointSet(data=self.state['xt'][kk, nvert:, :])
                    pp.saveVTK(self.outputDir + '/' + self.options['saveFile']+str(kk) + '_lmk.vtk')

            #dt = 1.0 /self.Tsize
            v = self.options['KparDiff'].applyK(self.state['xt'][0,...], self.state['at'][0,...])
            for kk in range(self.Tsize+1):
                fvDef.updateVertices(np.squeeze(self.state['xt'][kk, :nvert, :]))
                AV = fvDef.computeVertexArea()
                AV = AV[0]/AV0[0]
                vf = surfaces.vtkFields()
                vf.scalars.append('Jacobian')
                vf.scalars.append(self.state['Jt'][kk, :nvert, 0])
                vf.scalars.append('Jacobian_T')
                vf.scalars.append(AV)
                vf.scalars.append('Jacobian_N')
                vf.scalars.append(self.state['Jt'][kk, :nvert, 0]/AV)
                vf.scalars.append('displacement')
                vf.scalars.append(displ)
                if self.Tsize > 0:
                    displ += (v[:nvert, :]*nu).sum(axis=1) / self.Tsize
                if kk < self.Tsize:
                    nu = self.fv0ori*fvDef.computeVertexNormals()
                    v = self.options['KparDiff'].applyK(self.state['xt'][kk,...], self.state['at'][kk,...])
                    #v = self.v[kk,...]
                    kkm = kk
                else:
                    kkm = kk-1
                vf.vectors.append('velocity')
                vf.vectors.append(np.copy(v[:nvert, :]))
                fvDef.saveVTK2(self.options['outputDir'] +'/'+ self.options['saveFile']+str(kk+self.Tsize0)+'.vtk', vf)
                if self.match_landmarks:
                    pp = PointSet(data=self.state['xt'][kk, nvert:, :])
                    pp.saveVTK(self.outputDir + '/' + self.options['saveFile']+str(kk+self.Tsize0) + '_lmk.vtk')
        else:
            obj1, self.state= self.objectiveFunDef(self.control, withTrajectory=True, display=True)
            for k in range(self.nTarg):
                self.fvDef[k].updateVertices(np.squeeze(self.state['xt'][self.jumpIndex[k], :nvert, :]))


