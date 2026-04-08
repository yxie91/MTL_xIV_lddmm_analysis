import logging
import numpy as np
import numpy.linalg as la
from . import surfaces, surfaceDistances as sd
from . import pointSets
from .affineBasis import getExponential
from .surfaceMatching import SurfaceMatching, Control, State


#  Main class for surface matching
#        Template: surface class (from surface.py); if not specified, opens fileTemp
#        Target: surface class (from surface.py); if not specified, opens fileTarg
#        param: surfaceMatchingParam
#        verb: verbose printing
#        regWeight: multiplicative constant on object regularization
#        affineWeight: multiplicative constant on affine regularization
#        rotWeight: multiplicative constant on affine regularization (supercedes affineWeight)
#        scaleWeight: multiplicative constant on scale regularization (supercedes affineWeight)
#        transWeight: multiplicative constant on translation regularization (supercedes affineWeight)
#        testGradient: evaluates gradient accuracy over random direction (debug)
#        outputDir: where results are saved
#        saveFile: generic name for saved surfaces
#        affine: 'affine', 'similitude', 'euclidean', 'translation' or 'none'
#        maxIter: max iterations in conjugate gradient
class SurfaceTimeMatching(SurfaceMatching):
    def __init__(self, Template=None, Target=None, options=None):
        # self.rescaleTemplate = rescaleTemplate
        super().__init__(Template=Template, Target=Target, options=options)
        self.ds = 1.

    def getDefaultOptions(self):
        options = super().getDefaultOptions()
        options['rescaleTemplate'] = False
        options['volumeWeight'] = None
        options['times'] = None
        options['timeStep0'] = None
        return options

    def initialize_variables(self):
        # self.Tsize = int(round(1.0 / self.options['timeStep']))
        if self.options['timeStep0'] is None:
            self.options['timeStep0'] = self.options['timeStep']
        # self.Tsize0 = int(round(1.0 / self.options['timeStep0']))
        self.nvert = self.fvInit.vertices.shape[0]
        if self.match_landmarks:
            self.control['x0'] = np.concatenate((self.fvInit.vertices,
                                                 self.tmpl_lmk.vertices),
                                                axis=0)
            self.nlmk = self.tmpl_lmk.vertices.shape[0]
        else:
            self.control['x0'] = np.copy(self.fvInit.vertices)
            self.nlmk = 0
        self.x0try = np.copy(self.control['x0'])
        self.npt = self.control['x0'].shape[0]
        if self.options['times'] is None:
            self.times = (1+np.arange(self.nTarg))
        else:
            self.times = np.array(self.options['times'])

        self.maxT = self.times[-1]
        self.Tsize = int(round(self.maxT/self.options['timeStep']))
        self.jumpIndex = np.round(self.times/self.options['timeStep']).astype(int)
        # print(self.Tsize)
        self.isjump = np.zeros(self.Tsize+1, dtype=bool)
        for k in self.jumpIndex:
            self.isjump[k] = True


        self.control['at'] = np.zeros([self.Tsize, self.control['x0'].shape[0],
                                       self.control['x0'].shape[1]])
        if self.randomInit:
            self.control['at'] = np.random.normal(0, 1, self.control['at'].shape)
        self.controlTry['at'] = np.zeros([self.Tsize, self.control['x0'].shape[0], self.control['x0'].shape[1]])
        self.control['Afft'] = np.zeros([self.Tsize, self.affineDim])
        self.controlTry['Afft'] = np.zeros([self.Tsize, self.affineDim])
        self.state['xt'] = np.tile(self.control['x0'], [self.Tsize + 1, 1, 1])
        self.stateTry = State()
        self.stateTry['xt'] = np.copy(self.state['xt'])
        self.v = np.zeros([self.Tsize + 1, self.npt, self.dim])

        self.fvDef = []
        for k in range(self.nTarg):
            self.fvDef.append(surfaces.Surface(surf=self.fv0))
        self.def_lmk = []
        if self.match_landmarks:
            for k in range(self.nTarg):
                self.def_lmk.append(pointSets.PointSet(data=self.tmpl_lmk))
        self.a0 = np.zeros([self.control['x0'].shape[0], self.control['x0'].shape[1]])

        self.regweight_ = np.ones(self.Tsize)
        self.regweight_[range(self.jumpIndex[0])] = self.options['regWeight']
        self.saveFileList = []
        for kk in range(self.Tsize+1):
            if kk < self.jumpIndex[0]:
                self.saveFileList.append(self.options['saveFile'] + f'fromTemplate{kk:03d}')
            else:
                self.saveFileList.append(self.options['saveFile'] + f'{kk-self.jumpIndex[0]:03d}')

    def set_template_and_target(self, Template, Target, misc=None):
        if Template is None:
            logging.error('Please provide a template surface')
            return
        else:
            self.fv0 = surfaces.Surface(surf=Template, checkOrientation=self.options['checkOrientation'])

        if Target is None:
            logging.error('Please provide a list of target surfaces')
            return
        else:
            self.nTarg = len(Target)
            self.fv1 = []
            if self.options['errorType'] == 'L2Norm':
                for f in Target:
                    fv1 = surfaces.Surface()
                    fv1.readFromImage(f)
                    self.fv1.append(fv1)
            elif self.options['errorType'] == 'PointSet':
                for f in Target:
                    self.fv1.append(pointSets.PointSet(data=f))
            else:
                for f in Target:
                    self.fv1.append(surfaces.Surface(surf=f, checkOrientation=self.options['checkOrientation']))

        if self.options['rescaleTemplate']:
            f0 = np.fabs(self.fv0.surfVolume())
            f1 = np.fabs(self.fv1[0].surfVolume())
            self.fv0.updateVertices(self.fv0.vertices * (f1/f0)**(1./3))
            m0 = np.mean(self.fv0.vertices, axis=0)
            m1 = np.mean(self.fv1[0].vertices, axis=0)
            self.fv0.updateVertices(self.fv0.vertices + (m1-m0))

        self.fvInit = surfaces.Surface(surf=self.fv0)
        self.fix_orientation()
        if misc is not None and 'subsampleTargetSize' in misc\
           and misc['subsampleTargetSize'] > 0:
            self.fvInit.Simplify(misc['subsampleTargetSize'])
            logging.info('simplified template %d' %(self.fv0.vertices.shape[0]))
        self.fv0.saveVTK(self.outputDir+'/Template.vtk')
        for k, f in enumerate(self.fv1):
            f.saveVTK(self.outputDir+f'/Target{k:03d}.vtk')
        self.dim = self.fv0.vertices.shape[1]

    def set_landmarks(self, landmarks):
        if landmarks is None:
            self.match_landmarks = False
            self.tmpl_lmk = None
            self.targ_lmk = None
            self.def_lmk = None
            self.wlmk = 0
            return

        self.match_landmarks = True
        tmpl_lmk, targ_lmk, self.wlmk = landmarks
        self.tmpl_lmk = pointSets.PointSet(data=tmpl_lmk)
        self.targ_lmk = []
        for tl in targ_lmk:
            self.targ_lmk.append(pointSets.PointSet(data=tl))

    def initial_plot(self):
        pass

    def fix_orientation(self, fv1=None):
        if fv1 is None:
            fv1 = self.fv1
        if fv1 and issubclass(type(fv1[0]), surfaces.Surface):
            self.fv0.getEdges()
            self.closed = self.fv0.bdry.max() == 0
            if self.closed:
                v0 = self.fv0.surfVolume()
                if self.options['errorType'] == 'L2Norm' and v0 < 0:
                    self.fv0.flipFaces()
                    v0 = -v0
                z = self.fvInit.surfVolume()
                if z < 0:
                    self.fv0ori = -1
                else:
                    self.fv0ori = 1
            else:
                self.fv0ori = 1
                v0 = 0
            self.fv1ori = np.zeros(len(self.fv1), dtype=int)
            for k, f in enumerate(fv1):
                f.getEdges()
                closed = self.closed and f.bdry.max() == 0
                if closed:
                    v1 = f.surfVolume()
                    if v0*v1 < 0:
                        f.flipFaces()
                    z = f.surfVolume()
                    if z < 0:
                        self.fv1ori[k] = -1
                    else:
                        self.fv1ori[k] = 1
                else:
                    self.fv1ori[k] = 1
        else:
            self.fv0ori = 1
            self.fv1ori = None
        logging.info('orientation: {0:d}'.format(self.fv0ori))

    def dataTerm(self, _fvDef, var=None):
        obj = 0
        if var is None or 'fv1' not in var:
            fv1 = self.fv1
        else:
            fv1 = var['fv1']
        if self.match_landmarks:
            if var is None or 'lmk_def' not in var:
                _lmk_def = self.def_lmk
            else:
                _lmk_def = var['lmk_def']
            if var is None or 'lmk1' not in var:
                lmk1 = self.targ_lmk
            else:
                lmk1 = var['lmk1']
            for k, s in enumerate(_fvDef):
                obj += super().dataTerm(s, {'fv1':fv1[k], 'lmk_def':_lmk_def[k], 'lmk1':lmk1[k]})
        else:
            for k, s in enumerate(_fvDef):
                obj += super().dataTerm(s, {'fv1': fv1[k]})
                if self.options['volumeWeight']:
                    obj += self.options['volumeWeight'] * (s.surfVolume() - fv1[k].surfVolume()) ** 2
        if self.options['mode'] == 'debug':
            logging.info(f'data term: {obj+self.obj0:.4f}')
        return obj

    def objectiveFun(self):
        if self.obj is None:
            (self.obj, self.state) = self.objectiveFunDef(self.control, withTrajectory=True)
            self.obj0 = 0
            for k in range(self.nTarg):
                if self.options['errorType'] == 'L2Norm':
                    self.obj0 += sd.L2Norm0(self.fv1[k]) / (self.options['sigmaError'] ** 2)
                elif self.fun_obj0 is not None:
                    self.obj0 += self.fun_obj0(self.fv1[k]) / (self.options['sigmaError']**2)
                if self.match_landmarks:
                    self.obj0 += self.wlmk * self.lmk_obj0(self.targ_lmk[k]) / (self.options['sigmaError'] ** 2)
                self.fvDef[k].updateVertices(self.state['xt'][self.jumpIndex[k], :self.nvert, :])
                if self.match_landmarks:
                    self.def_lmk[k].vertices = self.state['xt'][self.jumpIndex[k], self.nvert:, :]
            self.obj += self.obj0 + self.dataTerm(self.fvDef, {'lmk_def':self.def_lmk})
        return self.obj

    def updateTry(self, dr, eps, objRef=None):
        objTry = self.obj0
        controlTry = Control()
        for k in dr.keys():
            if dr[k] is not None:
                controlTry[k] = self.control[k] - eps * dr[k]
        obj_, st = self.objectiveFunDef(controlTry, withTrajectory=True)
        objTry += obj_
        xtTry = st['xt']

        ff = []
        for k in range(self.nTarg):
            ff.append(surfaces.Surface(surf=self.fvDef[k]))
            ff[k].updateVertices(np.squeeze(xtTry[self.jumpIndex[k], :self.nvert, :]))
        if self.match_landmarks:
            pp = []
            for k in range(self.nTarg):
                pp.append(pointSets.PointSet(data=self.def_lmk[k]))
                pp[k].updateVertices(np.squeeze(xtTry[self.jumpIndex[k], self.nvert:, :]))
        else:
            pp = None

        objTry += self.dataTerm(ff, {'lmk_def':pp})
        if np.isnan(objTry):
            print('Warning: nan in updateTry')
            return 1e500

        if (objRef is None) or (objTry < objRef):
            self.controlTry = controlTry
            # self.atTry = atTry
            # self.AfftTry = AfftTry
            self.objTry = objTry
            self.stateTry['xt'] = xtTry

        return objTry

    def testEndpointGradient(self):
        eps0 = 1e-8
        f2 = [[], []]
        p2 = [[], []]
        grd = self.endPointGradient()
        grdscp = 0
        for k, f in enumerate(self.fvDef):
            dff = np.random.normal(size=f.vertices.shape)
            if self.match_landmarks:
                dpp = np.random.normal(size=self.def_lmk[k].vertices.shape)
                dall = np.concatenate((dff, dpp), axis=0)
            else:
                dall = dff
                dpp = None
            grdscp += (grd[k]*dall).sum()
            eps = [-eps0, eps0]
            for j in range(2):
                if self.match_landmarks:
                    pp = pointSets.PointSet(data=self.def_lmk[k])
                    pp.updateVertices(pp.vertices + eps[j] * dpp)
                else:
                    pp = None
                ff = surfaces.Surface(surf=f)
                ff.updateVertices(ff.vertices+eps[j]*dff)
                f2[j].append(ff)
                p2[j].append(pp)
        c0 = self.dataTerm(f2[0], {'lmk_def': p2[0]})
        c1 = self.dataTerm(f2[1], {'lmk_def': p2[1]})
        logging.info("test endpoint gradient: {0:.5f} {1:.5f}".format((c1-c0)/(2*eps0), -grdscp) )

    def endPointGradient(self, endPoint=None):
        if endPoint is None:
            endPoint = self.fvDef
            endPoint_lmk = self.def_lmk
        elif self.match_landmarks:
            endPoint_lmk = endPoint[1]
            endPoint = endPoint[0]
        else:
            endPoint_lmk = None
        px = []
        for k in range(self.nTarg):
            targGradient = np.zeros(endPoint[k].vertices.shape)
            if self.options['errorType'] == 'L2Norm':
                targGradient = -sd.L2NormGradient(endPoint[k], self.fv1[k].vfld)/(self.options['sigmaError']**2)
            else:
                if self.fv1:
                    if self.fun_objGrad is not None:
                        targGradient = -self.fun_objGrad(endPoint[k], self.fv1[k])/(self.options['sigmaError']**2)
                else:
                    targGradient = -self.fun_objGrad(endPoint[k])/(self.options['sigmaError']**2)
            if self.options['volumeWeight']:
                targGradient -= (2./3) * self.options['volumeWeight']*(endPoint[k].surfVolume() - self.fv1[k].surfVolume()) * self.fvDef[k].computeAreaWeightedVertexNormals()
            if self.match_landmarks:
                pxl = -self.wlmk * self.lmk_objGrad(endPoint_lmk[k], self.targ_lmk[k])/(self.options['sigmaError']**2)
                targGradient = np.concatenate((targGradient, pxl), axis=0)
            px.append(targGradient)
        return px

    def hamiltonianCovector(self, px1, KparDiff, regweight,
                            fv0=None, control=None):
        if fv0 is None:
            fv0 = self.fvInit
        if control is None:
            control = self.control
            # at = self.at
            current_at = True
            if self.varCounter == self.trajCounter:
                computeTraj = False
            else:
                computeTraj = True
        else:
            current_at = False
            computeTraj = True

        at = control['at']
        affine = self.affB.getTransforms(control['Afft'])

        if self.match_landmarks:
            x0 = np.concatenate((fv0.vertices, self.tmpl_lmk.vertices), axis=0)
        else:
            x0 = fv0.vertices
        N = x0.shape[0]
        dim = x0.shape[1]
        T = at.shape[0]
        nTarg = len(px1)
        timeStep = self.maxT / T
        if computeTraj:
            st = self.solveStateEquation(control=control, init_state=x0)
            xt = st['xt']
            if current_at:
                self.trajCounter = self.varCounter
                self.state['xt'] = xt
        else:
            xt = self.state['xt']
        pxt = np.zeros([T+1, N, dim])
        pxt[T, :, :] = px1[nTarg - 1]

        if affine is not None:
            A0 = affine[0]
            A = np.zeros([T, dim, dim])
            for k in range(A0.shape[0]):
                A[k, ...] = getExponential(timeStep*A0[k])
        else:
            A = None

        foo = self.createObject(fv0)
        for t in range(T):
            px = np.squeeze(pxt[T - t, :, :])
            z = np.squeeze(xt[T - t-1, :, :])
            a = np.squeeze(at[T - t-1, :, :])
            foo.updateVertices(z)
            v = KparDiff.applyK(z, a)
            if self.internalCost:
                grd = self.internalCostGrad(foo, v)
                Lv = grd['phi']
                DLv = self.options['internalWeight']*grd['x']
                zpx = self.options['KparDiff'].applyDiffKT(z, px, a, regweight=self.options['regWeight'], lddmm=True,
                                                      extra_term = -self.options['internalWeight']*Lv) - DLv
            else:
                zpx = self.options['KparDiff'].applyDiffKT(z, px, a, regweight=self.options['regWeight'], lddmm=True)
            if not (affine is None):
                pxt[T-t-1, :, :] = np.dot(px, A[T-t-1]) + timeStep * zpx
            else:
                pxt[T-t-1, :, :] = px + timeStep * zpx
            pxt[T - t - 1, :, :] += px1[T - t - 1, :, :]

        return pxt, xt

    def getGradient(self, coeff=1.0, update=None):
        if update is None:
            control = self.control
            if self.match_landmarks:
                endPoint = (self.fvDef, self.def_lmk)
            else:
                endPoint = self.fvDef
        else:
            control = Control()
            for k in update[0].keys():
                if update[0][k] is not None:
                    control[k] = self.control[k] - update[1]*update[0][k]
            st = self.solveStateEquation(control=control, init_state=self.control['x0'])
            xt = st['xt']
            if self.match_landmarks:
                endPoint0 = []
                endPoint1 = []
                for k in range(self.nTarg):
                    endPoint0.append(surfaces.Surface(surf=self.fv0))
                    endPoint0[k].updateVertices(xt[self.jumpIndex[k], :self.nvert, :])
                    endPoint1.append(pointSets.PointSet(data=xt[self.jumpIndex[k], self.nvert:, :]))
                endPoint = (endPoint0, endPoint1)
            else:
                endPoint = []
                for k in range(self.nTarg):
                    fvDef = surfaces.Surface(surf=self.fv0)
                    fvDef.updateVertices(xt[self.jumpIndex[k], :, :])
                    endPoint.append(fvDef)

        px1 = np.zeros(self.state['xt'].shape)
        px1_ = self.endPointGradient(endPoint=endPoint)
        nTarg = len(px1_)
        px1[-1, :, :] = px1_[nTarg-1]
        jk = nTarg - 2
        T = self.state['xt'].shape[0]-1
        for t in range(T):
            if (t < T - 1) and self.isjump[T - t - 1]: ###NEED TO CHECK
                px1[T - t - 1, :, :] = px1_[jk]
                jk -= 1

        dim2 = self.dim**2
        dt = self.maxT / self.Tsize

        foo = self.hamiltonianGradient(px1, control=control)
        grd = Control()
        grd['at'] = (dt/coeff) * foo['dat']
        grd['x0'] = np.zeros((self.npt, self.dim))

        if self.affineDim > 0:
            dA = foo['dA']
            db = foo['db']
            grd['Afft'] = 2*self.affineWeight.reshape([1, self.affineDim]) * self.control['Afft']
            for t in range(self.Tsize):
               dAff = self.affineBasis.T @ np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])])
               grd['Afft'][t] -=  dAff.reshape(grd['Afft'][t].shape)
            grd['Afft'] *= dt / (self.coeffAff*coeff)
        return grd

    def saveCorrectedTarget(self, X0, X1):
        for k, fv in enumerate(self.fv1):
            U = la.inv(X0[self.jumpIndex[k]])
            if self.options['errorType'] == "PointSet":
                f = pointSets.PointSet(data=fv)
                yyt = np.dot(f.vertices - X1[self.jumpIndex[k], ...], U.T)
                f.vertices = yyt
            else:
                f = surfaces.Surface(surf=fv)
                yyt = np.dot(f.vertices - X1[self.jumpIndex[k], ...], U.T)
                f.updateVertices(yyt)
            if self.match_landmarks:
                p = pointSets.PointSet(data=self.targ_lmk[k])
                yyt = np.dot(p.vertices - X1[-1, ...], U)
                p.updateVertices(yyt)
                p.saveVTK(self.outputDir + f'/Target{k:02d}LandmarkCorrected.vtk')
            f.saveVTK(self.outputDir + f'/Target{k:02d}Corrected.vtk')

    def saveHdf5(self, fileName):
        pass

    def updateEndPoint(self, st):
        for k in range(self.nTarg):
            self.fvDef[k].updateVertices(st['xt'][self.jumpIndex[k], :self.nvert, :])
            if self.match_landmarks:
                self.def_lmk[k].updateVertices(st['xt'][self.jumpIndex[k], self.nvert:, :])

