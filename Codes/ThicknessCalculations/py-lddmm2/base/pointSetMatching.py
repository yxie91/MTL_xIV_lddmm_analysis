from copy import deepcopy
import numpy as np
import scipy.linalg as la
import logging
from functools import partial
from . import (conjugateGradient as cg,
               kernelFunctions as kfun,
               pointEvolution as evol,
               pointEvolutionSemiReduced as evolSR,
               grid,
               bfgs,
               sgd)
from .pointSets import PointSet, saveTrajectories, savePoints
from . import pointsetDistances as psd
from .affineBasis import AffineBasis
from .basicMatching import BasicMatching
from .vtk_fields import vtkFields
from numpy.random import default_rng
rng = default_rng()


# Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'
# class PointSetMatchingParam(matchingParam.MatchingParam):
#     def __init__(self, timeStep = .1, algorithm='cg', Wolfe=True, KparDiff = None, KparDist = None,
#                  sigmaError = 1.0, errorType = 'measure'):
#         super().__init__(timeStep=timeStep, algorithm = algorithm, Wolfe=Wolfe,
#                          KparDiff = KparDiff, KparDist = KparDist, sigmaError=sigmaError,
#                          errorType = errorType)
#         self.sigmaError = sigmaError
#

class Control(dict):
    def __init__(self):
        super().__init__()
        self['at'] = None
        self['ct'] = None
        self['Afft'] = None


class State(dict):
    def __init__(self):
        super().__init__()
        self['xt'] = None
        self['yt'] = None
        self['Jt'] = None


# Main class for point set matching
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
class PointSetMatching(BasicMatching):
    def __init__(self, Template=None, Target=None, options=None):
        super().__init__(Template, Target, options)

        if self.options['algorithm'] == 'cg':
            self.euclideanGradient = False
        else:
            self.euclideanGradient = True

        self.Kdiff_dtype = self.options['pk_dtype']
        self.Kdist_dtype = self.options['pk_dtype']
        self.gradCoeff = 1
        self.setDotProduct(self.options['unreduced'])
        if self.options['algorithm'] == 'sgd':
            self.sgd = True
            self.unreduced = True
        else:
            self.sgd = False
            self.unreduced = self.options['unreduced']

        self.options['unreducedWeight'] *= 1000.0 / self.fv0.vertices.shape[0]
        self.ds = 1.

        if self.options['algorithm'] == 'sgd':
            self.set_sgd()

    def getDefaultOptions(self):
        options = super().getDefaultOptions()
        options['reweightCells'] = False
        options['unreducedResetRate'] = -1
        options['checkOrientation'] = True
        # options['fidelityWeight'] = 0.
        return options

    def set_passenger(self, passenger):
        self.passenger = passenger
        if isinstance(self.passenger, type(self.fv0)):
            self.passenger_points = self.passenger.vertices
        elif self.passenger is not None:
            self.passenger_points = self.passenger
        else:
            self.passenger_points = None
        self.passengerDef = deepcopy(self.passenger)

    def initialize_variables(self):
        self.x0 = np.copy(self.fv0.vertices)
        self.nvert = self.fv0.vertices.shape[0]
        if self.match_landmarks:
            self.x0 = np.concatenate(
                (self.fv0.vertices, self.tmpl_lmk.vertices), axis=0)
            self.nlmk = self.tmpl_lmk.vertices.shape[0]
        else:
            self.x0 = np.copy(self.fv0.vertices)
            self.nlmk = 0
        self.fvDef = deepcopy(self.fv0)
        if self.match_landmarks:
            self.def_lmk = PointSet(data=self.tmpl_lmk)
        self.npt = self.x0.shape[0]

        self.control = Control()
        self.controlTry = Control()
        self.Tsize = int(round(self.maxT/self.options['timeStep']))
        self.control['at'] = np.zeros([self.Tsize, self.x0.shape[0],
                                       self.x0.shape[1]])
        if self.randomInit:
            self.control['at'] = np.random.normal(0, .01,
                                                  self.control['at'].shape)
        self.controlTry['at'] = np.zeros([self.Tsize, self.x0.shape[0],
                                          self.x0.shape[1]])
        if self.affineDim > 0:
            self.control['Afft'] = np.zeros([self.Tsize, self.affineDim])
            self.controlTry['Afft'] = np.zeros([self.Tsize, self.affineDim])
        else:
            self.control['Afft'] = None
            self.controlTry['Afft'] = None

        self.state = State()
        self.state['xt'] = np.tile(self.x0, [self.Tsize+1, 1, 1])
        self.stateTry = State()
        self.stateTry['xt'] = np.copy(self.state['xt'])
        self.v = np.zeros([self.Tsize+1, self.npt, self.dim])

        if self.options['unreduced']:
            self.control['ct'] = np.tile(self.x0, [self.Tsize, 1, 1])
            if self.randomInit:
                self.control['ct'] += np.random.normal(0, 1,
                                                       self.control['ct'].shape)
            self.controlTry['ct'] = np.copy(self.control['ct'])
        else:
            self.control['ct'] = None
            self.controlTry['ct'] = None

        if self.options['algorithm'] == 'sgd':
            self.SGDSelectionPts = [None, None]

        self.passenger_points = None
        self.saveFileList = []
        for kk in range(self.Tsize+1):
            self.saveFileList.append(self.options['saveFile'] + f'{kk:03d}')

        if self.dim == 2:
            xmin = min(self.fv0.vertices[:, 0].min(),
                       self.fv1.vertices[:, 0].min())
            xmax = max(self.fv0.vertices[:, 0].max(),
                       self.fv1.vertices[:, 0].max())
            ymin = min(self.fv0.vertices[:, 1].min(),
                       self.fv1.vertices[:, 1].min())
            ymax = max(self.fv0.vertices[:, 1].max(),
                       self.fv1.vertices[:, 1].max())
            dx = 0.01*(xmax-xmin)
            dy = 0.01*(ymax-ymin)
            dxy = min(dx, dy)
            [x, y] = np.mgrid[(xmin-10*dxy):(xmax+10*dxy):dxy,
                              (ymin-10*dxy):(ymax+10*dxy):dxy]
            self.gridDef = grid.Grid(gridPoints=[x, y])
            self.gridxy = np.copy(self.gridDef.vertices)

    def set_sgd(self, control=100, template=100, target=100):
        logging.info("setting SGD parameters")
        self.weightSubset = 0.
        self.sgdEpsInit = 1.

        self.sgdNormalization = 'var'
        self.sgdBurnIn = 10000
        self.sgdMeanSelectControl = control
        self.sgdMeanSelectTemplate = template
        self.sgdMeanSelectTarget = target
        self.SGDFreezePointsCount = 0
        self.probSelectControl =\
            min(1.0, self.sgdMeanSelectControl / self.fv0.vertices.shape[0])
        self.probSelectControlPair =\
            min(1.0,
                self.sgdMeanSelectControl * (self.sgdMeanSelectControl-1) /
                (self.fv0.vertices.shape[0]*(self.fv0.vertices.shape[0]-1)))
        self.probSelectFaceTemplate =\
            min(1.0, self.sgdMeanSelectTemplate / self.fv0.faces.shape[0])
        self.probSelectFaceTemplatePair =\
            min(1.0,
                self.sgdMeanSelectTemplate * (self.sgdMeanSelectTemplate-1) /
                (self.fv0.faces.shape[0] * (self.fv0.faces.shape[0]-1)))
        self.probSelectFaceTarget =\
            min(1.0, self.sgdMeanSelectTarget / self.fv1.faces.shape[0])
        self.probSelectVertexTemplate = np.ones(self.fv0.vertices.shape[0])
        nf = np.zeros(self.fv0.vertices.shape[0])
        for k in range(self.fv0.faces.shape[0]):
            for j in range(self.fv0.faces.shape[1]):
                self.probSelectVertexTemplate[self.fv0.faces[k, j]] *= \
                    1 - self.sgdMeanSelectTemplate/(self.fv0.faces.shape[0]
                                                    - nf[self.fv0.faces[k, j]])
                nf[self.fv0.faces[k, j]] += 1
        self.probSelectVertexTemplate = 1 - self.probSelectVertexTemplate

    def setDotProduct(self, unreduced=False):
        if self.options['algorithm'] == 'cg' and not unreduced:
            self.euclideanGradient = False
            self.dotProduct = self.dotProduct_Riemannian
        else:
            self.euclideanGradient = True
            self.dotProduct = self.dotProduct_euclidean

    def set_template_and_target(self, Template, Target, misc=None):
        if Template is None:
            logging.error('Please provide a template surface')
            return
        else:
            self.fv0 = PointSet(data=Template)

        if Target is None:
            logging.error('Please provide a target surface')
            return
        else:
            self.fv1 = PointSet(data=Target)

        self.fv0.save(self.outputDir + '/Template.vtk')
        self.fv1.save(self.outputDir + '/Target.vtk')
        self.dim = self.fv0.vertices.shape[1]

    def set_fun(self, errorType, vfun=None):
        self.options['errorType'] = errorType
        if errorType == 'L2':
            self.fun_obj0 = psd.L2Norm0
            self.fun_obj = psd.L2NormDef
            self.fun_objGrad = psd.L2NormGradient
            self.sgdSelection = 'single'
        elif errorType == 'measure':
            self.fun_obj0 = partial(psd.measureNorm0,
                                    KparDist=self.options['KparDist'])
            self.fun_obj = partial(psd.measureNormDef,
                                   KparDist=self.options['KparDist'])
            self.fun_objGrad = partial(psd.measureNormGradient,
                                       KparDist=self.options['KparDist'])
            self.sgdSelection = 'pair'
        else:
            self.sgdSelection = None
            logging.error('Unknown error Type: ' + self.options['errorType'])

        if self.match_landmarks:
            self.lmk_obj0 = psd.L2Norm0
            self.lmk_obj = psd.L2NormDef
            self.lmk_objGrad = psd.L2NormGradient
        else:
            self.lmk_obj0 = None
            self.lmk_obj = None
            self.lmk_objGrad = None

        self.set_extra_term()

    def set_extra_term(self):
        self.extraTerm = None

    def set_parameters(self):
        super().set_parameters()
        sigmaKernel = 6.5
        orderKernel = 3
        sigmaDist = 2.5
        orderKDist = 3
        self.reset = True
        self.resetPK_ = True
        self.PKResetCount = 0
        self.TargetObj = -np.inf
        self.maxPKReset = 1
        typeKDist = 'gauss'
        typeKernel = 'gauss'

        if type(self.options['KparDiff']) in (list, tuple):
            typeKernel = self.options['KparDiff'][0]
            sigmaKernel = self.options['KparDiff'][1]
            if typeKernel == 'laplacian' and len(self.options['KparDiff']) > 2:
                orderKernel = self.options['KparDiff'][2]
            self.options['KparDiff'] = None

        if self.options['KparDiff'] is None:
            self.options['KparDiff'] = kfun.Kernel(name=typeKernel,
                                                   sigma=sigmaKernel,
                                                   order=orderKernel)

        if type(self.options['KparDist']) in (list, tuple):
            typeKDist = self.options['KparDist'][0]
            sigmaDist = self.options['KparDist'][1]
            if typeKDist == 'laplacian' and len(self.options['KparDist']) > 2:
                orderKdist = self.options['KparDist'][2]
            self.options['KparDist'] = None

        if self.options['KparDist'] is None:
            self.options['KparDist'] = kfun.Kernel(name=typeKDist,
                                                   sigma=sigmaDist,
                                                   order=orderKDist)

        self.options['KparDiff'].pk_dtype = self.options['pk_dtype']
        self.options['KparDist'].pk_dtype = self.options['pk_dtype']

        self.gradEps = self.options['gradTol']
        self.epsInit = self.options['epsInit']
        self.affineOnly = self.options['affineOnly']

        if self.options['affineKernel']:
            if self.options['affine'] in ('euclidean', 'affine'):
                if self.options['affine'] == 'euclidean' and self.options['rotWeight'] is not None:
                    w1 = self.options['rotWeight']
                else:
                    w1 = self.options['affineWeight']
                if self.options['transWeight'] is not None:
                    w2 = self.options['transWeight']
                else:
                    w2 = self.options['affineWeight']
                self.options['KparDiff'].setAffine(self.options['affine'],
                                                   w1=1/w1, w2=1/w2,
                                                   center=self.fv0.vertices.mean(axis=0))
                self.options['affine'] = 'none'
                self.affB = AffineBasis(self.dim, 'none')
                self.affineDim = self.affB.affineDim
            else:
                logging.info('Affine kernels only Euclidean or full affine')
                self.options['affineKernel'] = False

        if not self.options['affineKernel']:
            self.affB = AffineBasis(self.dim, self.options['affine'])
            self.affineDim = self.affB.affineDim
            self.affineBasis = self.affB.basis
            self.affineWeight = self.options['affineWeight'] * np.ones([self.affineDim, 1])
            if (len(self.affB.rotComp) > 0) and (self.options['rotWeight'] is not None):
                self.affineWeight[self.affB.rotComp] = self.options['rotWeight']
            if (len(self.affB.simComp) > 0) and (self.options['scaleWeight'] is not None):
                self.affineWeight[self.affB.simComp] = self.options['scaleWeight']
            if (len(self.affB.transComp) > 0) and (self.options['transWeight'] is not None):
                self.affineWeight[self.affB.transComp] = self.options['transWeight']

        self.coeffInitx = .1
        self.forceLineSearch = False
        self.saveEPDiffTrajectories = True
        self.varCounter = 0
        self.trajCounter = 0
        self.pkBuffer = 0

        self.unreducedResetRate = self.options['unreducedResetRate']
        # self.fidelityWeight = self.options['fidelityWeight']
        self.obj_unreduced_save = np.inf
        self.unreducedRecomputeWeightsOnly = False
        self.unreducedRecomputeWeightsMax = max(self.unreducedResetRate//2,
                                                100)
        if self.options['unreduced']:
            logging.info(f'unreduced reset: {self.unreducedResetRate}, {self.unreducedRecomputeWeightsMax}')

    def getInitState(self):
        if self.match_landmarks:
            x0 = np.concatenate((self.fv0.vertices, self.tmpl_lmk.vertices),
                                axis=0)
        else:
            x0 = self.fv0.vertices
        return x0

    def createObject(self, data, other=None):
        if other is None:
            return PointSet(data=data)
        else:
            return PointSet(data=data, weights=other)

    def updateObject(self, object, data, other=None):
        return object.updateVertices(data)

    def solveStateEquation(self, control=None, init_state=None, kernel=None,
                           options=None):
        if control is None:
            control = self.control
        if init_state is None:
            init_state = self.getInitState()
        if kernel is None:
            kernel = self.options['KparDiff']

        A = self.affB.getTransforms(control['Afft'])

        if self.unreduced:
            return evolSR.landmarkSemiReducedEvolutionEuler(init_state,
                                                            control['ct'],
                                                            control['at'],
                                                            kernel,
                                                            affine=A,
                                                            options=options)
        else:
            return evol.landmarkDirectEvolutionEuler(init_state, control['at'],
                                                     kernel,
                                                     affine=A,
                                                     options=options,
                                                     T=self.maxT)

    def set_landmarks(self, landmarks):
        if landmarks is None:
            self.match_landmarks = False
            self.tmpl_lmk = None
            self.targ_lmk = None
            self.def_lmk = None
            self.wlmk = 0
            return

        logging.info('Found landmarks')
        self.match_landmarks = True
        tmpl_lmk, targ_lmk, self.wlmk = landmarks
        self.tmpl_lmk = PointSet(data=tmpl_lmk)
        self.targ_lmk = PointSet(data=targ_lmk)

    def dataTerm(self, _fvDef, var=None):
        if type(var) is dict and 'fv1' in var:
            fv1 = var['fv1']
        else:
            fv1 = self.fv1
        obj = self.fun_obj(_fvDef, fv1) / (self.options['sigmaError']**2)

        if self.match_landmarks:
            if var is None or 'lmk_def' not in var:
                logging.error('Data term: Missing deformed landmarks')
            if var is not None and 'lmk1' in var:
                lmk1 = var['lmk1']
            else:
                lmk1 = self.targ_lmk
            obj += self.wlmk * self.lmk_obj(var['lmk_def'], lmk1)\
                / (self.options['sigmaError']**2)

        return obj

    def objectiveFunDef(self, control, var=None, withTrajectory=False,
                        withJacobian=False):
        if var is None or 'x0' not in var:
            x0 = self.x0
        else:
            x0 = var['x0']

        if var is None or 'kernel' not in var:
            kernel = self.options['KparDiff']
        else:
            kernel = var['kernel']

        if var is None or 'regWeight' not in var:
            regWeight = self.options['regWeight']
        else:
            regWeight = var['regWeight']

        if 'Afft' not in control:
            Afft = None
        else:
            Afft = control['Afft']

        if self.unreduced:
            ct = control['ct']
            at = control['at']
        else:
            ct = None
            at = control['at']

        timeStep = self.maxT/self.Tsize

        st = self.solveStateEquation(control=control, init_state=x0,
                                     kernel=kernel,
                                     options={'withJacobian': withJacobian})
        obj = 0
        obj1 = 0
        obj2 = 0
        obj3 = 0
        for t in range(self.Tsize):
            z = st['xt'][t, :, :]
            a = at[t, :, :]
            if self.unreduced:
                c = ct[t, :, :]
            else:
                c = None

            if self.unreduced:
                ca = kernel.applyK(c, a)
                ra = kernel.applyK(c, a, firstVar=z)
                obj += regWeight * timeStep * (a * ca).sum()
                obj3 += self.options['unreducedWeight'] \
                    * timeStep * ((c - z)**2).sum()
            else:
                ra = kernel.applyK(z, a)
                obj += regWeight*timeStep*(a*ra).sum()

            if hasattr(self, 'v'):
                self.v[t, :] = ra

            if self.extraTerm is not None:
                obj1 += self.extraTerm['coeff'] \
                    * self.extraTerm['fun'](z, ra) * timeStep
            if self.affineDim > 0:
                obj2 += timeStep * (self.affineWeight.reshape(Afft[t].shape)
                                    * Afft[t]**2).sum()

        obj += obj1 + obj2 + obj3
        if withTrajectory or withJacobian:
            return obj, st
        else:
            return obj

    def makeTryInstance(self, state):
        ff = self.createObject(state['xt'][-1, :, :],
                               other=self.fv0.vertex_weights)
        return ff

    def objectiveFun(self):
        if self.obj is None:
            self.obj0 = self.fun_obj0(self.fv1) / \
                (self.options['sigmaError']**2)
            if self.match_landmarks:
                self.obj0 += self.wlmk * self.lmk_obj0(self.targ_lmk) \
                    / (self.options['sigmaError']**2)
            self.objDef, self.state \
                = self.objectiveFunDef(self.control, withTrajectory=True)
            self.fvDef.updateVertices(self.state['xt'][-1, :, :])
            if self.match_landmarks:
                self.def_lmk.vertices = self.state['xt'][-1, self.nvert:, :]
            self.objData = self.dataTerm(self.fvDef)
            self.obj = self.obj0 + self.objData + self.objDef
        return self.obj

    def getVariable(self):
        return self.control

    def initVariable(self):
        return Control()

    def update(self, dr, eps):
        for k in dr.keys():
            if dr[k] is not None:
                self.control[k] -= eps * dr[k]

    def updateTry(self, dr, eps, objRef=None):
        controlTry = self.initVariable()
        for k in dr.keys():
            if dr[k] is not None:
                controlTry[k] = self.control[k] - eps * dr[k]
            else:
                controlTry[k] = self.control[k]

        objTryDef, st = self.objectiveFunDef(controlTry, withTrajectory=True)
        ff = self.makeTryInstance(st)
        if self.match_landmarks:
            pp = PointSet(data=self.def_lmk)
            pp.updateVertices(st['xt'][-1, self.nvert:, :])
        else:
            pp = None

        objTryData = self.dataTerm(ff, {'lmk_def': pp})
        objTry = self.obj0 + objTryData + objTryDef

        if np.isnan(objTry):
            # logging.info('Warning: nan in updateTry')
            return 1e500

        if (objRef is None) or (objTry < objRef):
            self.controlTry = deepcopy(controlTry)
            self.stateTry = deepcopy(st)
            self.objTry = objTry
            self.objTryData = objTryData
            self.objTryDef = objTryDef

        return objTry

    def testEndpointGradient(self):
        ff = deepcopy(self.fvDef)
        dff = np.random.normal(size=ff.vertices.shape)
        if self.match_landmarks:
            dpp = np.random.normal(size=self.def_lmk.vertices.shape)
            dall = np.concatenate((dff, dpp), axis=0)
        else:
            dall = dff
            dpp = None
        eps0 = 1e-6
        c = []
        for eps in [-eps0, eps0]:
            ff = self.createObject(self.fvDef)
            ff.updateVertices(ff.vertices+eps*dff)
            if self.match_landmarks:
                pp = PointSet(data=self.def_lmk)
                pp.updateVertices(pp.vertices + eps * dpp)
            else:
                pp = None
            c.append(self.dataTerm(ff, {'lmk_def': pp}))
        ff.vertices += eps*dff
        grd = self.endPointGradient()
        logging.info(f"test endpoint gradient: {(c[1]-c[0])/(2*eps):.5f} {(grd*dall).sum():.5f}")
        logging.info(f'{c[0]+self.obj0:.5f}')

    def endPointGradient(self, endPoint=None):
        if endPoint is None:
            endPoint = self.fvDef
            endPoint_lmk = self.def_lmk
        elif self.match_landmarks:
            endPoint_lmk = endPoint[1]
            endPoint = endPoint[0]
        else:
            endPoint_lmk = None

        if self.options['errorType'] == 'measure':
            px = self.fun_objGrad(endPoint, self.fv1)
        else:
            px = self.fun_objGrad(endPoint, self.fv1)

        if self.match_landmarks:
            pxl = self.wlmk*self.lmk_objGrad(endPoint_lmk, self.targ_lmk)
            px = np.concatenate((px, pxl), axis=0)

        return px / self.options['sigmaError']**2

    def diagonalTermCorrectionEndCost(self, fv0, endPoint=None):
        p0 = self.probSelectFaceTemplate
        sqp0 = self.probSelectFaceTemplatePair
        s0 = (1 / sqp0 - 1 / p0)
        return np.zeros(fv0.vertices.shape), fv0.faces.shape[0] * s0

    def SGDSelectBatches(self):
        if self.sgdMeanSelectTemplate >= self.fv0.faces.shape[0]:
            I0 = np.arange(self.fv0.faces.shape[0])
        else:
            I0 = rng.choice(self.fv0.faces.shape[0],
                            self.sgdMeanSelectTemplate, replace=False)

        SGDBatches = dict()
        SGDBatches['stateFaces'] = I0
        select0 = np.zeros(self.fv0.faces.shape[0], dtype=bool)
        select0[I0] = True
        fv0, I00 = self.fv0.select_faces(select0)
        SGDBatches['stateVertexes'] = I00
        fv0.face_weights /= np.sqrt(self.probSelectFaceTemplatePair)

        SGDBatches['fv0'] = fv0

        if self.sgdMeanSelectTarget >= self.fv1.faces.shape[0]:
            I1 = np.arange(self.fv1.faces.shape[0])
        else:
            I1 = rng.choice(self.fv1.faces.shape[0],
                            self.sgdMeanSelectTarget, replace=False)

        SGDBatches['targetFaces'] = I1
        select1 = np.zeros(self.fv1.faces.shape[0], dtype=bool)
        select1[I1] = True
        fv1, I11 = self.fv1.select_faces(select1)
        p1 = self.probSelectFaceTarget * self.probSelectFaceTemplate\
            / np.sqrt(self.probSelectFaceTemplatePair)
        fv1.face_weights /= p1
        SGDBatches['targetVertexes'] = I11
        SGDBatches['fv1'] = fv1

        if self.sgdMeanSelectControl <= self.control['ct'].shape[1]:
            J0 = rng.choice(self.control['ct'].shape[1],
                            self.sgdMeanSelectControl, replace=False)
        else:
            J0 = np.arange(self.control['ct'].shape[1])

        SGDBatches['controlPoints'] = J0

        return SGDBatches

    def endPointGradientSGD(self, SGDBatches=None):
        if SGDBatches is None:
            SGDBatches = self.SGDSelectBatches()
        st = self.solveStateEquation(init_state=SGDBatches['fv0'].vertices)
        xt = st['xt']
        endPoint = self.createObject(SGDBatches['fv0'])
        endPoint.updateVertices(xt[-1, :, :])
        px_ = self.fun_objGrad(endPoint, SGDBatches['fv1'])
        if self.sgdMeanSelectTemplate < self.fv0.faces.shape[0]:
            dpx, dobj = self.diagonalTermCorrectionEndCost(SGDBatches['fv0'])
            px_ -= dpx
        else:
            dobj = 0

        obj = (self.fun_obj(endPoint, SGDBatches['fv1']) - dobj) \
            / (self.options['sigmaError'] ** 2)
        return px_ / self.options['sigmaError'] ** 2, xt, SGDBatches, obj

    def checkSGDEndpointGradient(self):
        endPoint = self.createObject(self.fv0)
        st = self.solveStateEquation(init_state=self.fv0.vertices)
        xt = st['xt']
        endPoint.updateVertices(xt[-1, :, :])
        objTrue = self.fun_obj(endPoint, self.fv1) \
            / (self.options['sigmaError']**2)
        pxTrue = self.endPointGradient(endPoint=endPoint)

        px = np.zeros(pxTrue.shape)
        vpx = np.zeros(pxTrue.shape)
        obj = 0
        nsim = 500
        count = np.zeros(self.fv0.vertices.shape[0])
        for k in range(nsim):
            px_, xt2, SGDBatches, obj_ = self.endPointGradientSGD()
            # logging.info(f'{np.fabs(xt2- xt[:, selection[0], :]).sum():.6f}')
            px[SGDBatches['stateVertexes'], :] += px_
            vpx[SGDBatches['stateVertexes'], :] += px_**2
            count[SGDBatches['stateVertexes']] += 1
            obj += obj_

        px /= nsim
        vpx = vpx/nsim - px**2
        obj /= nsim
        count /= nsim
        diff = np.sqrt(nsim)*(np.fabs(px - pxTrue) / np.sqrt(vpx)).mean()
        logging.info(f'check SGD end point: {objTrue:.4f} {obj:.4f} gradient: {diff:.4f}')

    def hamiltonianGradient(self, px1, kernel=None, regWeight=None, x0=None,
                            control=None):
        if regWeight is None:
            regWeight = self.options['regWeight']
        if x0 is None:
            x0 = self.x0

        if control is None:
            control = self.control
        affine = self.affB.getTransforms(control['Afft'])
        if kernel is None:
            kernel = self.options['KparDiff']

        if self.options['unreduced']:
            return evolSR.landmarkSemiReducedHamiltonianGradient(x0,
                                                                 control['ct'],
                                                                 control['at'],
                                                                 px1,
                                                                 kernel,
                                                                 regWeight,
                                                                 weightSubset=self.options['unreducedWeight'],
                                                                 affine=affine, getCovector=True)
        else:
            return evol.landmarkHamiltonianGradient(x0, control['at'], px1,
                                                    kernel,
                                                    regWeight,
                                                    affine=affine,
                                                    getCovector=True,
                                                    extraTerm=self.extraTerm,
                                                    euclidean=self.euclideanGradient,
                                                    T=self.maxT)

    def setUpdate(self, update):
        control = Control()
        for k in update[0].keys():
            if update[0][k] is not None:
                control[k] = self.control[k] - update[1] * update[0][k]
        st = self.solveStateEquation(control=control, init_state=self.x0)
        xt = st['xt']
        if self.match_landmarks:
            endPoint0 = self.createObject(self.fv0)
            endPoint0.updateVertices(xt[-1, :self.nvert, :])
            endPoint1 = PointSet(data=xt[-1, self.nvert:, :])
            endPoint = (endPoint0, endPoint1)
        else:
            endPoint = self.createObject(self.fv0)
            endPoint.updateVertices(xt[-1, :, :])
        st = State()
        st['xt'] = xt

        return control, st, endPoint

    def objectiveFunSGD(self, SGDBatches=None, control=None):
        if SGDBatches is None:
            SGDBatches = self.SGDSelectBatches()
        if control is None:
            control = self.control
        st = self.solveStateEquation(init_state=SGDBatches['fv0'].vertices,
                                     control=control)
        xt = st['xt']
        endPoint = self.createObject(SGDBatches['fv0'])
        endPoint.updateVertices(xt[-1, :, :])
        if self.sgdMeanSelectTemplate < self.fv0.faces.shape[0]:
            _, dobj = self.diagonalTermCorrectionEndCost(SGDBatches['fv0'])
        else:
            dobj = 0

        obj = (self.fun_obj(endPoint, SGDBatches['fv1']) - dobj) \
            / (self.options['sigmaError'] ** 2)

        IJ = np.intersect1d(SGDBatches['stateVertexes'],
                            SGDBatches['controlPoints'])
        control_ = Control()
        control_['ct'] = self.control['ct'][:, SGDBatches['controlPoints'], :]
        control_['at'] = self.control['at'][:, SGDBatches['controlPoints'], :]
        controlProb = self.probSelectControl
        M = self.control['at'].shape[1]
        pprob = controlProb * (M * controlProb - 1) / (M - 1)
        dprob = 1 / controlProb - 1 / pprob

        objDef = 0
        kernel = self.options['KparDiff']
        timeStep = 1/control_['at'].shape[0]
        regWeight = self.options['regWeight']
        stateProb = self.probSelectVertexTemplate
        for t in range(control_['at'].shape[0]):
            c = control_['ct'][t, :, :]
            a = control_['at'][t, :, :]
            z = xt[t, :, :]
            x = np.zeros(self.x0.shape)
            x[SGDBatches['stateVertexes'], :] = xt[t, :, :]
            ca = kernel.applyK(c,a)
            objDiag = (1/controlProb) * (c**2).sum() \
                      - (2 /controlProb) * (self.control['ct'][t, IJ, :] * x[IJ, :]/stateProb[IJ, None]).sum() \
                      + (z**2/stateProb[SGDBatches['stateVertexes'], None]).sum()
            objDef += (1/pprob) * regWeight * timeStep * ((a * ca).sum()  + dprob * (a**2).sum() )+ \
                   self.options['unreducedWeight'] * timeStep * objDiag

        return objDef + obj

    def checkSGDObjective(self):
        endPoint = self.createObject(self.fv0)
        st = self.solveStateEquation(init_state=self.fv0.vertices)
        xt = st['xt']
        endPoint.updateVertices(xt[-1, :, :])

        objTrue = self.objectiveFunDef(self.control) \
            + self.fun_obj(endPoint, self.fv1) / (self.options['sigmaError']**2)
        obj = 0
        nsim = 500
        for k in range(nsim):
            obj_ = self.objectiveFunSGD()
            # logging.info(f'{np.fabs(xt2- xt[:, selection[0], :]).sum():.6f}')
            obj += obj_

        obj /= nsim
        logging.info(f'check SGD objective: {objTrue:.4f} {obj:.4f}')

    def getGradientSGD(self, coeff=1.0, SGDBatches=None, test=False,
                       updateState=False):
        if test and self.options['mode'] == 'debug':
            self.checkSGDObjective()
            self.TestGradientSGD()
        A = self.affB.getTransforms(self.control['Afft'])
        px1, xt, SGDBatches, _ = self.endPointGradientSGD(SGDBatches=SGDBatches)
        if updateState:
            self.state['xt'][:, SGDBatches['stateVertexes'], :] = xt

        foo = evolSR.landmarkSemiReducedHamiltonianGradient(self.x0,
                                                            self.control['ct'],
                                                            self.control['at'],
                                                            -px1,
                                                            self.options['KparDiff'],
                                                            self.options['regWeight'],
                                                            getCovector = True,
                                                            affine = A,
                                                            weightSubset=self.options['unreducedWeight'],
                                                            controlSubset = SGDBatches['controlPoints'],
                                                            stateSubset=SGDBatches['stateVertexes'],
                                                            controlProb=self.probSelectControl,
                                                            stateProb=self.probSelectVertexTemplate,
                                                            forwardTraj=xt)
        dt = self.maxT / self.Tsize
        dim2 = self.dim**2
        grd = Control()
        if self.unreducedRecomputeWeightsOnly:
            grd['ct'] = np.zeros(foo['dct'].shape)
            self.SGDFreezePointsCount += 1
        else:
            grd['ct'] = (dt/coeff) * foo['dct']
        grd['at'] = (dt/coeff) * foo['dat']

        if self.affineDim > 0:
            grd['Afft'] = np.zeros(self.control['Afft'].shape)
            dA = foo['dA']
            db = foo['db']
            grd['Afft'] = 2*self.affineWeight.reshape([1, self.affineDim])\
                * self.control['Afft']
            for t in range(self.Tsize):
                dAff = self.affineBasis.T @ np.vstack([dA[t].reshape([dim2, 1]),
                                                       db[t].reshape([self.dim, 1])])
                grd['Afft'][t] -= dAff.reshape(grd['Afft'][t].shape)
            grd['Afft'] *= dt / (self.coeffAff*coeff)
        else:
            grd['Afft'] = None
        return grd

    def TestGradientSGD(self):
        SGDBatches = self.SGDSelectBatches()
        grd0 = self.getGradientSGD(SGDBatches=SGDBatches, test=False,
                                   updateState=False)
        eps = 1e-8
        dir = self.randomDir(ControlBatch=SGDBatches['controlPoints'])
        message = 'Test SGD Gradient: '
        for key in self.control.keys():
            if self.control[key] is not None:
                control1 = deepcopy(self.control)
                control2 = deepcopy(self.control)
                dcontrol = dir[key]
                control1[key] += eps * dcontrol
                control2[key] -= eps * dcontrol
                expansion = (dcontrol * grd0[key]).sum()
                obj1 = self.objectiveFunSGD(SGDBatches=SGDBatches,
                                            control=control1)
                obj2 = self.objectiveFunSGD(SGDBatches=SGDBatches,
                                            control=control2)
                message += \
                    f'  {key}: {(obj1 - obj2)/(2*eps):.5f}, {expansion:.5f}'
        logging.info(message)

    def getGradient(self, coeff=1.0, update=None, forceComplete=False):
        if self.options['algorithm'] == 'sgd' and not forceComplete:
            return self.getGradientSGD(coeff=coeff, updateState=True)
        if update is None:
            control = self.control
            if self.match_landmarks:
                endPoint = (self.fvDef, self.def_lmk)
            else:
                endPoint = self.fvDef
        else:
            control, state, endPoint = self.setUpdate(update)

        dim2 = self.dim**2
        dt = self.maxT / self.Tsize
        px1 = -self.endPointGradient(endPoint=endPoint)
        foo = self.hamiltonianGradient(px1, control=control)
        grd = Control()
        if self.unreduced:
            if self.unreducedRecomputeWeightsOnly:
                grd['ct'] = np.zeros(foo['dct'].shape)
            else:
                grd['ct'] = (dt/coeff) * foo['dct']
            grd['at'] = (dt / coeff) * foo['dat']
        else:
            grd['at'] = (dt / coeff) * foo['dat']

        if self.affineDim > 0:
            grd['Afft'] = np.zeros(self.control['Afft'].shape)
            dA = foo['dA']
            db = foo['db']
            grd['Afft'] = 2*self.affineWeight.reshape([1, self.affineDim])*control['Afft']
            for t in range(self.Tsize):
                dAff = self.affineBasis.T @ \
                   np.vstack([dA[t].reshape([dim2, 1]),
                              db[t].reshape([self.dim, 1])])
                grd['Afft'][t] -= dAff.reshape(grd['Afft'][t].shape)
            grd['Afft'] *= dt / (self.coeffAff*coeff)
        return grd

    def resetPK(self, newType=None):
        if self.PKResetCount < self.maxPKReset:
            if newType is None:
                self.options['KparDiff'].pk_dtype = self.Kdiff_dtype
                self.options['KparDist'].pk_dtype = self.Kdist_dtype
            else:
                self.options['KparDiff'].pk_dtype = newType
                self.options['KparDist'].pk_dtype = newType
                if newType != self.Kdiff_dtype:
                    logging.info(f'Reset {self.PKResetCount}')

        self.pkBuffer = 0
        if self.PKResetCount == self.maxPKReset:
            logging.info('Maximum number of resets attained')
        else:
            self.PKResetCount += 1

    def startOfIteration(self):
        if self.options['algorithm'] != 'sgd':
            if self.reset:
                obj0 = self.objectiveFun()
                self.obj = None
                obj = self.objectiveFun()
                if self.gpuAvailable and self.resetPK_:
                    self.obj = None
                    if self.Kdiff_dtype != 'float64':
                        logging.info('switching to float64')
                        self.resetPK('float64')
                        obj2 = self.objectiveFun()
                else:
                    obj2 = obj
                self.resetPK_ = True
                if self.gpuAvailable and self.Kdiff_dtype != 'float64':
                    logging.info(f"recomputing Objective {obj0:.5f} {obj:.5f} {obj2:.5f}")
                self.pkBuffer = 0

    def randomDir(self, ControlBatch=None):
        dirfoo = Control()
        if self.affineOnly:
            dirfoo['at'] = np.zeros((self.Tsize, self.npt, self.dim))
        else:
            if ControlBatch is None:
                dirfoo['at'] = np.random.randn(self.Tsize, self.npt, self.dim)
            else:
                d = np.random.randn(self.Tsize, self.npt, self.dim)
                dirfoo['at'] = np.zeros(d.shape)
                dirfoo['at'][:, ControlBatch, :] = d[:, ControlBatch, :]
        if self.unreduced:
            if self.unreducedRecomputeWeightsOnly:
                dirfoo['ct'] = np.zeros((self.Tsize, self.npt, self.dim))
            else:
                if ControlBatch is None:
                    dirfoo['ct'] \
                        = np.random.normal(0, 1, size=self.control['ct'].shape)
                else:
                    d = np.random.normal(0, 1, size=self.control['ct'].shape)
                    dirfoo['ct'] = np.zeros(d.shape)
                    dirfoo['ct'][:, ControlBatch, :] = d[:, ControlBatch, :]
                dirfoo['ct'][0, :, :] = 0
        if self.affineDim > 0:
            dirfoo['Afft'] = np.random.randn(self.Tsize, self.affineDim)
        return dirfoo

    def dotProduct_Riemannian(self, g1, g2):
        res = np.zeros(len(g2))
        for t in range(self.Tsize):
            z = self.state['xt'][t, :, :]
            gg = g1['at'][t, :, :]
            u = self.options['KparDiff'].applyK(z, gg)
            if self.affineDim > 0:
                uu = g1['Afft'][t]
            else:
                uu = 0
            ll = 0
            for gr in g2:
                ggOld = gr['at'][t, :, :]
                res[ll] = res[ll] + (ggOld * u).sum()
                if self.affineDim > 0:
                    res[ll] += (uu*gr['Afft'][t]).sum() * self.coeffAff
                ll = ll + 1
        return res

    def dotProduct_euclidean(self, g1, g2):
        res = np.zeros(len(g2))
        for k in g1.keys():
            if g1[k] is not None:
                for ll, gr in enumerate(g2):
                    res[ll] += (g1[k]*gr[k]).sum()
        return res

    def saveCorrectedTarget(self, X0, X1):
        U = la.inv(X0[-1])
        f = self.createObject(self.fv1)
        yyt = np.dot(f.vertices - X1[-1, ...], U)
        f.updateVertices(yyt)
        f.saveVTK(self.outputDir + '/TargetCorrected.vtk')
        if self.match_landmarks:
            p = PointSet(data=self.targ_lmk)
            yyt = np.dot(p.vertices - X1[-1, ...], U)
            p.updateVertices(yyt)
            p.saveVTK(self.outputDir + '/TargetLandmarkCorrected.vtk')

    def saveCorrectedEvolution(self, fv0, state, control,
                               fileName='evolution'):
        Jacobian = state['Jt']
        f = self.createObject(fv0)
        if self.match_landmarks:
            p = PointSet(data=self.tmpl_lmk)
        else:
            p = None

        X = self.affB.integrateFlow(control['Afft'])
        displ = np.zeros(state['xt'].shape[1])
        dt = self.maxT / self.Tsize
        fn = []
        if type(fileName) is str:
            for kk in range(self.Tsize + 1):
                fn.append(fileName + f'_corrected{kk:03d}')
        else:
            fn = fileName
        vt = None
        for t in range(self.Tsize + 1):
            U = la.inv(X[0][t])
            yyt = (state['xt'][t, ...] - X[1][t, ...]) @ U.T
            zt = (state['xt'][t, ...] - X[1][t, ...]) @ U.T
            if t < self.Tsize:
                atCorr = control['at'][t, ...] @ U.T
                vt = self.options['KparDiff'].applyK(yyt, atCorr, firstVar=zt)
            f.updateVertices(yyt)
            if self.match_landmarks:
                p.updateVertices(yyt[self.nvert:, :])
                p.saveVTK(self.outputDir + '/' + fn[t] + '_lmk.vtk')
            vf = vtkFields('POINT_DATA', self.fv0.vertices.shape[0])
            if Jacobian is not None:
                vf.scalars['Jacobian'] = np.exp(Jacobian[t, :, 0])

            vf.scalars['displacement'] = displ
            vf.vectors['velocity'] = vt
            f.save(self.outputDir + '/' + fn[t] + '.vtk', vf)
            displ += dt * np.sqrt((vt**2).sum(axis=1))

        self.saveCorrectedTarget(X[0], X[1])

    def saveEvolution(self, fv0, state, passenger=None, fileName='evolution',
                      velocity=None, grid=None):
        xt = state['xt']
        Jacobian = state['Jt']
        if velocity is None:
            velocity = self.v

        fn = []
        if type(fileName) is str:
            for kk in range(self.Tsize + 1):
                fn.append(fileName + f'{kk:03d}')
        else:
            fn = fileName

        fvDef = self.createObject(fv0)
        nvert = fv0.vertices.shape[0]
        v = velocity[0, :nvert, :]
        displ = np.zeros(nvert)
        dt = self.maxT / self.Tsize
        for kk in range(self.Tsize + 1):
            fvDef.updateVertices(np.squeeze(xt[kk, :nvert, :]))
            vf = vtkFields('POINT_DATA', self.fv0.vertices.shape[0])
            if Jacobian is not None:
                vf.scalars['Jacobian'] = np.exp(Jacobian[kk, :nvert, 0])

            vf.scalars['displacement'] = displ
            if kk < self.Tsize:
                v = velocity[kk, :nvert, :]
                kkm = kk
            else:
                kkm = kk - 1
            if self.dim < 3:
                x = velocity[kkm, :nvert]
                vf.vectors['velocity'] \
                    = np.concatenate((x,
                                      np.zeros((x.shape[0], 3 - x.shape[1]))),
                                     axis=1)
            else:
                vf.vectors['velocity'] = velocity[kkm, :nvert]
            fvDef.save(self.outputDir + '/' + fn[kk] + '.vtk', vf)
            displ += dt * np.sqrt((v**2).sum(axis=1))
            if self.match_landmarks:
                pp = PointSet(data=xt[kk, nvert:, :])
                pp.saveVTK(self.outputDir + '/' + fn[kk] + '_lmk.vtk')

            if passenger is not None and passenger[0] is not None:
                if isinstance(passenger[0], type(self.fv0)):
                    fvp = self.createObject(passenger[0])
                    fvp.updateVertices(passenger[1][kk, ...])
                    fvp.saveVTK(self.outputDir+'/'+fn[kk]+'_passenger.vtk')
                else:
                    savePoints(self.outputDir+'/'+fn[kk]+'_passenger.vtk',
                               passenger[1][kk, ...])

            if grid is not None and self.dim == 2:
                self.gridDef.vertices = np.copy(grid['yt'][kk, :, :])
                if grid['Jyt'] is None:
                    self.gridDef.saveVTK(self.outputDir+'/grid'+str(kk)+'.vtk')
                else:
                    self.gridDef.saveVTK(self.outputDir+'/grid'+str(kk)+'.vtk',
                                         vtkFields=vtkFields('POINT_DATA',
                                                             self.gridDef.vertices.shape[0],
                                                             scalars={'logJacobian':grid['Jyt'][kk, :, 0],
                                                                    'finalLogJacobian':grid['Jyt'][-1, :, 0]}))

    def saveHdf5(self, fileName='result'):
        return None

    def acceptVarTry(self):
        self.obj = self.objTry
        self.objDef = self.objTryDef
        self.objData = self.objTryData
        self.control = deepcopy(self.controlTry)
        self.state = deepcopy(self.stateTry)
        self.updateEndPoint(self.state)

    def updateEndPoint(self, state):
        self.fvDef.updateVertices(state['xt'][-1, :, :])
        if self.match_landmarks:
            self.def_lmk.updateVertices(state['xt'][-1, self.nvert:, :])

    def endOfIterationSGD(self, forceSave=False):
        if self.unreducedResetRate > 0 and self.iter % self.unreducedResetRate == 0:
            dist = ((self.control['ct'] - self.state['xt'][-1, :, :])**2).sum(axis=-1).max()
            logging.info(f'SGD: Resetting trajectories: max distance = {dist:.4f}')
            self.control['ct'] = np.copy(self.state['xt'][:-1, :, :])
            self.unreducedRecomputeWeightsOnly = True
            self.SGDFreezePointsCount = 0
            self.controlTry['ct'] = np.copy(self.control['ct'])

        if self.unreducedRecomputeWeightsOnly and self.SGDFreezePointsCount == self.unreducedRecomputeWeightsMax:
            logging.info('SGD: Returning to full gradient')
            self.unreducedRecomputeWeightsOnly = False

        if forceSave or self.iter % self.saveRate == 0:
            saveTrajectories(self.outputDir + '/' + self.options['saveFile'] + 'curvesSGDState.vtk',
                             self.state['xt'])
            saveTrajectories(self.outputDir + '/' + self.options['saveFile'] + 'curvesSGDControl.vtk',
                             self.control['ct'])
            self.saveEvolution(self.fv0, self.state)
            obj, st = self.objectiveFunDef(self.control, withTrajectory=True)
            fvdef_ = PointSet(data=self.fv0)
            fvdef_.updateVertices(st['xt'][-1, :, :])
            obj += self.dataTerm(fvdef_) + self.fun_obj0(self.fv1) \
                / (self.options['sigmaError'] ** 2)
            self.saveEvolution(self.fv0, st, fileName='evolution_complete',
                               velocity=st['vt'])
            logging.info(f"Objective on complete problem: {obj:.4f}")

    def stopCondition(self):
        if self.unreducedRecomputeWeightsOnly and self.obj < self.obj_unreduced_save:
            return True
        return False

    def unreducedResetTrajectories(self):
        if (self.unreduced and self.unreducedResetRate > 0
            and self.iter % self.unreducedResetRate == 0
            and self.obj < self.obj_unreduced_save
            and not self.unreducedRecomputeWeightsOnly):
            self.obj_unreduced_save = self.obj
            dist0 = np.sqrt(((self.state['xt'][0, :, :] - self.state['xt'][-1, :, :])**2).sum(axis=-1).max())
            dist = np.sqrt(((self.control['ct'] - self.state['xt'][:-1, :, :])**2).sum(axis=-1).max())
            if dist > 0.001 * dist0:
                control = deepcopy((self.control))
                control['ct'] = self.state['xt'][:-1, :, :]
                logging.info(f'Resetting trajectories: max distance = {dist:.4f} trajectory distance = {dist0:.4f}')
                self.control = deepcopy(control)
                self.controlTry['ct'] = np.copy(self.control['ct'])
                self.unreducedRecomputeWeightsOnly = True
                logging.info('Optimizing weights only')
                self.reset = True
                currentIter = self.iter
                self.TargetObj = self.obj
                bfgs.bfgs(self, verb=self.options['verb'],
                          maxIter=self.unreducedRecomputeWeightsMax,
                          epsInit=1.,
                          Wolfe=self.options['Wolfe'],
                          lineSearch=self.options['lineSearch'], memory=50)
                logging.info('End of weight optimization')
                self.TargetObj = -np.inf
                self.iter = currentIter
                self.unreducedRecomputeWeightsOnly = False
                self.reset = True
                self.resetPK_ = False

    def endOfProcedure(self):
        if self.iter % self.saveRate != 0:
            self.endOfIteration(forceSave=True)

    def endOfIteration(self, forceSave=False):
        self.iter += 1
        if self.options['algorithm'] == 'sgd':
            self.endOfIterationSGD(forceSave=forceSave)
            return

        if self.options['testGradient']:
            self.testEndpointGradient()
        if self.iter >= self.affBurnIn:
            self.coeffAff = self.coeffAff2

        dim2 = self.dim ** 2
        if self.affineDim > 0:
            A = [np.zeros([self.Tsize, self.dim, self.dim]),
                 np.zeros([self.Tsize, self.dim])]
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, self.control['Afft'][t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2 + self.dim]
        else:
            A = None

        if ((forceSave or self.iter % self.saveRate == 0) and
            not self.unreducedRecomputeWeightsOnly):
            logging.info('Saving point sets...')
            if self.dim == 2:
                ytGrid = self.solveStateEquation(options={'withPointSet':self.gridxy, 'withJacobian':True})
            else:
                ytGrid = None

            st = self.solveStateEquation(options={'withJacobian':True, 'withPointSet':self.passenger_points})
            xt = st['xt']
            Jt = st['Jt']
            yt = st['yt']

            if self.passenger_points is not None:
                if isinstance(self.passenger, type(self.fv0)):
                    self.passengerDef.updateVertices(yt[-1, ...])
                else:
                    self.passengerDef = deepcopy(yt[-1, ...])

            self.trajCounter = self.varCounter
            self.state['xt'] = xt
            self.state['Jt'] = Jt
            self.state['yt'] = yt

            if self.options['saveTrajectories']:
                saveTrajectories(self.outputDir + '/'
                                 + self.options['saveFile']
                                 + 'curves.vtk', self.state['xt'])

            self.updateEndPoint(self.state)

            if (self.options['affine'] == 'euclidean'
                or self.options['affine'] == 'translation'):
                self.saveCorrectedEvolution(self.fv0, self.state, self.control,
                                            fileName=self.saveFileList)
            self.saveEvolution(self.fv0, self.state, fileName=self.saveFileList,
                               passenger=(self.passenger, yt), grid=ytGrid)
            if self.unreduced:
                saveTrajectories(self.outputDir + '/' + self.options['saveFile']
                                 + 'curvesUnreducedState.vtk',
                                 self.state['xt'])
                saveTrajectories(self.outputDir + '/' + self.options['saveFile']
                                 + 'curvesUnreducedControl.vtk',
                                 self.control['ct'])
            self.saveHdf5(fileName=self.outputDir + '/output.h5')

        if self.unreduced:
            self.unreducedResetTrajectories()

        if (self.options['KparDiff'].pk_dtype != self.Kdiff_dtype
            and self.PKResetCount < self.maxPKReset):
            if self.pkBuffer > 5:
                logging.info("return to original pk_dtype")
                self.resetPK()
                self.pkBuffer = 0
            else:
                self.pkBuffer += 1

    def optimizeMatching(self):
        if self.options['algorithm'] in ('cg', 'bfgs'):
            self.coeffAff = self.coeffAff2
            # grd = self.getGradient(self.gradCoeff)
            # [grd2] = self.dotProduct(grd, [grd])
            #
            # if self.gradEps < 0:
            #     self.gradEps = max(0.001, np.sqrt(grd2) / 10000)
            # logging.info('Gradient lower bound: %f' % (self.gradEps))
            self.coeffAff = self.coeffAff1

            if self.options['algorithm'] == 'cg':
                cg.cg(self, verb=self.options['verb'],
                      maxIter=self.options['maxIter'],
                      TestGradient=self.options['testGradient'], epsInit=0.1)
            elif self.options['algorithm'] == 'bfgs':
                bfgs.bfgs(self, verb=self.options['verb'],
                          maxIter=self.options['maxIter'],
                          TestGradient=self.options['testGradient'],
                          epsInit=1.,
                          Wolfe=self.options['Wolfe'], memory=50)
        elif self.options['algorithm'] == 'sgd':
            logging.info(f'Running stochastic gradient descent {self.sgdEpsInit: .12f}')
            sgd.sgd(self, verb=self.options['verb'],
                    maxIter=self.options['maxIter'],
                    TestGradient=self.options['testGradient'],
                    burnIn=self.sgdBurnIn, epsInit=self.sgdEpsInit,
                    normalization=self.sgdNormalization)

