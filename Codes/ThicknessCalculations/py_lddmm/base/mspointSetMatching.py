import time
from copy import deepcopy
import numpy as np
import scipy.linalg as la
import logging
from functools import partial
from . import (conjugateGradient as cg,
               kernelFunctions as kfun,
               mspointEvolution as evol,
               grid,
               bfgs)
from .pointSets import PointSet, saveTrajectories, savePoints
from . import pointSets, pointsetDistances as psd
from .affineBasis import AffineBasis
from .pointSetMatching import PointSetMatching
from numpy.random import default_rng
from .pointSets import PointSet, saveTrajectories, savePoints
import keopscore
from .affineBasis import getExponential, gradExponential
from .basicMatching import BasicMatching
from .vtk_fields import vtkFields
from numpy.random import default_rng
rng = default_rng()
import pdb
# This document assumes that the templates can have different # of points by creating lists.

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
class MultiscalePointSetMatching(BasicMatching):
    def __init__(self, Template=None, Target=None, options=None):
        # super().__init__(Template, Target, options)
        self.setInitialOptions(options)
        if self.options['algorithm'] == 'cg':
             self.euclideanGradient = False
        else:
            self.euclideanGradient = True

        self.maxT = self.options['maxT']
        self.fun_obj = None
        self.fun_obj0 = None
        self.fun_objGrad = None
        self.obj0 = 0
        self.coeffAff = 1
        self.obj = None
        self.objDef = 0
        self.objData = 0
        self.objTry = np.inf
        self.objTryDef = 0
        self.objTryData = 0
        self.control = Control()
        self.controlTry = Control()
        self.state = State()
        self.setOutputDir(self.options['outputDir'])
        self.set_landmarks(self.options['Landmarks'])
        self.base_scale = options['base_scale']
        self.scales = options['scales']

        self.nbase_scales = len(self.base_scale)
        self.nscales = len(self.scales)
        self.set_template_and_target(Template, Target, misc=self.options)
        self.burnIn = self.options['burnIn']
        self.gpuAvailable = keopscore.config.config.use_cuda

        self.reset = False
        self.Kdiff_dtype = self.options['pk_dtype']
        self.Kdist_dtype = self.options['pk_dtype']
        self.unreduced = self.options['unreduced']
        if not self.unreduced:
            self.options['unreducedWeight'] = 0
            #print np.fabs(self.fv1.surfel-self.fv0.surfel).max()
        self.set_parameters()
        self.set_fun(self.options['errorType'], vfun=self.options['vfun'])
        self.setDotProduct(self.options['unreduced'])

        self.initialize_variables()
        self.gradCoeff = 1.
        self.set_passenger(self.options['passenger'])
        self.pplot = self.options['pplot']
        if self.pplot:
            self.initial_plot()
        self.initialSave()

    def resetPK(self, newType = None):
        if self.PKResetCount < self.maxPKReset:
            l1 = len(self.options['KparDiff'])
            l2 = len(self.options['KparDiff'][0])
            for ii in range(l1):
                for jj in range(l2):
                    if newType is None:
                        self.options['KparDiff'][ii][jj].pk_dtype = self.Kdiff_dtype
                        # self.options['KparDist'][ii][jj].pk_dtype = self.Kdist_dtype
                        # self.options['KparDist'].pk_dtype = self.Kdist_dtype
                    else:
                        self.options['KparDiff'][ii][jj].pk_dtype = newType
                        # self.options['KparDist'][ii][jj].pk_dtype = newType
                        # self.options['KparDist'].pk_dtype = newType

                # self.options['KparDiff'].pk_dtype = self.Kdiff_dtype
                # self.options['KparDist'].pk_dtype = self.Kdist_dtype
        self.pkBuffer = 0
        self.PKResetCount += 1
        logging.info(f'Reset {self.PKResetCount}')
        if self.PKResetCount == self.maxPKReset:
            logging.info('Maximum number of resets attained')

    def getDefaultOptions(self):
        options = super().getDefaultOptions()
        options['reweightCells'] = False
        options['unreducedResetRate'] = -1
        options['checkOrientation'] = True
        options['scales'] = np.linspace(.1, 2., 20)
        options['base_scales'] = [0, 19]
        options['nscales'] = len(options['scales'])
        options['nbase_scales'] = len(options['base_scales'])

        return options

    def set_passenger(self, passenger):
        # Needs to modify passenger
        self.passenger = passenger

        if isinstance(self.passenger, type(self.fv0)):
            self.passenger_points = self.passenger.vertices
        elif self.passenger is not None:
            self.passenger_points = self.passenger
        else:
            self.passenger_points = None
        self.passengerDef = deepcopy(self.passenger)

    def set_fun(self, errorType, vfun=None):
        self.options['errorType'] = errorType
        self.match_landmarks = self.options['match_landmarks']
        if errorType == 'L2':
            self.fun_obj0 = psd.L2Norm0
            self.fun_obj = psd.L2NormDef
            self.fun_objGrad = psd.L2NormGradient
            self.sgdSelection = 'single'
        elif errorType == 'measure':
            self.fun_obj0 = partial(psd.measureNorm0, KparDist=self.options['KparDist'])
            self.fun_obj = partial(psd.measureNormDef, KparDist=self.options['KparDist'])
            self.fun_objGrad = partial(psd.measureNormGradient, KparDist=self.options['KparDist'])
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


    def set_template_and_target(self, Template, Target, misc=None):
        # multiscale template and target:
        # has one more dimension of size # of (base) scales in dimension 0.
        nbase_scales = self.options['nbase_scales']
        # base_scales = self.base_scales
        if Template is None:
            logging.error('Please provide a template surface')
            return
        else:
            self.fv0 = []
            for ii in range(nbase_scales):
                self.fv0.append(PointSet(data=Template[ii]))
            # self.fv0 = fv0
            # self.fv0 = PointSet(data=Template)

        if Target is None:
            logging.error('Please provide a target surface')
            return
        else:
            self.fv1 = []
            for ii in range(nbase_scales):
                self.fv1.append(PointSet(data=Target[ii]))
            # self.fv1 = fv1
            # self.fv1 = PointSet(data=Target)

        for ii in range(nbase_scales):
            self.fv0[ii].save(self.outputDir + f'/Template{ii}.vtk')
            self.fv1[ii].save(self.outputDir + f'/Target{ii}.vtk')
        self.dim = self.fv0[0].vertices.shape[1]
        # self.fv0.save(self.outputDir + '/Template.vtk')
        # self.fv1.save(self.outputDir + '/Target.vtk')
        # self.dim = self.fv0.vertices.shape[1]

    def initialize_variables(self):
        # self.base_scales = self.options['base_scales']
        # self.nbase_scales = len(self.base_scales)
        # self.scales = self.options['scales']
        # self.nscales = len(self.scales)
        self.randomInit = self.options['randomInit']
        self.match_landmarks = self.options['match_landmarks']
        x0 = []
        self.nvert = np.zeros(self.nbase_scales)
        for ii in range(self.nbase_scales):
            x0.append(self.fv0[ii].vertices)
            self.nvert[ii] = self.fv0[ii].vertices.shape[0]
        if self.match_landmarks:
            nlmk = np.zeros(self.nbase_scales)
            for ii in range(self.nbase_scales):
                self.x0 = np.concatenate(self.fv0[ii].vertices, self.tmpl_lmk[ii].vertices, axis=0)
                nlmk = self.tmpl_lmk[ii].vertices.shape[0]
            self.nlmk = nlmk
        else:
            self.x0 = x0
            self.nlmk = 0

        self.fvDef = deepcopy(self.fv0)
        if self.match_landmarks:
            def_lmk = []
            for ii in range(self.nbase_scales):
                def_lmk.append(PointSet(data=self.tmpl_lmk[ii]))
            self.def_lmk = def_lmk
        else:
            self.def_lmk = None
        self.npt = np.zeros(self.nbase_scales).astype('int')
        for ii in range(self.nbase_scales):
            self.npt[ii] = self.x0[ii].shape[0]
        # self.npt = self.x0[0].shape[1]

        self.control = Control()
        self.controlTry = Control()

        self.Tsize = int(round(self.maxT/self.options['timeStep']))
        self.control['at'] = []
        self.controlTry['at'] = []
        for ii in range(self.nbase_scales):
            self.control['at'].append(np.zeros([self.Tsize, self.x0[ii].shape[0], self.x0[ii].shape[1]]))
            if self.randomInit:
                self.control['at'].append(np.random.normal(0, .01, self.control['at'][ii].shape))
            self.controlTry['at'].append(np.zeros([self.Tsize, self.x0[ii].shape[0], self.x0[ii].shape[1]]))

        # self.control['at'] = np.zeros([self.nbase_scale, self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        # if self.randomInit:
        #     self.control['at'] = np.random.normal(0, .01, self.control['at'].shape)
        # self.controlTry['at'] = np.zeros([self.nbase_scale, self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        self.state = State()
        self.stateTry = State()
        self.state['xt'] = []
        self.stateTry['xt'] = []
        self.v = []
        for ii in range(self.nbase_scales):
            self.state['xt'].append(np.tile(self.x0[ii], [self.Tsize+1, 1, 1]))
            self.stateTry['xt'].append(np.tile(self.x0[ii], [self.Tsize+1, 1, 1]))
            self.v.append(np.zeros([self.Tsize + 1, self.npt[ii], self.dim]))
        # self.state['xt'] = np.tile(self.x0, [1, self.Tsize+1, 1, 1])
        # self.stateTry['xt'] = np.copy(self.state['xt'])
        # self.v = np.zeros([self.nbase_scale, self.Tsize+1, self.npt, self.dim])
        # try npt has different dimensions

        self.passenger_points = None
        self.saveFileList = []
        for ii in range(self.nbase_scales):
            iisaveFileList = []
            for kk in range(self.Tsize+1):
                iisaveFileList.append(self.options['saveFile']+f'{ii}_{kk:03d}')
                # self.saveFileList.append(self.options['saveFile'] + f'{kk:03d}')
            self.saveFileList.append(iisaveFileList)

        if self.dim == 2:
            xmin_total = np.zeros(self.nbase_scales)
            xmax_total = np.zeros(self.nbase_scales)
            ymin_total = np.zeros(self.nbase_scales)
            ymax_total = np.zeros(self.nbase_scales)
            for ii in range(self.nbase_scales):
                xmin_total[ii] = min(self.fv0[ii].vertices[:,0].min(), self.fv1[ii].vertices[:,0].min())
                xmax_total[ii] = max(self.fv0[ii].vertices[:,0].max(), self.fv1[ii].vertices[:,0].max())
                ymin_total[ii] = min(self.fv0[ii].vertices[:,1].min(), self.fv1[ii].vertices[:,1].min())
                ymax_total[ii] = max(self.fv0[ii].vertices[:,1].max(), self.fv1[ii].vertices[:,1].max())
            xmin = min(xmin_total)
            xmax = max(xmax_total)
            ymin = min(ymin_total)
            ymax = max(ymax_total)
            dx = 0.01*(xmax-xmin)
            dy = 0.01*(ymax-ymin)
            dxy = min(dx,dy)
            [x,y] = np.mgrid[(xmin-10*dxy):(xmax+10*dxy):dxy, (ymin-10*dxy):(ymax+10*dxy):dxy]
            self.gridDef = []
            self.gridxy = []
            for ii in range(self.nscales):
                self.gridDef.append(grid.Grid(gridPoints=[x, y]))
                self.gridxy.append(self.gridDef[ii].vertices)
            # self.gridDef = grid.Grid(gridPoints=[x, y])
            # self.gridxy = np.copy(self.gridDef.vertices)

    def set_extra_term(self):
        self.extraTerm = None

    def set_parameters(self):
        # not changed yet
        super().set_parameters()
        sigmaKernel = 6.5
        orderKernel = 3
        sigmaDist = 2.5
        orderKDist = 3
        self.reset = True
        self.resetPK_ = True
        self.PKResetCount = 0
        self.TargetObj = -np.inf
        self.maxPKReset = 10
        typeKDist = 'gauss'
        typeKernel = 'gauss'

        # Maybe need to change this
        if type(self.options['KparDiff']) in (list, tuple):
            pass





        #     # modified the numbers: all + 2
        #     typeKernel = self.options['KparDiff'][2]
        #     sigmaKernel = self.options['KparDiff'][3]
        #     if typeKernel == 'laplacian' and len(self.options['KparDiff']) > 4:
        #         orderKernel = self.options['KparDiff'][4]
        #     self.options['KparDiff'] = None
        #
        # # dont know how to modify this
        # if self.options['KparDiff'] is None:
        #     self.options['KparDiff'] = kfun.Kernel(name = typeKernel, sigma = sigmaKernel, order=orderKernel)
        #
        # if type(self.options['KparDist']) in (list,tuple):
        #     typeKDist = self.options['KparDist'][0]
        #     sigmaDist = self.options['KparDist'][1]
        #     if typeKDist == 'laplacian' and len(self.options['KparDist']) > 2:
        #         orderKdist = self.options['KparDist'][2]
        #     self.options['KparDist'] = None
        #
        # if self.options['KparDist'] is None:
        #     self.options['KparDist'] = kfun.Kernel(name = typeKDist, sigma = sigmaDist, order= orderKDist)
        for ii in range(self.options['nbase_scales']):
            for jj in range(self.options['nbase_scales']):
                self.options['KparDiff'][ii][jj].pk_dtype = self.options['pk_dtype']
                # self.options['KparDist'][ii][jj].pk_dtype = self.options['pk_dtype']

        # self.options['KparDist'].pk_dtype = self.options['pk_dtype']

        self.gradEps = self.options['gradTol']
        self.epsInit = self.options['epsInit']
        self.affineOnly = self.options['affineOnly']

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
        self.unreducedRecomputeWeightsMax = max(self.unreducedResetRate//2, 100)
        if self.options['unreduced']:
            logging.info(f'unreduced reset: {self.unreducedResetRate}, {self.unreducedRecomputeWeightsMax}')
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
                self.options['KparDiff'].setAffine(self.options['affine'], w1=1/w1, w2=1/w2,
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

    def getInitState(self):
        x0 = []
        for ii in range(self.nbase_scales):
            if self.match_landmarks:
                x0.append(np.concatenate((self.fv0[ii].vertices), self.tmpl_lmk[ii].vertices), axis=0)
            else:
                x0.append(self.fv0[ii].vertices)
        return x0

    def createObject(self, data, other=None):
        if other is None:
            return PointSet(data=data)
        else:
            return PointSet(data=data, weights=other)

    def updateObject(self, object, data, other=None):
        return object.updateVertices(data)

    def solveStateEquation(self, control=None, init_state=None, kernel=None, options=None):
        if control is None:
            control = self.control
        if init_state is None:
            init_state = self.getInitState()
        if kernel is None:
            kernel = self.options['KparDiff']

        A = self.affB.getTransforms(control['Afft'])

        return evol.mslandmarkDirectEvolutionEuler(init_state, control['at'], kernel,
                                                 affine=A, options=options, T=self.maxT)

    def dataTerm(self, _fvDef, var = None):
        if type(var) is dict and 'fv1' in var:
            fv1 = var['fv1']
        else:
            fv1 = self.fv1
        obj = 0
        for ii in range(self.nbase_scales):
            obj += self.fun_obj(_fvDef[ii], fv1[ii]) / (self.options['sigmaError']**2)

        if self.match_landmarks:
            if var is None or not 'lmk_def' in var:
                logging.error('Data term: Missing deformed landmarks')
            if var is not None and 'lmk1' in var:
                lmk1 = var['lmk1']
            else:
                lmk1 = self.targ_lmk
            for ii in range(self.nbase_scales):
                obj += self.wlmk * self.lmk_obj(var['lmk_def'][ii], lmk1[ii]) / (self.options['sigmaError']**2)

        return obj

    def objectiveFunDef(self, control, var = None, withTrajectory = False, withJacobian=False):
        if var is None or 'x0' not in var:
            x0 = self.x0
        else:
            x0 = var['x0']
        # if self.match_landmarks:
        #     x0 = np.concatenate((x0, self.tmpl_lmk.vertices), axis=0)

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

        st = self.solveStateEquation(control=control, init_state=x0, kernel=kernel,
                                     options={'withJacobian':withJacobian})
        obj = 0
        obj1 = 0
        obj2 = 0
        obj3 = 0
        # nbase_scale = self.nbase_scale

        for t in range(self.Tsize):
            for ii in range(self.nbase_scales):
                z = st['xt'][ii][t]
                aii = at[ii][t]

                if self.unreduced:
                    c = ct[ii, t]
                else:
                    c = None
                if self.unreduced:
                    pass
                else:
                    ra = 0
                    for jj in range(self.nbase_scales):
                        ra += kernel[ii][jj].applyK(st['xt'][jj][t], at[jj][t], firstVar=z)
                    obj += regWeight*timeStep*(at[ii][t]*ra).sum()
                if hasattr(self, 'v'):
                    self.v[ii][t] = ra
                # if self.extraTerm is not None:
                #     obj1 += self.extraTerm['coeff'] * self.extraTerm['fun'](z, ra) * timeStep

        obj += obj1 + obj2 + obj3
        if withTrajectory or withJacobian:
            return obj, st
        else:
            return obj

    def makeTryInstance(self, state):
        # xt = state['xt'][:, -1, :, :]
        ff = []
        for ii in range(self.nbase_scales):
            ff.append(self.createObject(state['xt'][ii][-1], other=self.fv0[ii].vertex_weights))
        # ff = self.createObject(xx, other=self.fv0.vertex_weights)
        # ff = self.createObject(state['xt'][-1,:,:], other=self.fv0.vertex_weights)

        return ff

    def objectiveFun(self):
        if self.obj == None:
            self.obj0 = 0
            self.objData = 0
            self.objDef, self.state = self.objectiveFunDef(self.control, withTrajectory=True)
            for ii in range(self.nbase_scales):
                self.obj0 += self.fun_obj0(self.fv1[ii]) / (self.options['sigmaError']**2)
                if self.match_landmarks:
                    self.obj0 += self.wlmk * self.lmk_obj0(self.targ_lmk[ii]) / (self.options['sigmaError']**2)
                self.fvDef[ii].updateVertices(self.state['xt'][ii][-1])
                if self.match_landmarks:
                    self.def_lmk[ii].vertices = self.state['xt'][ii][-1, self.nvert[ii]:, :]
            self.objData += self.dataTerm(self.fvDef)
            self.obj = self.obj0 + self.objData + self.objDef
        return self.obj
    def updateTry(self, dr, eps, objRef=None):
        controlTry = self.initVariable()
        for k in dr.keys():
            if dr[k] is not None:
                output_key = []
                for ii in range(len(dr[k])):
                    output_key.append(self.control[k][ii] - eps*dr[k][ii])
                # controlTry[k] = self.control[k] - eps * dr[k]
                controlTry[k] = output_key
            else:
                controlTry[k] = self.control[k]

        objTryDef, st = self.objectiveFunDef(controlTry, withTrajectory=True)
        ff = self.makeTryInstance(st)
        if self.match_landmarks:
            pp = []
            for ii in range(self.nbase_scales):
                pp.append(PointSet(data=self.def_lmk[ii]))
                pp[ii].updateVertices(st['xt'][ii][-1, self.nvert[ii]:, :])
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
        #print(f'Test: {np.abs(self.fvDef.vertices-self.fv1.vertices).max():.6f}')
        dff = []
        dpp = []
        dall = []
        for ii in range(self.nbase_scales):
            dff.append(np.random.normal(size=ff[ii].vertices.shape))
        # dff = np.random.normal(size=ff.vertices.shape)
            if self.match_landmarks:
                dpp.append(np.random.normal(size=self.def_lmk[ii].vertices.shape))
                dall.append(np.concatenate((dff[ii], dpp[ii]), axis=0))
            else:
                dall = dff
                dpp = None
        eps0 = 1e-6
        c = []
        for eps in [-eps0, eps0]:
            f1 = []
            pp = []
            for ii in range(self.nbase_scales):
                f1.append(self.createObject(self.fvDef[ii]))
                f1[ii].updateVertices(ff[ii].vertices + eps*dff[ii])
                if self.match_landmarks:
                    pp.append(PointSet(data=self.def_lmk[ii]))
                    pp[ii].updateVertices(pp[ii].vertices + eps * dpp[ii])
                else:
                    pp = None
            c.append(self.dataTerm(f1, {'lmk_def': pp}))
        # for ii in range(self.nbase_scales):
        #     ff[ii].vertices += eps*dff[ii]
        grd = self.endPointGradient()
        sum_output = 0
        for ii in range(self.nbase_scales):
            sum_output += (grd[ii]*dall[ii]).sum()
        logging.info(f"test endpoint gradient: {(c[1]-c[0])/(2*eps):.5f} {sum_output:.5f}")
        logging.info(f'{c[0]+self.obj0:.5f}')

    def endPointGradient(self, endPoint= None):
        if endPoint is None:
            endPoint = self.fvDef
            endPoint_lmk = self.def_lmk
        elif self.match_landmarks:
            endPoint_lmk = endPoint[1]
            # endPoint_lmk = endPoint_lmk[None,:,:]
            endPoint = endPoint[0]
        else:
            endPoint_lmk = None
        px = []
        for ii in range(self.nbase_scales):
            px.append(self.fun_objGrad(endPoint[ii], self.fv1[ii]))
            if self.match_landmarks:
                pxl = self.wlmk * self.lmk_objGrad(endPoint_lmk[ii], self.targ_lmk[ii])
                px[ii] = np.concatenate((px[ii], pxl), axis=0)
            px[ii] /= self.options['sigmaError']**2
        return px
    def hamiltonianGradient(self, px1, kernel=None, regWeight=None, x0=None, control=None):
        if regWeight is None:
            regWeight = self.options['regWeight']
        if x0 is None:
            x0 = self.x0
        if self.match_landmarks:
            for ii in range(self.nbase_scales):
                x0[ii] = np.concatenate((x0, self.tmpl_lmk[ii].vertices), axis=0)
        # if self.match_landmarks:
        #     x0 = np.concatenate((x0, self.tmpl_lmk.vertices), axis=0)

        if control is None:
            control = self.control
        affine = self.affB.getTransforms(control['Afft'])
        if kernel is None:
            kernel = self.options['KparDiff']

        return evol.mslandmarkHamiltonianGradient(x0, control['at'], px1, kernel, regWeight,
                                                  affine=affine,
                                                  getCovector=True, extraTerm=self.extraTerm,
                                                  euclidean=self.euclideanGradient, T=self.maxT)

    def initVariable(self):
        return Control()

    # effect of update is unknown
    def setUpdate(self, update):
        control = Control()
        for k in update[0].keys():
            if update[0][k] is not None:
                ss = len(update[0][k])
                output_ = []
                for ii in range(ss):
                    output_.append(self.control[k][ii] - update[1]*update[0][k][ii])
                control[k] = output_
                # control[k] = self.control[k] - update[1] * update[0][k]
        st = self.solveStateEquation(control=control, init_state=self.x0)
        # st = evol.landmarkDirectEvolutionEuler(self.x0, control['at'], self.options['KparDiff'], affine=A)
        xt = st['xt']
        if self.match_landmarks:
            endPoint0 = []
            endPoint1 = []
            for ii in range(self.nbase_scales):
                endPoint0.append(self.createObject(self.fv0[ii]))
                endPoint0[ii].updateVertices(xt[ii][-1, :self.nvert[ii], :])
                endPoint1.append(PointSet(data=xt[ii][-1, self.nvert[ii]:, :]))

            # endPoint0 = self.createObject(self.fv0)
            # for ss in range(self.nbase_scale):
            #     endPoint0.updateVertices(xt[ss, -1, :self.nvert, :])
            #     endPoint1 = PointSet(data=xt[ss, -1, self.nvert:, :])
            endPoint = (endPoint0, endPoint1)
        else:
            endPoint = []
            for ii in range(self.nbase_scales):
                endPoint.append(self.createObject(self.fv0[ii]))
                endPoint[ii].updateVertices(xt[ii][-1, :, :])
        st = State()
        st['xt'] = xt

        return control, st, endPoint


    def getGradient(self, coeff=1.0, update=None, forceComplete=False):
        # if self.options['algorithm'] == 'sgd' and not forceComplete:
        #     return self.getGradientSGD(coeff=coeff, updateState=True)
        if update is None:
            control = self.control
            if self.match_landmarks:
                endPoint = (self.fvDef, self.def_lmk)
            else:
                endPoint = self.fvDef
        else:
            control, state, endPoint = self.setUpdate(update)
            # print('set update ', control['at'][0].max(), state['xt'][0].max(), endPoint[0].vertices.max())

        dim2 = self.dim**2
        dt = self.maxT / self.Tsize
        px1 = self.endPointGradient(endPoint=endPoint)
        for ii in range(self.nbase_scales):
            px1[ii] = -px1[ii]
        foo = self.hamiltonianGradient(px1, control=control)
        #foo2 = self.hamiltonianGradient_(px1, control=control)
        grd = Control()
        if self.unreduced:
            pass
            # if self.unreducedRecomputeWeightsOnly:
            #     grd['ct'] = np.zeros(foo['dct'].shape)
            # else:
            #     grd['ct'] = (dt/coeff) * foo['dct']
            # grd['at'] = (dt / coeff) * foo['dat']
        else:
            grd_tmp = []
            for ii in range(self.nbase_scales):
                grd_tmp.append((dt / coeff) * foo['dat'][ii])
            grd['at'] = grd_tmp

        # if self.euclideanGradient and not self.unreduced:
        #     for t in range(self.Tsize):
        #         z = state['xt'][t, :, :]
        #         grd['at'][t,:,:] = self.options['KparDiff'].applyK(z, grd['at'][t, :,:])

        # if self.affineDim > 0:
        #     grd['Afft'] = np.zeros(self.control['Afft'].shape)
        #     dA = foo['dA']
        #     db = foo['db']
        #     grd['Afft'] = 2*self.affineWeight.reshape([1, self.affineDim])*control['Afft']
        #     for t in range(self.Tsize):
        #        dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
        #        grd['Afft'][t] -=  dAff.reshape(grd['Afft'][t].shape)
        #     grd['Afft'] *= dt / (self.coeffAff*coeff)
        return grd


    def startOfIteration(self):
        if self.options['algorithm'] != 'sgd':
            if self.reset:
                obj0 = self.objectiveFun()
                self.obj = None
                obj = self.objectiveFun()
                if self.gpuAvailable and self.resetPK_:
                    self.obj = None
                    logging.info('switching to float64')
                    self.resetPK('float64')
                    obj2 = self.objectiveFun()
                else:
                    obj2 = obj
                self.resetPK_ = True
                if self.gpuAvailable:
                    logging.info(f"recomputing Objective {obj0:.5f} {obj:.5f} {obj2:.5f}")
                self.pkBuffer = 0

    def dotProduct_Riemannian(self, g1, g2):
        # g1, g2 are of size dAt
        res = np.zeros(len(g2))
        for t in range(self.Tsize):
            for ii in range(self.nbase_scales):
                for jj in range(self.nbase_scales):
                    zi = self.state['xt'][ii][t]
                    zj = self.state['xt'][jj][t]
                    gg = g1['at'][jj][t]
                    u = self.options['KparDiff'][ii][jj].applyK(zj, gg, firstVar=zi)
                    # u = self.options['KparDiff'][ii][jj].applyK(z, gg)
                    if self.affineDim > 0:
                        uu = g1['Afft'][t]
                    else:
                        uu = 0
                    ll = 0
                    for gr in g2:
                        ggOld = gr['at'][jj][t]
                        res[ll] = res[ll] + (ggOld * u).sum()
                        if self.affineDim > 0:
                            res[ll] += (uu * gr['Afft'][t]).sum() * self.coeffAff
                        ll = ll + 1
        return res

    def dotProduct_euclidean(self, g1, g2):
        res = np.zeros(len(g2))
        for k in g1.keys():
            if g1[k] is not None:
                for ll, gr in enumerate(g2):
                    rtn = 0
                    for ii in range(self.nbase_scales):
                        rtn += (g1[k][ii]*gr[k][ii]).sum()
                    res[ll] += rtn

        return res

    def addProd(self, dir1, dir2, beta):
        dr = dict()
        for k in dir1.keys():
            if dir1[k] is not None and dir2[k] is not None:
                dr_k = []
                for ii in range(self.nbase_scales):
                    dr_k.append(dir1[k][ii] + beta*dir2[k][ii])
                dr[k] = dr_k
            else:
                dr[k] = None
        return dr
    def prod(self, dir1, beta):
        dr = dict()
        for k in dir1.keys():
            if dir1[k] is not None:
                dr_k = []
                for ii in range(self.nbase_scales):
                    dr_k.append(beta*dir1[k][ii])
                dr[k] = dr_k
            else:
                dr[k] = None
        return dr

    def randomDir(self, ControlBatch = None):
        dirfoo = Control()
        if self.affineOnly:
            dirfoo['at'] = np.zeros((self.Tsize, self.npt, self.dim))
        else:
            if ControlBatch is None:
                dirfoo['at'] = []
                for ii in range(self.nbase_scales):
                    dirfoo['at'].append(np.random.randn(self.Tsize, self.npt[ii], self.dim))
            else:
                d = np.random.randn(self.Tsize, self.npt, self.dim)
                dirfoo['at'] = np.zeros(d.shape)
                dirfoo['at'][:, ControlBatch, :] = d[:, ControlBatch, :]
        if self.affineDim > 0:
            dirfoo['Afft'] = np.random.randn(self.Tsize, self.affineDim)
        return dirfoo

    def saveEvolution(self, fv0, state, passenger=None, fileName='evolution', velocity=None, grid=None):
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
        fvDef = []
        nvert = np.zeros(self.nbase_scales).astype('int')
        v = []
        for ii in range(self.nbase_scales):
            fvDef.append(self.createObject(fv0[ii]))
            nvert[ii] = fv0[ii].vertices.shape[0]
            v.append(velocity[ii][0, :nvert[ii], :])
        displ = []
        for ii in range(self.nbase_scales):
            displ.append(np.zeros(nvert[ii]))
        dt = self.maxT / self.Tsize
        for kk in range(self.Tsize + 1):
            for ii in range(self.nbase_scales):
                fvDef[ii].updateVertices(np.squeeze(xt[ii][kk, :nvert[ii], :]))
                vf = vtkFields('POINT_DATA', self.fv0[ii].vertices.shape[0])
                if Jacobian is not None:
                    vf.scalars['Jacobian'] = np.exp(Jacobian[ii][kk, :nvert[ii], 0])
                vf.scalars['displacement'] = displ[ii]
                if kk < self.Tsize:
                    v[ii] = velocity[ii][kk, :nvert[ii], :]
                    kkm = kk
                else:
                    kkm = kk - 1
                if self.dim < 3:
                    x = velocity[ii][kkm, :nvert[ii]]
                    vf.vectors['velocity'] = np.concatenate((x, np.zeros((x.shape[0], 3 - x.shape[1]))), axis=1)
                else:
                    vf.vectors['velocity'] = velocity[ii][kkm, :nvert[ii]]
                fvDef[ii].save(self.outputDir + '/' + f'base{ii}' + fn[ii][kk] + '.vtk', vf)
                displ[ii] += dt * np.sqrt((v[ii] ** 2).sum(axis=1))
                if self.match_landmarks:
                    pp = PointSet(data=xt[ii][kk, nvert[ii]:, :])
                    pp.saveVTK(self.outputDir + '/' + f'base{ii}' + fn[ii][kk] + '_lmk.vtk')

                if passenger is not None and passenger[0] is not None:
                    if isinstance(passenger[0], type(self.fv0)):
                        fvp = self.createObject(passenger[0])
                        fvp.updateVertices(passenger[1][kk, ...])
                        fvp.saveVTK(self.outputDir + '/' + fn[ii][kk] + '_passenger.vtk')
                    else:
                        savePoints(self.outputDir + '/' + fn[ii][kk] + '_passenger.vtk', passenger[1][kk, ...])

                if grid is not None and self.dim == 2:
                    self.gridDef[ii].vertices = np.copy(grid['yt'][ii][kk]) # diffeo at scale ii
                    if grid['Jyt'] is None:
                        self.gridDef[ii].saveVTK(self.outputDir + '/grid' + f'base{ii}_time' + str(kk) + '.vtk')
                    else:
                        self.gridDef[ii].saveVTK(self.outputDir + '/grid' + f'base{ii}_time' + str(kk) + '.vtk',
                                             vtkFields=vtkFields('POINT_DATA', self.gridDef[ii].vertices.shape[0],
                                                                 scalars={'logJacobian': grid['Jyt'][ii][kk, :, 0],
                                                                          'finalLogJacobian': grid['Jyt'][ii][-1, :, 0]}))
                        # saveVTK(.., vtkfield(original grid, scalars={logJacobian, finalJacobian}))


    def saveHdf5(self, fileName='result'):
        return None
    def acceptVarTry(self):
        self.obj = self.objTry
        self.objDef = self.objTryDef
        self.objData = self.objTryData
        self.control = deepcopy(self.controlTry)
        self.state = deepcopy(self.stateTry)
        self.updateEndPoint(self.state)
        # print('acceptVarTry ', self.control['at'][0].max(), self.state['xt'][0].max(), self.fvDef[0].vertices.max())


    def updateEndPoint(self, state):
        for ii in range(self.nbase_scales):
            self.fvDef[ii].updateVertices(state['xt'][ii][-1])
            if self.match_landmarks:
                self.def_lmk[ii].updateVertices(state['xt'][ii][-1, self.nvert[ii]:, :])
        # self.fvDef.updateVertices(state['xt'][:, -1, :, :])
        # if self.match_landmarks:
        #     self.def_lmk.updateVertices(state['xt'][:, -1, self.nvert:, :])

    def stopCondition(self):
        if self.unreducedRecomputeWeightsOnly and self.obj < self.obj_unreduced_save:
            return True
        return False
    def endOfProcedure(self):
        if self.iter % self.saveRate != 0:
            self.endOfIteration(forceSave=True)

    def endOfIteration(self, forceSave=False):
        # t0 = time.process_time()
        self.iter += 1
        # if self.options['algorithm'] == 'sgd':
        #     self.endOfIterationSGD(forceSave=forceSave)
        #     return

        if self.options['testGradient']:
            self.testEndpointGradient()
        if self.iter >= self.affBurnIn:
            self.coeffAff = self.coeffAff2

        dim2 = self.dim ** 2
        if self.affineDim > 0:
            # not used
            A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, self.control['Afft'][t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2 + self.dim]
        else:
            A = None
        if (forceSave or self.iter % self.saveRate == 0) and not self.unreducedRecomputeWeightsOnly:
            logging.info('Saving point sets...')
            if self.dim==2:
                ytGrid = self.solveStateEquation(options={'withPointSet':self.gridxy, 'withJacobian':True})
            else:
                ytGrid = None

            st = self.solveStateEquation(options={'withJacobian':True, 'withPointSet':self.passenger_points})
            xt = st['xt']
            Jt = st['Jt']
            yt = st['yt']

            if self.passenger_points is not None:
                if isinstance(self.passenger, type(self.fv0)):
                    for ii in range(self.nbase_scales):
                        self.passengerDef[ii].updateVertices(yt[ii][-1,...])
                else:
                    for ii in range(self.nbase_scales):
                        self.passengerDef[ii] = deepcopy(yt[ii][-1,...])

            self.trajCounter = self.varCounter
            self.state['xt'] = xt
            self.state['Jt'] = Jt
            self.state['yt'] = yt

            if self.options['saveTrajectories']:
                for ii in range(self.nbase_scales):
                    saveTrajectories(self.outputDir + '/' + self.options['saveFile'] + f'base{ii}' + 'curves.vtk', self.state['xt'][ii])


            self.updateEndPoint(self.state)

            if self.options['affine'] == 'euclidean' or self.options['affine'] == 'translation':
                self.saveCorrectedEvolution(self.fv0, self.state, self.control, fileName=self.saveFileList)
            self.saveEvolution(self.fv0, self.state, fileName=self.saveFileList, passenger=(self.passenger, yt), grid=ytGrid)
            # if self.unreduced:
            #     saveTrajectories(self.outputDir + '/' + self.options['saveFile'] + 'curvesUnreducedState.vtk',
            #                                self.state['xt'])
            #     saveTrajectories(self.outputDir + '/' + self.options['saveFile'] + 'curvesUnreducedControl.vtk',
            #                                self.control['ct'])
            self.saveHdf5(fileName=self.outputDir + '/output.h5')
        #
        # if self.unreduced:
        #     self.unreducedResetTrajectories()

        if self.options['KparDiff'][0][0].pk_dtype != self.Kdiff_dtype and self.PKResetCount < self.maxPKReset:
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
                cg.cg(self, verb = self.options['verb'], maxIter = self.options['maxIter'],
                      TestGradient=self.options['testGradient'], epsInit=0.1)
            elif self.options['algorithm'] == 'bfgs':
                bfgs.bfgs(self, verb = self.options['verb'], maxIter = self.options['maxIter'],
                          TestGradient=self.options['testGradient'], epsInit=1.,
                          Wolfe=self.options['Wolfe'], memory=self.options['bfgs_memory'])
        # elif self.options['algorithm'] == 'sgd':
        #     logging.info(f'Running stochastic gradient descent {self.sgdEpsInit: .12f}')
        #     sgd.sgd(self, verb=self.options['verb'], maxIter=self.options['maxIter'],
        #             #TestGradient=False,
        #             TestGradient=self.options['testGradient'],
        #             burnIn=self.sgdBurnIn, epsInit=self.sgdEpsInit, normalization=self.sgdNormalization)
        #




