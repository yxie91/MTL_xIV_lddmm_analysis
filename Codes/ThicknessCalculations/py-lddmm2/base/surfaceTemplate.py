import numpy as np
from copy import deepcopy
import scipy.linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
import h5py
from . import surfaces
from .surfaceMatching import SurfaceMatching, Control as SMControl, State as SMState
from . import surfaceDistances as sd
import multiprocessing as mp
from . import conjugateGradient as cg, bfgs as bfgs, kernelFunctions as kfun
from .surfaceExamples import Ellipse_pygal, FlatEllipse


class Control(dict):
    def __init__(self):
        super().__init__()
        self['x0'] = None
        self['c0'] = SMControl()
        self['cAll'] = []


class State(dict):
    def __int__(self):
        super().__init__()
        self['st0'] = SMState()
        self['stAll'] = []


# Main class for surface template estimation
#        HyperTmpl: surface class (from surface.py); if not specified, opens fileHTempl
#        Targets: list of surface class (from surface.py); if not specified, open them from fileTarg
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
class SurfaceTemplate(SurfaceMatching):

    def __init__(self, Template=None, Target=None, options=None):
        super().__init__(Template, Target, options)

    def set_template_and_target(self, Template, Target, misc=None):
        if Target is None:
            logging.error('Please provide Target surfaces')
            return
        else:
            self.fv1 = []
            for ff in Target:
                self.fv1.append(surfaces.Surface(surf=ff))

        self.Ntarg = len(self.fv1)
        if Template is None:
            logging.info('Computing average hypertemplate')
            self.fv0 = self.createHypertemplate(self.fv1)
        else:
            self.fv0 = surfaces.Surface(surf=Template)

        self.fix_orientation()
        self.fv0.saveVTK(self.outputDir + '/' + 'HyperTemplate.vtk')
        for kk in range(self.Ntarg):
            self.fv1[kk].saveVTK(self.outputDir + '/' + 'Target'+str(kk)+'.vtk')
        self.dim = self.fv0.vertices.shape[1]

    def initialize_variables(self):
        self.Ntarg = len(self.fv1)
        self.npt = self.fv0.vertices.shape[0]
        self.dim = self.fv0.vertices.shape[1]
        self.iter = 0
        self.fvTmpl = surfaces.Surface(surf=self.fv0)
        self.fvTmplTry = surfaces.Surface(surf=self.fv0)
        self.Tsize = int(round(1.0 / self.options['timeStep']))
        self.control = Control()
        self.controlTry = Control()
        self.state = State()

        self.control['x0'] = self.fv0.vertices
        # self.control['c0'] = SMControl()
        # self.controlTry['c0'] = SMControl()
        self.control['c0']['at'] = np.zeros([self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1]])
        self.controlTry['c0']['at'] = np.zeros([self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1]])
        # if self.affineDim > 0:
        #     self.control['c0']['Afft'] = np.zeros([self.Tsize, self.affineDim])
        self.state['st0'] = SMState()
        self.state['st0']['xt'] = np.tile(self.fv0.vertices, [self.Tsize+1, 1, 1])

        #Template to all
        self.control['cAll'] = []
        self.controlTry['cAll'] = []
        self.state['stAll'] = []
        self.fvDef = []
        self.fvDefTry = []
        for ff in self.fv1:
            sm = SMControl()
            sm['at'] = np.zeros([self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1]])
            if self.affineDim > 0:
                sm['Afft'] = np.zeros([self.Tsize, self.affineDim])
            self.control['cAll'].append(sm)
            sm = SMControl()
            sm['at'] = np.zeros([self.Tsize, self.fv0.vertices.shape[0], self.fv0.vertices.shape[1]])
            if self.affineDim > 0:
                sm['Afft'] = np.zeros([self.Tsize, self.affineDim])
            self.controlTry['cAll'].append(sm)
            sms = SMState()
            sms['xt'] = np.tile(self.fv0.vertices, [self.Tsize+1, 1, 1])
            self.state['stAll'].append(sms)
            self.fvDef.append(surfaces.Surface(surf=self.fv0))
            self.fvDefTry.append(surfaces.Surface(surf=self.fv0))

    def fix_orientation(self, fv1=None):
        if self.fv0.surfArea() > 0:
            self.fv0.flipFaces()
        for ff in self.fv1:
            if ff.surfArea() > 0:
                ff.flipFaces()

    def getDefaultOptions(self):
        options = super().getDefaultOptions()
        options['lambdaPrior'] = 1.
        options['updateTemplate'] = True
        options['updateAllTraj'] = True
        options['templateBurnIn'] = 10
        options['sgd'] = None
        options['KparDiff0'] = None

        return options

    def set_parameters(self):
        super().set_parameters()
        if self.affineDim > 0:
            self.options['updateAffine'] = True
        else:
            self.options['updateAffine'] = False

        self.tmplCoeff = 1. # self.options['lambdaPrior']/self.Ntarg
        self.gradCoeff = 1. # *self.Ntarg#self.fv0.vertices.shape[0]

        if self.options['sgd'] is None:
            self.options['sgd'] = self.Ntarg
        else:
            self.options['sgd'] = self.options['sgd']

        self.select = np.ones(self.Ntarg, dtype=bool)
        if type(self.options['KparDiff0']) in (list,tuple):
            typeKernel = self.options['KparDiff0'][0]
            sigmaKernel = self.options['KparDiff0'][1]
            if typeKernel == 'laplacian' and len(self.options['KparDiff']) > 2:
                orderKernel = self.options['KparDiff0'][2]
            else:
                orderKernel = 4
            self.options['KparDiff0'] = kfun.Kernel(name=typeKernel,
                                                    sigma=sigmaKernel,
                                                    order=orderKernel)
        elif self.options['KparDiff0'] is None:
            self.options['KparDiff0'] = self.options['KparDiff']

    def init(self, ff):
        self.fv0 = ff.fvTmpl
        self.fv1 = ff.fv1

        self.fvTmpl = surfaces.Surface(surf=self.fv0)
        self.options = ff.options

        self.Tsize = int(round(1.0/self.options['timeStep']))

        self.control = deepcopy(ff.control)
        self.state = deepcopy(ff.state)

        self.Ntarg = len(self.fv1)
        self.tmplCoeff = 1.0 #float(self.Ntarg)
        self.obj = None #ff.obj
        self.obj0 = ff.obj0
        self.objTry = ff.objTry

        self.fvDef = deepcopy(ff.fvDef)

    def createHypertemplate(self, fv, targetSize=1000):
        dim = fv[0].vertices.shape[1]
        m = np.zeros(dim)
        I = np.zeros((dim, dim))
        N = 0
        v = 0
        sa = 0
        for f in fv:
            N += f.vertices.shape[0]
            m += f.vertices.sum(axis=0)
            I += (f.vertices[:, :, None] * f.vertices[:, None, :]).sum(axis=0)
            v += np.abs(f.surfVolume())
            sa += np.abs(f.surfArea())
        m /= N
        v /= len(fv)
        sa /= len(fv)
        I = I/N - m[:, None] * m[None, :]
        fv[0].getEdges()
        if len(fv[0].bdry) == 0:
            S = Ellipse_pygal(center=m, I=I, targetSize=targetSize)
            w = np.fabs(S.surfVolume())
            S.updateVertices(m[None, :] + (S.vertices - m[None, :])*(v/w)**(1/3))
            logging.info(f'closed surface; Volumes: {v:0.4f} {w:0.4f} {np.fabs(S.surfVolume()):0.4f}')
        else:
            D, V = LA.eigh(I)
            S = FlatEllipse(a=np.sqrt(D[2]), b=np.sqrt(D[1]), outerRes=100,
                            nrings=10)
            sa2 = np.abs(S.surfArea())
            S.updateVertices(m[None, :] + (S.vertices[:, ::-1] @ V) * (sa/sa2)**(1/2))
            logging.info(f'open surface; Areas: {sa:0.4f} {sa2:0.4f} {np.fabs(S.surfArea()):0.4f}')
        return S

    def dataTerm(self, _fvDef, var=None):
        c = float(self.Ntarg)/self.select.sum()
        obj = 0
        if self.options['errorType'] == 'L2Norm':
            for k, f in enumerate(_fvDef):
                if self.select[k]:
                    obj += c*sd.L2Norm(f, self.fv1[k].vfld) / (self.options['sigmaError'] ** 2)
        else:
            for k, f in enumerate(_fvDef):
                if self.select[k]:
                    obj += c*self.fun_obj(f, self.fv1[k]) / (self.options['sigmaError']**2)
        return obj

    def objectiveFun(self, force=False):
        if self.obj is None or force:
            self.obj0 = 0
            c = float(self.Ntarg) / self.select.sum()
            for k, fv in enumerate(self.fv1):
                if self.select[k]:
                    self.obj0 += c*self.fun_obj0(fv) / (self.options['sigmaError']**2)

            self.obj = self.obj0

            affineDim = self.affineDim
            self.affineDim = 0
            var = {'kernel': self.options['KparDiff0'],
                   'regWeight': self.options['lambdaPrior']}
            obj, st = self.objectiveFunDef(self.control['c0'], var=var,
                                           withTrajectory=True)
            self.state['xt'] = st['xt']
            self.affineDim = affineDim
            self.fvTmpl.updateVertices(np.squeeze(self.state['xt'][-1]))
            self.obj += obj
            ff = surfaces.Surface(surf=self.fv0)
            for (kk, ctr) in enumerate(self.control['cAll']):
                if self.select[kk]:
                    var = {'fv0': self.fvTmpl}
                    obj, st = self.objectiveFunDef(ctr, var=var,
                                                   withTrajectory=True)
                    self.state['stAll'][kk] = st
                    ff.updateVertices(st['xt'][-1, :, :])
                    self.obj += c*(obj + self.fun_obj(ff, self.fv1[kk]) / (self.options['sigmaError']**2))
                    #self.xtAll[kk] = np.copy(foo[1])
                    self.fvDef[kk] = surfaces.Surface(surf=ff)

        return self.obj

    def getVariable(self):
        return self.fvTmpl

    def copyDir(self, dr):
        return deepcopy(dr)

    def randomDir(self):
        dfoo = Control()
        if self.options['updateTemplate']:
            dfoo['c0']['at'] = np.random.randn(self.Tsize,
                                               self.fv0.vertices.shape[0],
                                               self.fv0.vertices.shape[1])
        else:
            dfoo['c0']['at'] = np.zeros((self.Tsize,
                                         self.fv0.vertices.shape[0],
                                         self.fv0.vertices.shape[1]))

        dfoo['cAll'] = []
        if self.options['updateAllTraj']:
            for k in range(self.Ntarg):
                smc = SMControl()
                smc['at'] = np.random.randn(self.Tsize,
                                            self.fv0.vertices.shape[0],
                                            self.fv0.vertices.shape[1])
                if self.options['updateAffine']:
                    smc['Afft'] = np.random.randn(self.Tsize, self.affineDim)
                elif self.affineDim > 0:
                    smc['Afft'] = np.zeros((self.Tsize, self.affineDim))
                dfoo['cAll'].append(smc)
        else:
            for k in range(self.Ntarg):
                smc = SMControl()
                smc['at'] = np.zeros((self.Tsize,
                                      self.fv0.vertices.shape[0],
                                      self.fv0.vertices.shape[1]))
                if self.affineDim > 0:
                    smc['Afft'] = np.zeros((self.Tsize, self.affineDim))
                dfoo['cAll'].append(smc)
        return(dfoo)

    def updateTry(self, direction, eps, objRef=None):
        objTry = self.obj0
        controlTry = Control()
        if self.options['updateTemplate']:
            # controlTry['c0'] = SMControl()
            controlTry['c0']['at'] = self.control['c0']['at'] - eps * direction['c0']['at']
        else:
            controlTry = deepcopy(self.control)

        var = {'kernel': self.options['KparDiff0'],
               'regWeight': self.options['lambdaPrior']}
        affineDim = self.affineDim
        self.affineDim = 0
        obj_, st = self.objectiveFunDef(controlTry['c0'], var=var,
                                        withTrajectory=True)
        self.affineDim = affineDim
        objTry += obj_
        x0 = np.squeeze(st['xt'][-1, ...])
        fvTmplTry = surfaces.Surface(surf=self.fv0)
        fvTmplTry.updateVertices(x0)
        controlTry['cAll'] = []
        _ff = []
        c = float(self.Ntarg)/self.select.sum()
        for (kk, d) in enumerate(direction['cAll']):
            if self.select[kk]:
                smc = SMControl()
                smc['at'] = self.control['cAll'][kk]['at'] - eps * d['at']
                if self.options['updateAffine']:
                    smc['Afft'] = self.control['cAll'][kk]['Afft'] - eps * d['Afft']
                else:
                    smc['Afft'] = deepcopy(self.control['cAll'][kk]['Afft'])
                controlTry['cAll'].append(smc)
                var = {'fv0': fvTmplTry}
                obj_, st = self.objectiveFunDef(smc, var=var,
                                                withTrajectory=True)
                objTry += c*obj_
                ff = surfaces.Surface(surf=self.fv0)
                ff.updateVertices(np.squeeze(st['xt'][-1, ...]))
                _ff.append(ff)
            else:
                _ff.append([])
                smc = SMControl()
                smc['at'] = deepcopy(self.control['cAll'][kk]['at'])
                smc['Afft'] = deepcopy(self.control['cAll'][kk]['Afft'])
                controlTry['cAll'].append(smc)

        objTry += self.dataTerm(_ff)

        if (objRef is None) or (objTry < objRef):
            self.controlTry = controlTry
            self.objTry = objTry
            self.fvTmplTry = surfaces.Surface(fvTmplTry)
            for (kk, f) in enumerate(_ff):
                if self.select[kk]:
                    self.fvDefTry[kk] = surfaces.Surface(surf=f)

        return objTry

    def gradientComponent(self, q, kk, fvDef, ctrAll, fvTmpl):
        if self.options['errorType'] == 'L2Norm':
            px1 = -sd.L2NormGradient(fvDef, self.fv1[kk].vfld) / self.options['sigmaError'] ** 2
        else:
            px1 = -self.fun_objGrad(fvDef, self.fv1[kk]) / self.options['sigmaError']**2

        hgrad = self.hamiltonianGradient(px1, kernel=self.options['KparDiff'],
                                         fv0=fvTmpl, control=ctrAll)
        grd = SMControl()
        grd['dat'] = hgrad['dat']
        grd['dA'] = hgrad['dA']
        grd['db'] = hgrad['db']
        pxTmpl = hgrad['pxt'][0, ...]

        q.put([kk, grd, pxTmpl])
        return True

    def gradientComponent_no_queue(self, kk, fvDef, ctrAll, fvTmpl):
        if self.options['errorType'] == 'L2Norm':
            px1 = -sd.L2NormGradient(fvDef, self.fv1[kk].vfld) / self.options['sigmaError'] ** 2
        else:
            px1 = -self.fun_objGrad(fvDef, self.fv1[kk]) / self.options['sigmaError']**2

        hgrad = self.hamiltonianGradient(px1, kernel=self.options['KparDiff'],
                                         fv0=fvTmpl, control=ctrAll)
        # grd = SMControl()
        # grd['dat'] = hgrad['dat']
        # grd['dA'] = hgrad['dA']
        # grd['db'] = hgrad['db']
        # pxTmpl = hgrad['pxt'][0, ...]
        # q.put([kk, grd, pxTmpl])
        return hgrad

    def testEndpointGradient(self):
        _ff0 = []
        _dff = []
        _ff1 = []

        eps = 1e-6
        incr = 0
        for k in range(self.Ntarg):
            ff = surfaces.Surface(surf=self.fvDef[k])
            dff = np.random.normal(size=ff.vertices.shape)
            ff.updateVertices(ff.vertices+eps*dff)
            _ff0.append(ff)
            ff = surfaces.Surface(surf=self.fvDef[k])
            ff.updateVertices(ff.vertices-eps*dff)
            _ff1.append(ff)
            _dff.append(dff)
            if self.options['errorType'] == 'L2Norm':
                grd = sd.L2NormGradient(self.fvDef[k], self.fv1[k].vfld)\
                    / self.options['sigmaError'] ** 2
            else:
                grd = self.fun_objGrad(self.fvDef[k], self.fv1[k]) \
                    / self.options['sigmaError']**2
            incr += (grd*dff).sum()
        c1 = self.dataTerm(_ff1)
        c0 = self.dataTerm(_ff0)
        logging.info(f"test endpoint gradient: {(c0-c1)/(2*eps):.5f} {incr:.5f}")

    def getGradient(self, coeff=1.0, update=None):
        return self.getGradient_no_queue(coeff, update)

    def getGradient_no_queue(self, coeff=1.0, update=None):
        if update is None:
            control = self.control
            fvTmpl = self.fvTmpl
            fvDef = self.fvDef
        else:
            control = Control()
            control['c0']['at'] = self.control['c0']['at']\
                - update[1] * update[0]['c0']['at']
            affineDim = self.affineDim
            self.affineDim = 0
            # obj1, st = self.objectiveFunDef(control['c0'],
            #                                 var={'kernel': self.options['KparDiff0']},
            #                                 withTrajectory=True)
            st = self.solveStateEquation(control['c0'],
                                         kernel=self.options['KparDiff0'],
                                         init_state=self.fv0.vertices)
            self.affineDim = affineDim
            self.state['st0'] = st
            fvTmpl = surfaces.Surface(surf=self.fvTmpl)
            fvTmpl.updateVertices(np.squeeze(st['xt'][-1, :, :]))
            fvDef = []

            for kk in range(self.Ntarg):
                if self.select[kk]:
                    ctr = SMControl()
                    ctr['at'] = self.control['cAll'][kk]['at']\
                        - update[1] * update[0]['cAll'][kk]['at']
                    # print(update[0]['cAll'][kk].keys())
                    if update[0]['cAll'][kk]['Afft'] is not None:
                        ctr['Afft'] = self.control['cAll'][kk]['Afft']\
                            - update[1] * update[0]['cAll'][kk]['Afft']
                    control['cAll'].append(ctr)
                    stAll = self.solveStateEquation(ctr, kernel=self.options['KparDiff'],
                                                    init_state=fvTmpl.vertices)

                    # obj1, stAll = self.objectiveFunDef(ctr,
                    #                                    var={'kernel': self.options['KparDiff'], 'fv0':fvTmpl},
                    #                                    withTrajectory=True)
                    self.state['stAll'][kk] = stAll
                    fvDef.append(surfaces.Surface(self.fvDef[kk]))
                    fvDef[kk].updateVertices(stAll['xt'][-1, ...])
                else:
                    control['cAll'].append(None)
                    fvDef.append(None)

        pxTmpl = np.zeros(self.control['c0']['at'].shape[1:])

        grd = Control()
        for kk in range(self.Ntarg):
            dir = SMControl()
            dir['at'] = np.zeros(self.control['c0']['at'].shape)
            grd['cAll'].append(dir)

        dim2 = self.dim**2
        c = self.Ntarg/self.select.sum()
        for kk in range(self.Ntarg):
            if self.select[kk]:
                grd_ = self.gradientComponent_no_queue(kk, fvDef[kk],
                                                       control['cAll'][kk],
                                                       fvTmpl)
                dat = grd_['dat']/(coeff*self.Tsize)
                smc = SMControl()
                smc['at'] = c*dat
                if self.affineDim > 0:
                    dA = grd_['dA']
                    db = grd_['db']
                    dAfft = 2*self.affineWeight.reshape([1, self.affineDim]) *\
                        self.control['cAll'][kk]['Afft']

                    for t in range(self.Tsize):
                        dAff = self.affineBasis.T\
                            @ np.vstack([dA[t].reshape([dim2, 1]),
                                        db[t].reshape([self.dim, 1])])
                        dAfft[t] -= dAff.reshape(dAfft[t].shape)
                    dAfft /= (self.coeffAff*coeff*self.Tsize)
                    smc['Afft'] = c*dAfft
                grd['cAll'][kk] = smc
                pxTmpl += c*grd_['pxt'][0]

        affineDim = self.affineDim
        self.affineDim = 0
        res = self.hamiltonianGradient(pxTmpl, kernel=self.options['KparDiff0'],
                                       regWeight=self.options['lambdaPrior'],
                                       fv0=self.fv0, control=control['c0'])
        self.affineDim = affineDim
        if self.options['updateTemplate']:
            grd['c0']['at'] = res['dat'] / (self.tmplCoeff*coeff*self.Tsize)
        else:
            grd['c0']['at'] = np.zeros(res['dat'].shape)

        return grd

    def getGradient_with_queue(self, coeff=1.0, update=None):
        if update is None:
            control = self.control
            fvTmpl = self.fvTmpl
            fvDef = self.fvDef
        else:
            control = Control()
            control['c0']['at'] = self.control['c0']['at']\
                - update[1] * update[0]['c0']['at']
            affineDim = self.affineDim
            self.affineDim = 0
            # obj1, st = self.objectiveFunDef(control['c0'],
            #                                 var={'kernel': self.options['KparDiff0']},
            #                                 withTrajectory=True)
            st = self.solveStateEquation(control['c0'],
                                         kernel=self.options['KparDiff0'],
                                         init_state=self.fv0.vertices)
            self.affineDim = affineDim
            self.state['st0'] = st
            fvTmpl = surfaces.Surface(surf=self.fvTmpl)
            fvTmpl.updateVertices(np.squeeze(st['xt'][-1, :, :]))
            fvDef = []

            for kk in range(self.Ntarg):
                if self.select[kk]:
                    ctr = SMControl()
                    ctr['at'] = self.control['cAll'][kk]['at']\
                        - update[1] * update[0]['cAll'][kk]['at']
                    # print(update[0]['cAll'][kk].keys())
                    if update[0]['cAll'][kk]['Afft'] is not None:
                        ctr['Afft'] = self.control['cAll'][kk]['Afft']\
                            - update[1] * update[0]['cAll'][kk]['Afft']
                    control['cAll'].append(ctr)
                    stAll = self.solveStateEquation(ctr, kernel=self.options['KparDiff'],
                                                    init_state=fvTmpl.vertices)
                    self.state['stAll'][kk] = stAll
                    fvDef.append(surfaces.Surface(self.fvDef[kk]))
                    fvDef[kk].updateVertices(stAll['xt'][-1, ...])
                else:
                    control['cAll'].append(None)
                    fvDef.append(None)

        pxTmpl = np.zeros(self.control['c0']['at'].shape[1:])
        q = mp.Queue()

        procGrd = []
        for kk in range(self.Ntarg):
            procGrd.append(mp.Process(target=self.gradientComponent,
                                        args=(q, kk, fvDef[kk],
                                            control['cAll'][kk], fvTmpl)))
        for kk in range(self.Ntarg):
            logging.info('{0:d}, {1:d}'.format(kk, self.Ntarg))
            procGrd[kk].start()
        logging.info("end start")
        for kk in range(self.Ntarg):
            logging.info("join {0:d}".format(kk))
            procGrd[kk].join()
            # self.compGrd[kk] = True
        # print 'all joined'

        grd = Control()
        for kk in range(self.Ntarg):
            dir = SMControl()
            dir['at'] = np.zeros(self.control['c0']['at'].shape)
            grd['cAll'].append(dir)

        dim2 = self.dim**2
        c = self.Ntarg/self.select.sum()
        while not q.empty():
            kk_, grd_, pxTmpl_ = q.get()
            dat = grd_['dat']/(coeff*self.Tsize)
            smc = SMControl()
            smc['at'] = c*dat
            if self.affineDim > 0:
                dA = grd_['dA']
                db = grd_['db']
                dAfft = 2 * self.affineWeight.reshape([1, self.affineDim]) *\
                    self.control['cAll'][kk_]['Afft']

                for t in range(self.Tsize):
                    dAff = self.affineBasis.T\
                        @ np.vstack([dA[t].reshape([dim2, 1]),
                                     db[t].reshape([self.dim, 1])])
                    dAfft[t] -= dAff.reshape(dAfft[t].shape)
                dAfft /= (self.coeffAff*coeff*self.Tsize)
                smc['Afft'] = c*dAfft
            grd['cAll'][kk_] = smc
            pxTmpl += c*pxTmpl_

        affineDim = self.affineDim
        self.affineDim = 0
        res = self.hamiltonianGradient(pxTmpl, kernel=self.options['KparDiff0'],
                                       regWeight=self.options['lambdaPrior'],
                                       fv0=self.fv0, control=control['c0'])
        self.affineDim = affineDim
        if self.options['updateTemplate']:
            grd['c0']['at'] = res['dat'] / (self.tmplCoeff*coeff*self.Tsize)
        else:
            grd['c0']['at'] = np.zeros(res['dat'].shape)

        return grd

    def dotProduct_Riemannian(self, g1, g2):
        res = np.zeros(len(g2))
        for (kk, gg1) in enumerate(g1['cAll']):
            if self.select[kk]:
                for t in range(self.Tsize):
                    z = self.state['stAll'][kk]['xt'][t, :, :]
                    gg = np.squeeze(gg1['at'][t, :, :])
                    u = self.options['KparDiff'].applyK(z, gg)
                    uu = gg1['Afft'][t]
                    ll = 0
                    for gr in g2:
                        ggOld = np.squeeze(gr['cAll'][kk]['at'][t, :, :])
                        res[ll] += (ggOld*u).sum()
                        if self.affineDim > 0:
                            res[ll] += (uu*gr['cAll'][kk]['Afft'][t]).sum() * self.coeffAff
                        ll = ll + 1
        for t in range(g1['c0']['at'].shape[0]):
            z = self.state['st0']['xt'][t, :, :]
            gg = g1['c0']['at'][t, :, :]
            u = self.options['KparDiff0'].applyK(z, gg)
            ll = 0
            for gr in g2:
                ggOld = gr['c0']['at'][t, :, :]
                res[ll]  += (ggOld*u).sum()*(self.tmplCoeff)
                ll = ll + 1
        return res

    def dotProduct_euclidean(self, g1, g2):
        res = np.zeros(len(g2))
        for (kk, gg1) in enumerate(g1['cAll']):
            if self.select[kk]:
                for t in range(self.Tsize):
                    gg = gg1['at'][t, :, :]
                    if self.affineDim > 0:
                        uu = gg1['Afft'][t]
                    else:
                        uu = 0
                    ll = 0
                    for gr in g2:
                        ggOld = gr['cAll'][kk]['at'][t, :, :]
                        res[ll] += (ggOld*gg).sum()
                        if self.affineDim > 0:
                            res[ll] += (uu*gr['cAll'][kk]['Afft'][t]).sum() * self.coeffAff
                        ll = ll + 1
        for t in range(g1['c0']['at'].shape[0]):
            gg = g1['c0']['at'][t, :, :]
            ll = 0
            for gr in g2:
                ggOld = gr['c0']['at'][t, :, :]
                res[ll]  += (ggOld*gg).sum() *(self.tmplCoeff)
                ll = ll + 1
        return res

    def acceptVarTry(self):
        self.obj = self.objTry
        self.control['c0'] = deepcopy(self.controlTry['c0'])
        self.fvTmpl = surfaces.Surface(self.fvTmplTry)
        for kk, f in enumerate(self.fvDefTry):
            if self.select[kk]:
                self.control['cAll'][kk] = deepcopy(self.controlTry['cAll'][kk])
                self.fvDef[kk] = surfaces.Surface(surf=f)
        #logging.info('Obj Fun: ' + str(self.objectiveFun(force=True)))

    def SMaddProd(self, dir1, dir2, beta):
        dr = SMControl()
        for k in dir1.keys():
            if dir1[k] is not None and dir2[k] is not None:
                dr[k] = dir1[k] + beta * dir2[k]
        return dr

    def addProd(self, dir1, dir2, coeff):
        res = Control()
        res['c0'] = self.SMaddProd(dir1['c0'], dir2['c0'], coeff)
        for kk, dd in enumerate(dir1['cAll']):
            if self.select[kk]:
                smc = self.SMaddProd(dir1['cAll'][kk], dir2['cAll'][kk], coeff)
                res['cAll'].append(smc)
            else:
                res['cAll'].append(SMControl())
            # print('addprod ', self.select[kk], res['cAll'][kk].keys())
                # res['all'][kk]['diff'] = dd['diff'] + coeff*dir2['all'][kk]['diff']
                # res['all'][kk]['aff'] = dd['aff'] + coeff*dir2['all'][kk]['aff']
        return res

    def prod(self, dir1, coeff):
        res = Control()
        res['c0'] = super().prod(dir1['c0'], coeff)
        for kk, dd in enumerate(dir1['cAll']):
            if self.select[kk]:
                smc = super().prod(dir1['cAll'][kk], coeff)
                res['cAll'].append(smc)
            else:
                res['cAll'].append(SMControl())
        return res

    def saveHdf5(self, fileName='result'):
        fout = h5py.File(fileName, 'w')
        LDDMMResult = fout.create_group('LDDMM Results')
        parameters = LDDMMResult.create_group('parameters')
        parameters.create_dataset('Time steps', data=self.Tsize)
        parameters.create_dataset('Template Deformation Kernel type', data = self.options['KparDiff0'].name)
        parameters.create_dataset('Template Deformation Kernel width', data = self.options['KparDiff0'].sigma)
        parameters.create_dataset('TemplateDeformation Kernel order', data = self.options['KparDiff0'].order)
        parameters.create_dataset('Deformation Kernel type', data = self.options['KparDiff'].name)
        parameters.create_dataset('Deformation Kernel width', data = self.options['KparDiff'].sigma)
        parameters.create_dataset('Deformation Kernel order', data = self.options['KparDiff'].order)
        parameters.create_dataset('Spatial Varifold Kernel type', data = self.options['KparDist'].name)
        parameters.create_dataset('Spatial Varifold width', data = self.options['KparDist'].sigma)
        parameters.create_dataset('Spatial Varifold order', data = self.options['KparDist'].order)
        htemplate = LDDMMResult.create_group('hypertemplate')
        htemplate.create_dataset('vertices', data=self.fv0.vertices)
        htemplate.create_dataset('faces', data=self.fv0.faces)
        template = LDDMMResult.create_group('template')
        template.create_dataset('vertices', data=self.fvTmpl.vertices)
        template.create_dataset('alpha', data=self.control['c0']['at'])
        targets = []
        for t in range(self.Ntarg):
            target = LDDMMResult.create_group(f'target{t}')
            target.create_dataset('vertices', data=self.fv1[t].vertices)
            target.create_dataset('faces', data=self.fv1[t].faces)
            deformedTemplate = LDDMMResult.create_group(f'deformedTemplate{t}')
            deformedTemplate.create_dataset('vertices', data=self.fvDef[t].vertices)
            deformedTemplate.create_dataset('alpha', data=self.control['cAll'][t]['at'])
            deformedTemplate.create_dataset('affine', data='None')

            descriptors = deformedTemplate.create_group('descriptors')

            AV0 = self.fvTmpl.computeVertexArea()
            AV = self.fvDef[t].computeVertexArea()[0]/AV0[0]
            if self.state['stAll'][t]['Jt'] is not None:
                descriptors.create_dataset('Jacobian', data=self.state['stAll'][t]['Jt'][-1,:])
            else:
                descriptors.create_dataset('Jacobian', data=np.empty(0))
            descriptors.create_dataset('Surface Jacobian', data=AV)
            descriptors.create_dataset('Displacement', data=self.state['stAll'][t]['xt'][-1,...]
                                       -self.state['stAll'][t]['xt'][0,...])

        fout.close()

    def testGradientFun(self, obj, grd, gradCoeff, opt=None, dotProduct=None):
        self.testEndpointGradient()
        dirfoo0 = self.randomDir()
        epsfoo = 1e-8
        dirfoo = deepcopy(dirfoo0)
        if self.affineDim > 0:
            dirfoo['Afft'] *= 0
        if self.unreduced:
            dirfoo['ct'] *= 0
        objfoo1 = self.updateTry(dirfoo, epsfoo, obj - 1e10)
        [grdfoo] = self.dotProduct(grd, [dirfoo])
        objfoo2 = self.updateTry(dirfoo, -epsfoo, obj - 1e10)
        logging.info('Test Gradient: %.6f %.6f' % ((objfoo1 - objfoo2) / (2.0 * epsfoo), -grdfoo * gradCoeff))



    def endOfIteration(self, forceSave=False):
        self.iter += 1
        #if self.testGradient:
        #    self.testEndpointGradient()
        if self.iter >= self.affBurnIn:
            self.options['updateAffine'] = False
            self.coeffAff = self.coeffAff2
        if forceSave or self.iter >= self.options['templateBurnIn']:
            self.updateAllTraj = True
        affineDim = self.affineDim
        self.affineDim = 0
        obj1, st = self.objectiveFunDef(self.control['c0'], var = {'kernel': self.options['KparDiff0']},
                                                   withTrajectory=True, withJacobian=True)
        self.affineDim = affineDim
        self.state['st0'] = st
        self.fvTmpl.updateVertices(st['xt'][-1, :, :])
        self.fvTmpl.saveVTK(self.outputDir +'/'+ 'Template.vtk', scalars = st['Jt'][-1, :,0], scal_name='Jacobian')
        for kk in range(self.Ntarg):
            if self.select[kk]:
                obj1, st = self.objectiveFunDef(self.control['cAll'][kk],
                                                var = {'kernel': self.options['KparDiff'], 'fv0':self.fvTmpl},
                                                withTrajectory=True, withJacobian=True)
                self.state['stAll'][kk] = st
                self.fvDef[kk].updateVertices(st['xt'][-1, ...])
                self.fvDef[kk].saveVTK(self.outputDir +'/'+ self.options['saveFile']+str(kk)+'.vtk', scalars = st['Jt'][-1, :,0],
                                       scal_name='Jacobian')
        if forceSave or self.iter % self.options['saveRate'] == 0:
            logging.info('saving hdf5')
            self.saveHdf5(self.outputDir+'output.h5')
        if self.pplot:
            fig=plt.figure(1)
            #fig.clf()
            ax = Axes3D(fig,)
            lim0 = self.addSurfaceToPlot(self.fvTmpl, ax, ec = 'k', fc = 'b')
            ax.set_xlim(lim0[0][0], lim0[0][1])
            ax.set_ylim(lim0[1][0], lim0[1][1])
            ax.set_zlim(lim0[2][0], lim0[2][1])
            #ax.auto_scale()
            plt.pause(0.1)
        #logging.info('Obj Fun: ' + str(self.objectiveFun(force=True)))


    #def endOfProcedure(self):
    #    logging.info('Obj Fun: ' + str(self.objectiveFun(force=True)))


    def computeTemplate(self):
        # self.coeffAff = self.coeffAff2
        # grd = self.getGradient(self.gradCoeff)
        # [grd2] = self.dotProduct(grd, [grd])
        # self.coeffAff = self.coeffAff1
        # self.gradEps = max(0.1, np.sqrt(grd2) / 10000)
        meanObj = 0
        if self.options['sgd'] < self.Ntarg:
            for k in range(self.options['maxIter']//self.options['sgd']):
                self.select = np.zeros(self.Ntarg, dtype=bool)
                sel = np.random.permutation(self.Ntarg)
                #sel[0] = 8
                s = 'Selected: '
                for kk in range(self.options['sgd']):
                    s += str(sel[kk]) + ' '
                    self.select[sel[kk]] = True
                #self.select[144] = True
                logging.info('\nRandom step ' + str(k) + ' ' + s)
                self.epsMax = 10./(self.options['sgd']*(k+1))
                self.reset = True
                meanObj += self.objectiveFun()
                cg.cg(self, verb=self.options['verb'], maxIter=10, TestGradient=self.options['testGradient'],
                      epsInit=0.01, Wolfe=False)
                # else:
                #     bfgs.bfgs(self, verb=self.options['verb'], maxIter=10, TestGradient=True,
                #               epsInit=1.)
                logging.info('\nMean Objective {0:f}'.format(meanObj/(k+1)))
            nv0 = self.fv0.vertices.shape[0]
            self.fv0.subDivide(1)
            self.fv0.Simplify(nv0)
            #sgd = (10, 0.00001)
        else:
            if self.options['algorithm'] == 'cg':
                cg.cg(self, verb=self.options['verb'], maxIter=self.options['maxIter'],
                      TestGradient=self.options['testGradient'], epsInit=1.)
            else:
                bfgs.bfgs(self, verb=self.options['verb'], maxIter=self.options['maxIter'],
                      TestGradient=self.options['testGradient'], epsInit=1.)
        return self



