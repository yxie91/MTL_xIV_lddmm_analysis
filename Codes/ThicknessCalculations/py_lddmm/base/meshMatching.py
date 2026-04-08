from copy import deepcopy
import numpy as np
import h5py
import logging
from functools import partial
from . import kernelFunctions as kfun
from . import meshes, meshDistances as msd
from . import pointSetMatching
from .vtk_fields import vtkFields


# Main class for image varifold matching
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
class MeshMatching(pointSetMatching.PointSetMatching):
    def __init__(self, Template, Target, options=None):
        # param=None, maxIter=1000,
        #          regWeight = 1.0, affineWeight = 1.0, verb=True,
        #          rotWeight = None, scaleWeight = None, transWeight = None,
        #          testGradient=True, saveFile = 'evolution', internalCost = None, internalWeight = 1.,
        #          saveTrajectories = False, affine = 'none', outputDir = '.',pplot=True):

        # if param is None:
        #     self.param = MeshMatchingParam()
        # else:
        #     self.param = param

        super().__init__(Template, Target, options)
        # self.setInitialOptions(options)
        # self.internalCost = internalCost
        # self.internalWeight = internalWeight
        # pointSetMatching.PointSetMatching.__init__(self, Template=Template, Target=Target, param=param, maxIter=maxIter,
        #          regWeight=regWeight, affineWeight=affineWeight, verb=verb,
        #          rotWeight=rotWeight, scaleWeight=scaleWeight, transWeight=transWeight,
        #          testGradient=testGradient, saveFile=saveFile,
        #          saveTrajectories=saveTrajectories, affine=affine, outputDir=outputDir, pplot=pplot)

        self.Kim_dtype = self.options['pk_dtype']

    def getDefaultOptions(self):
        options = super().getDefaultOptions()
        options['KparIm'] = None
        options['lame_lambda'] = None
        options['lame_mu'] = None
        options['meshIntegrity'] = True
        options['meshIntegrityCoeff'] = 1000.
        return options

    def set_parameters(self):
        super().set_parameters()
        sigmaKim = 6.5
        orderKim = 3
        typeKim = 'gauss'
        self.meanVolume = (self.fv0.volumes).mean()
        if type(self.options['KparIm']) in (list, tuple):
            typeKim = self.options['KparIm'][0]
            sigmaKim = self.options['KparIm'][1]
            if typeKim == 'laplacian' and len(self.options['KparIm']) > 2:
                orderKim = self.options['KparIm'][2]
            self.options['KparIm'] = None

        if self.options['KparIm'] is None:
            self.options['KparIm'] = kfun.Kernel(name=typeKim, sigma=sigmaKim, order=orderKim)

    def createObject(self, data, other=None):
        if isinstance(data, meshes.Mesh):
            fv = meshes.Mesh(data=data)
        else:
            fv = meshes.Mesh(data=self.fv0)
            fv.updateVertices(data)
        return fv

    # def initialize_variables(self):
    #     self.x0 = np.copy(self.fv0.vertices)
    #     self.fvDef = deepcopy(self.fv0)
    #     self.npt = self.x0.shape[0]
    #
    #     self.Tsize = int(round(1.0/self.options['timeStep']))
    #     self.control['at'] = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
    #     self.controlTry['at'] = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
    #     self.control['Afft'] = np.zeros([self.Tsize, self.affineDim])
    #     self.controlTry['Afft'] = np.zeros([self.Tsize, self.affineDim])
    #     self.state['xt'] = np.tile(self.x0, [self.Tsize+1, 1, 1])
    #     self.v = np.zeros([self.Tsize+1, self.npt, self.dim])

    def set_template_and_target(self, Template, Target, misc=None):
        self.fv0 = meshes.Mesh(data=Template, checkOrientation=True)
        self.fv1 = meshes.Mesh(data=Target, checkOrientation=True)

        self.fv0.save(self.outputDir + '/Template.vtk')
        self.fv1.save(self.outputDir + '/Target.vtk')
        self.dim = self.fv0.vertices.shape[1]

    def set_fun(self, errorType, vfun=None):
        self.options['errorType'] = errorType
        self.fun_obj0 = partial(msd.varifoldNorm0, KparDist=self.options['KparDist'], KparIm=self.options['KparIm'])
        self.fun_obj = partial(msd.varifoldNormDef, KparDist=self.options['KparDist'], KparIm=self.options['KparIm'])
        self.fun_objGrad = partial(msd.varifoldNormGradient, KparDist=self.options['KparDist'],
                                   KparIm=self.options['KparIm'])
        # self.fun_obj0 = msd.varifoldNorm0
        # self.fun_obj = msd.varifoldNormDef
        # self.fun_objGrad = msd.varifoldNormGradient

        self.set_extra_term()

    def set_extra_term(self):
        if self.unreduced:
            if self.options['internalCost'] is not None:
                logging.warning('Warning: Internal cost option not available for unreduced variables')
                self.options['internalCost'] = None

        self.internalcostList = []
        if isinstance(self.options['internalCost'], str):
            self.options['internalCost'] = [[self.options['internalCost'], self.options['internalWeight']]]
        elif self.options['internalCost'] is None:
            self.options['internalCost'] = []
        else:
            self.options['internalCost'] = list(self.options['internalCost'])

        # if self.options['meshIntegrity']:
        #     extraTerm = {}
        #     extraTerm['fun'] = partial(msd.meshConsistencyConstraint, faces=self.fv0.faces, scale=0.01*self.meanVolume)
        #     extraTerm['grad'] = partial(msd.meshConsistencyConstraintGrad, faces=self.fv0.faces,
        #                                 scale = 0.01*self.meanVolume)
        #     extraTerm['coeff'] = self.options['meshIntegrityCoeff']
        #     self.internalcostList.append(extraTerm)

        for d in self.options['internalCost']:
            if d[0] == 'divergence':
                extraTerm = {}
                extraTerm['fun'] = partial(msd.square_divergence, faces=self.fv0.faces)
                extraTerm['grad'] = partial(msd.square_divergence_grad, faces=self.fv0.faces)
                extraTerm['coeff'] = self.options['internalWeight']
                self.internalcostList.append([extraTerm])
            elif d[0] == 'normalized_divergence':
                extraTerm = {}
                extraTerm['fun'] = partial(msd.normalized_square_divergence, faces=self.fv0.faces)
                extraTerm['grad'] = partial(msd.normalized_square_divergence_grad, faces=self.fv0.faces)
                extraTerm['coeff'] = self.options['internalWeight']
            elif d[0] in ('elastic', 'elastic_energy'):
                extraTerm = {}
                if self.options['lame_lambda'] is None:
                    self.options['lame_lambda'] = 1.
                    self.options['lame_mu'] = 1.
                extraTerm['fun'] = partial(msd.elasticEnergy, faces=self.fv0.faces, lbd=self.options['lame_lambda'],
                                           mu = self.options['lame_mu'])
                extraTerm['grad'] = partial(msd.elasticEnergy_grad, faces=self.fv0.faces, lbd=self.options['lame_lambda'],
                                            mu = self.options['lame_mu'])
                extraTerm['coeff'] = self.options['internalWeight']
            else:
                if d[0] is not None:
                    logging.warning("Internal cost not recognized: " + self.options['internalCost'])
                extraTerm = None
            self.internalcostList.append(extraTerm)

        def extraTermFun(x, v):
            res = 0
            for ex in self.internalcostList:
                if ex is not None:
                    res += ex['coeff'] * ex['fun'](x, v)
            return res

        def extraTermGrad(x, v, variables='both'):
            grad = {}
            grad['phi'] = np.zeros(v.shape)
            grad['x'] = np.zeros(x.shape)
            for ex in self.internalcostList:
                #print(ex)
                if ex is not None:
                    grd = ex['grad'](x, v, variables=variables)
                    #print(grd)
                    if variables == 'both' or variables == 'phi':
                        grad['phi'] += ex['coeff'] * grd['phi']
                    if variables == 'both' or variables == 'x':
                        grad['x'] += ex['coeff'] * grd['x']
            return grad

        self.extraTerm = {'fun': extraTermFun, 'grad': extraTermGrad, 'coeff': 1.0}

    def dataTerm(self, _fvDef, _fvInit=None):
        # logging.info('dataTerm ' + self.param.KparIm.name)
        # if self.param.errorType == 'classification':
        #     obj = pointSets.LogisticScoreL2(_fvDef, self.fv1, self.u, w=self.wTr, intercept=self.intercept, l1Cost=self.l1Cost) \
        #           / (self.param.sigmaError**2)
        #     #obj = pointSets.LogisticScore(_fvDef, self.fv1, self.u) / (self.param.sigmaError**2)
        obj = self.fun_obj(_fvDef, self.fv1) / (self.options['sigmaError'] ** 2)
        if self.options['meshIntegrity']:
            obj += self.options['timeStep'] * (self.options['meshIntegrityCoeff'] *
                    msd.meshConsistencyConstraint(_fvDef.vertices, None,
                                                  _fvDef.faces, scale=0.01*self.meanVolume))
        return obj

    def objectiveFun(self):
        if self.obj is None:
            self.obj0 = self.fun_obj0(self.fv1) / (self.options['sigmaError'] ** 2)
            self.objDef, self.state = self.objectiveFunDef(self.control, withTrajectory=True)
            self.fvDef.updateVertices(self.state['xt'][-1, :, :])
            self.objData = self.dataTerm(self.fvDef)
            self.obj = self.obj0 + self.objData + self.objDef
        return self.obj

    def saveHdf5(self, fileName):
        fout = h5py.File(fileName, 'w')
        LDDMMResult = fout.create_group('LDDMM Results')
        parameters = LDDMMResult.create_group('parameters')
        parameters.create_dataset('Time steps', data=self.Tsize)
        parameters.create_dataset('Deformation Kernel type', data=self.options['KparDiff'].name)
        parameters.create_dataset('Deformation Kernel width', data=self.options['KparDiff'].sigma)
        parameters.create_dataset('Deformation Kernel order', data=self.options['KparDiff'].order)
        parameters.create_dataset('Spatial Varifold Kernel type', data=self.options['KparDist'].name)
        parameters.create_dataset('Spatial Varifold width', data=self.options['KparDist'].sigma)
        parameters.create_dataset('Spatial Varifold order', data=self.options['KparDist'].order)
        parameters.create_dataset('Image Varifold Kernel type', data=self.options['KparIm'].name)
        parameters.create_dataset('Image Varifold width', data=self.options['KparIm'].sigma)
        parameters.create_dataset('Image Varifold order', data=self.options['KparIm'].order)
        template = LDDMMResult.create_group('template')
        template.create_dataset('vertices', data=self.fv0.vertices)
        template.create_dataset('faces', data=self.fv0.faces)
        template.create_dataset('image', data=self.fv0.image)
        target = LDDMMResult.create_group('target')
        target.create_dataset('vertices', data=self.fv1.vertices)
        target.create_dataset('faces', data=self.fv1.faces)
        target.create_dataset('image', data=self.fv1.image)
        deformedTemplate = LDDMMResult.create_group('deformedTemplate')
        deformedTemplate.create_dataset('vertices', data=self.fvDef.vertices)
        variables = LDDMMResult.create_group('variables')
        for k in self.control.keys():
            # logging.info('variable: ' + k)
            variables.create_dataset(k, data=self.control[k])
        # if self.control['Afft'] is not None:
        #     variables.create_dataset('affine', data=self.control['Afft'])
        # else:
        #     variables.create_dataset('affine', data='None')
        descriptors = LDDMMResult.create_group('descriptors')

        # if self.affineDim > 0:
        #     A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        #     dim2 = self.dim**2
        #     for t in range(self.Tsize):
        #         AB = np.dot(self.affineBasis, self.control['Afft'][t])
        #         A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
        #         A[1][t] = AB[dim2:dim2 + self.dim]
        # else:
        #     A = None

        st = self.solveStateEquation(options={'withJacobian': True})
        #
        # (xt, Jt) = evol.landmarkDirectEvolutionEuler(self.x0, self.control['at'], self.options['KparDiff'], affine=A,
        #                                              withJacobian=True)
        xt = st['xt']
        Jt = st['Jt']

        AV0 = self.fv0.computeVertexVolume()
        AV = self.fvDef.computeVertexVolume()/AV0
        descriptors.create_dataset('Jacobian', data=Jt[-1, :])
        descriptors.create_dataset('Surface Jacobian', data=AV)
        descriptors.create_dataset('Displacement', data=xt[-1, ...] - xt[0, ...])
        fout.close()

    def makeTryInstance(self, state):
        ff = meshes.Mesh(data=self.fvDef)
        ff.updateVertices(state['xt'][-1, :, :])
        return ff

    def endPointGradient(self, endPoint=None):
        if endPoint is None:
            endPoint = self.fvDef
        px = self.fun_objGrad(endPoint, self.fv1) / self.options['sigmaError']**2
        if self.options['meshIntegrity']:
            px += (self.options['meshIntegrityCoeff'] *
                   msd.meshConsistencyConstraintGrad(endPoint.vertices, None, faces=endPoint.faces,
                                                     variables='x', scale=0.01*self.meanVolume)['x']) * self.options['timeStep']
        return px

    def testEndpointGradient(self):
        ff = deepcopy(self.fvDef)
        dff = np.random.normal(size=ff.vertices.shape)
        eps = 1e-6
        ff.updateVertices(ff.vertices + eps*dff)
        c1 = self.dataTerm(ff)
        ff = deepcopy(self.fvDef)
        ff.updateVertices(ff.vertices - eps*dff)
        c2 = self.dataTerm(ff)
        grd = self.endPointGradient()
        logging.info("test endpoint gradient: {0:.5f} {1:.5f}".format((c1-c2)/(2*eps), (grd*dff).sum()) )

    def setUpdate(self, update):
        control = pointSetMatching.Control()
        for k in update[0].keys():
            if update[0][k] is not None:
                control[k] = self.control[k] - update[1] * update[0][k]

        # if control['Afft'] is not None:
        #     A = self.affB.getTransforms(control['Afft'])
        # else:
        #     A = None

        st = self.solveStateEquation(control=control)
        xt = st['xt']
        endPoint = meshes.Mesh(data=self.fv0)
        endPoint.updateVertices(xt[-1, :, :])
        st = pointSetMatching.State()
        st['xt'] = xt
        return control, st, endPoint

    def resetPK(self, newType=None):
        if newType is None:
            self.options['KparIm'].pk_dtype = self.Kim_dtype
        else:
            self.options['KparIm'].pk_dtype = newType
        super().resetPK(newType=newType)

    def endOfIteration(self, endP=False):
        self.iter += 1
        if self.options['testGradient']:
            self.testEndpointGradient()

        if self.iter >= self.affBurnIn:
            self.coeffAff = self.coeffAff2
        if (self.iter % self.saveRate == 0 or endP):
            logging.info('Saving Points...')
            (obj1, self.state) = self.objectiveFunDef(self.control, withTrajectory=True)

            self.fvDef.updateVertices(self.state['xt'][-1, :, :])

            st = self.solveStateEquation(options={'withJacobian': True})
            xt = st['xt']
            Jt = st['Jt']
            for kk in range(self.Tsize+1):
                fvDef = meshes.Mesh(data=self.fvDef)
                fvDef.updateVertices(xt[kk, :, :])
                vf1 = vtkFields('CELL_DATA', self.fv0.faces.shape[0])
                vf1.scalars['logJacobianFromRatio'] = np.log(np.maximum(fvDef.volumes/self.fv0.volumes, 1e-10))
                vf2 = vtkFields('POINT_DATA', self.fv0.vertices.shape[0])
                vf2.scalars['logJacobianFromODE'] = Jt[kk, :, 0]
                fvDef.save(self.outputDir + '/' + self.options['saveFile'] + str(kk) + '.vtk', vtkFields=(vf2, vf1))

        if self.unreduced:
            self.unreducedResetTrajectories()

        (obj1, self.state) = self.objectiveFunDef(self.control, withTrajectory=True)
        self.fvDef.updateVertices(np.squeeze(self.state['xt'][-1, :, :]), checkOrientation=True)
        self.options['KparDiff'].pk_dtype = self.Kdiff_dtype
        self.options['KparDist'].pk_dtype = self.Kdist_dtype
        self.options['KparIm'].pk_dtype = self.Kim_dtype
        if self.options['algorithm'] != 'sgd':
            logging.info(f'Objective function components: Def={self.objDef:.04f} Data={self.objData+ self.obj0:0.4f}')

    def endOfProcedure(self):
        self.endOfIteration(endP=True)
