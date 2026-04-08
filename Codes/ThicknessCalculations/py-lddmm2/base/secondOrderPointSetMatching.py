import logging
import numpy as np
import numpy.linalg as la
from .pointSets import PointSet, savePoints
from .pointSetMatching import PointSetMatching
from . import pointEvolution as evol
# from .affineBasis import getExponential, gradExponential


class State(dict):
    def __init__(self):
        super().__init__()
        self['xt'] = None
        self['at'] = None
        self['yt'] = None
        self['Jt'] = None


class Control(dict):
    def __init__(self):
        super().__init__()
        self['a0'] = None
        self['Afft'] = None
        self['x0'] = None


## Main class for surface matching
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
class SecondOrderPointSetMatching(PointSetMatching):
    def __init__(self, Template=None, Target=None, options=None):
        # self.rescaleTemplate = rescaleTemplate
        super().__init__(Template=Template, Target=Target, options=options)


    def getDefaultOptions(self):
        options = super().getDefaultOptions()
        options['controlWeight'] = 1.0
        options['initialMomentum'] = None
        options['saveCorrected'] = False
        return options


    def set_parameters(self):
        super().set_parameters()

    def initialize_variables(self):
        self.nvert = self.fv0.vertices.shape[0]
        if self.match_landmarks:
            self.control['x0'] = np.concatenate((self.fv0.vertices, self.tmpl_lmk.vertices), axis=0)
            self.nlmk = self.tmpl_lmk.vertices.shape[0]
        else:
            self.control['x0'] = np.copy(self.fv0.vertices)
            self.nlmk = 0

        self.x0 = self.control['x0']
        if self.match_landmarks:
            self.def_lmk = PointSet(data=self.tmpl_lmk)
        # self.x0 = self.fv0.vertices
        self.fvDef = self.createObject(self.fv0)
        self.npt = self.x0.shape[0]

        self.Tsize = int(round(1/self.options['timeStep']))
        self.state = State()
        self.control = Control()
        self.control['x0'] = self.x0
        if self.options['initialMomentum']==None:
            self.state['xt'] = np.tile(self.x0, [self.Tsize+1, 1, 1])
            self.control['a0'] = np.zeros([self.x0.shape[0], self.x0.shape[1]])
            self.state['at'] = np.tile(self.control['a0'], [self.Tsize+1, 1, 1])
        else:
            self.control['a0'] = self.options['initialMomentum']
            self.state = self.solveStateEquation()

        #self.v = np.zeros([self.Tsize+1, self.npt, self.dim])
        self.controlTry = Control()
        self.controlTry['a0'] = np.zeros([self.x0.shape[0], self.x0.shape[1]])
        self.control['Afft'] = np.zeros([self.Tsize, self.affineDim])
        self.controlTry['Afft'] = np.zeros([self.Tsize, self.affineDim])


    def solveStateEquation(self, control= None, init_state = None, kernel = None, options=None):
        if control is None:
            control = self.control
        if init_state is None:
            init_state = self.x0
        if kernel is None:
            kernel = self.options['KparDiff']

        A = self.affB.getTransforms(control['Afft'])

        return evol.secondOrderEvolution(init_state, control['a0'], kernel, self.options['timeStep'],
                                         affine=A, options=options)

    def  objectiveFunDef(self, control, var = None, withTrajectory = False, withJacobian=False, display=False):
        if var is None or 'Init' not in var:
            x0 = self.x0
        else:
            x0 = var['Init'][0]
            
        a0 = control['a0']
        Afft = control['Afft']

        timeStep = self.options['timeStep']
        dim2 = self.dim**2

        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, Afft[t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]

        st = self.solveStateEquation(control=control, init_state=x0, options={'withJacobian':withJacobian})
        obj0 = 0.5 * (a0 * self.options['KparDiff'].applyK(x0,a0)).sum()
        obj2 = 0
        for t in range(self.Tsize):
            if self.affineDim > 0:
                obj2 +=  timeStep * (self.affineWeight.reshape(Afft[t].shape) * Afft[t]**2).sum()/2
            #print xt.sum(), at.sum(), obj
        obj = obj2+obj0
        if display:
            logging.info(f'deformation terms: init {obj0:.4f}, aff {obj2:.4f}')
        if withJacobian or withTrajectory:
            return obj, st
        else:
            return obj


    def initVariable(self):
        return Control()
    def getVariable(self):
        return self.control
    


    def setUpdate(self, update):
        control = Control()
        for k in update[0].keys():
            if update[0][k] is not None:
                control[k] = self.control[k] - update[1] * update[0][k]
        st = self.solveStateEquation(control=control)

        if self.match_landmarks:
            endPoint0 = self.createObject(self.fv0)
            endPoint0.updateVertices(st['xt'][-1, :self.nvert, :])
            endPoint1 = PointSet(data=st['xt'][-1, self.nvert:, :])
            endPoint = (endPoint0, endPoint1)
        else:
            endPoint = self.createObject(self.fv0)
            self.updateObject(endPoint, st['xt'][-1, :, :])

        return control, st, endPoint


    def getGradient(self, coeff=1.0, update=None):
        if update is None:
            control = self.control
            if self.match_landmarks:
                endPoint = (self.fvDef, self.def_lmk)
            else:
                endPoint = self.fvDef
            # state = self.state
        else:
            control, state, endPoint = self.setUpdate(update)

        A = self.affB.getTransforms(control['Afft'])
        px1 = -self.endPointGradient(endPoint=endPoint)
        pa1 = np.zeros(self.control['a0'].shape)

        foo = evol.secondOrderGradient(self.x0, control['a0'], px1, pa1, self.options['KparDiff'],
                                       self.options['timeStep'], affine=A)
        grd = Control()
        # # if self.euclideanGradient:
        # #     grd['a0'] = self.options['KparDiff'].applyK(self.x0, foo['da0'])/coeff
        # else:
        grd['a0'] = foo['da0']/coeff
        grd['Afft'] = np.zeros(self.control['Afft'].shape)
        if self.affineDim > 0:
            dim2 = self.dim**2
            dA = foo['dA']
            db = foo['db']
            grd['Afft'] = 2*self.affineWeight.reshape([1, self.affineDim])*control['Afft']
            for t in range(self.Tsize):
               dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
               grd['Afft'][t] -=  dAff.reshape(grd['Afft'][t].shape)
            grd['Afft'] /= (self.coeffAff*coeff*self.Tsize)

        return grd



    def randomDir(self):
        dirfoo = Control()
        dirfoo['a0'] = np.random.randn(self.npt, self.dim)
        if self.affineDim > 0:
            dirfoo['Afft'] = np.random.randn(self.Tsize, self.affineDim)
        else:
            dirfoo['Afft'] = None
        return dirfoo


    def dotProduct_Riemannian(self, g1, g2):
        res = np.zeros(len(g2))
        z = self.x0
        gg = g1['a0']
        u = self.options['KparDiff'].applyK(z, gg)
        ll=0
        for gr in g2:
            ggOld = gr['a0']
            res[ll] = res[ll] + (ggOld * u).sum()
            ll = ll + 1

        if self.affineDim > 0:
            for t in range(self.Tsize):
                uu = g1['Afft'][t]
                ll = 0
                for gr in g2:
                    res[ll] += (uu*gr['Afft'][t]).sum() * self.coeffAff
                    ll = ll + 1
        return res

    def dotProduct_euclidean(self, g1, g2):
        res = np.zeros(len(g2))
        u = g1['a0']
        ll=0
        for gr in g2:
            ggOld = gr['a0']
            res[ll] = res[ll] + (ggOld * u).sum()
            ll = ll + 1

        if self.affineDim > 0:
            for t in range(self.Tsize):
                uu = g1['Afft'][t]
                ll = 0
                for gr in g2:
                    res[ll] += (uu*gr['Afft'][t]).sum()
                    ll = ll + 1
        return res


    def saveEvolution(self, fv0, state, passenger=None, fileName='evolution',
                      velocity=None, grid=None):

        if velocity is None:
            v = np.zeros([self.Tsize+1, self.npt, self.dim])

            for kk in range(self.Tsize):
                v[kk] = self.options['KparDiff'].applyK(state['xt'][kk], state['at'][kk])
        else:
            v = velocity
        super().saveEvolution(fv0, state, passenger=passenger, fileName=fileName,
                              velocity=v, grid=grid)
    #
    #     fvDef.updateVertices(self.state['xt'][-1, :self.nvert:, :])
    #     vf = vtkFields('POINT_DATA', self.fv0.vertices.shape[0])
    #     scalars = dict()
    #     scalars['Jacobian'] = self.state['Jt'][kk, :self.nvert]
    #     scalars['displacement'] = displ[:self.nvert]
    #     if self.Tsize > 0:
    #         displ += np.sqrt((v ** 2).sum(axis=-1))
    #     if self.match_landmarks:
    #         pp = PointSet(data=self.state['xt'][kk, self.nvert:, :])
    #         pp.saveVTK(self.outputDir + '/' + self.options['saveFile']+str(kk+self.Tsize) + '_lmk.vtk')
    #     if kk < self.Tsize:
    #         v = self.options['KparDiff'].applyK(self.state['xt'][kk,...], self.state['at'][kk,...])

    #     xt = state['xt']
    #     Jacobian = state['Jt']
    #     if velocity is None:
    #         velocity = self.v
    #
    #     fn = []
    #     if type(fileName) is str:
    #         for kk in range(self.Tsize + 1):
    #             fn.append(fileName + f'{kk:03d}')
    #     else:
    #         fn = fileName
    #
    #     fvDef = self.createObject(fv0)
    #     nvert = fv0.vertices.shape[0]
    #     v = velocity[0, :nvert, :]
    #     displ = np.zeros(nvert)
    #     dt = self.maxT / self.Tsize
    #     for kk in range(self.Tsize + 1):
    #         fvDef.updateVertices(np.squeeze(xt[kk, :nvert, :]))
    #         vf = vtkFields('POINT_DATA', self.fv0.vertices.shape[0])
    #         if Jacobian is not None:
    #             vf.scalars['Jacobian'] = np.exp(Jacobian[kk, :nvert, 0])
    #
    #         vf.scalars['displacement'] = displ
    #         if kk < self.Tsize:
    #             v = velocity[kk, :nvert, :]
    #             kkm = kk
    #         else:
    #             kkm = kk - 1
    #         if self.dim < 3:
    #             x = velocity[kkm, :nvert]
    #             vf.vectors['velocity'] \
    #                 = np.concatenate((x,
    #                                   np.zeros((x.shape[0], 3 - x.shape[1]))),
    #                                  axis=1)
    #         else:
    #             vf.vectors['velocity'] = velocity[kkm, :nvert]
    #         fvDef.save(self.outputDir + '/' + fn[kk] + '.vtk', vf)
    #         displ += dt * np.sqrt((v**2).sum(axis=1))
    #         if self.match_landmarks:
    #             pp = PointSet(data=xt[kk, nvert:, :])
    #             pp.saveVTK(self.outputDir + '/' + fn[kk] + '_lmk.vtk')
    #
    #         if passenger is not None and passenger[0] is not None:
    #             if isinstance(passenger[0], type(self.fv0)):
    #                 fvp = self.createObject(passenger[0])
    #                 fvp.updateVertices(passenger[1][kk, ...])
    #                 fvp.saveVTK(self.outputDir+'/'+fn[kk]+'_passenger.vtk')
    #             else:
    #                 savePoints(self.outputDir+'/'+fn[kk]+'_passenger.vtk',
    #                            passenger[1][kk, ...])
    #
    #         if grid is not None and self.dim == 2:
    #             self.gridDef.vertices = np.copy(grid['yt'][kk, :, :])
    #             if grid['Jyt'] is None:
    #                 self.gridDef.saveVTK(self.outputDir+'/grid'+str(kk)+'.vtk')
    #             else:
    #                 self.gridDef.saveVTK(self.outputDir+'/grid'+str(kk)+'.vtk',
    #                                      vtkFields=vtkFields('POINT_DATA',
    #                                                          self.gridDef.vertices.shape[0],
    #                                                          scalars={'logJacobian':grid['Jyt'][kk, :, 0],
    #                                                                 'finalLogJacobian':grid['Jyt'][-1, :, 0]}))
    #
    #


    def endOfIteration(self, forceSave = False):
        self.iter += 1
        if self.iter >= self.affBurnIn:
            self.affine = 'none'
            #self.coeffAff = self.coeffAff2
        if (self.iter % self.saveRate == 0):
            logging.info('Saving surfaces...')
            control = self.control
            obj1, self.state = self.objectiveFunDef(control, withTrajectory=True, withJacobian=True,
                                                    display=self.options['verb'])
            self.fvDef.updateVertices(self.state['xt'][-1, :, :])
            # dim2 = self.dim**2
            # if self.affineDim > 0:
            #     A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            #     for t in range(self.Tsize):
            #         AB = np.dot(self.affineBasis, self.control['Afft'][t])
            #         A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
            #         A[1][t] = AB[dim2:dim2+self.dim]
            # else:
            #     A = None

            dt = 1.0 / self.Tsize
            if self.options['saveCorrected']:
                f = self.createObject(self.x0)
                if self.match_landmarks:
                    p = PointSet(data=self.tmpl_lmk)
                else:
                    p = None
                X = self.affB.integrateFlow(self.control['Afft'])
                displ = np.zeros(self.x0.shape[0])
                atCorr = np.zeros(self.state['at'].shape)
                for t in range(self.Tsize+1):
                    U = la.inv(X[0][t,...])
                    yyt = self.state['xt'][t,...]
                    yyt = np.dot(yyt - X[1][t, ...], U.T)
                    scalars = dict()
                    scalars['displacement'] = displ
                    if t < self.Tsize:
                        a = self.state['at'][t,...]
                        a = np.dot(a, X[0][t,...])
                        atCorr[t,...] = a
                        vt = self.options['KparDiff'].applyK(yyt, a)
                        vt = np.dot(vt, U.T)
                        displ += dt*np.sqrt((vt**2).sum(axis=-1))
                    self.updateObject(f, yyt)
                    if self.match_landmarks:
                        p.updateVertices(yyt[self.nvert:, :])
                        p.saveVTK(self.outputDir + '/' + self.options['saveFile']+'Corrected'+str(t+self.Tsize) + '_lmk.vtk')
                    scalars = dict()
                    scalars['Jacobian'] = self.state['Jt'][t, :]
                    vectors = dict()
                    vectors['velocity'] = vt
                    savePoints(self.options['outputDir'] +'/'+self.options['saveFile']+'Corrected'+str(t+self.Tsize)+'.vtk',
                               f.vertices, vectors = vectors, scalars=scalars)
#                (foo,zt) = evol.landmarkDirectEvolutionEuler(self.x0, atCorr, self.options['KparDiff'], withPointSet = self.fv0Fine.vertices)
#                for t in range(self.Tsize+1):
#                    f.updateVertices(zt[t,...])
#                    f.saveVTK(self.options['outputDir'] +'/'+self.saveFile+'CorrectedCheck'+str(t)+'.vtk')
#                (foo,foo2,zt) = evol.secondOrderEvolution(self.x0, atCorr[0,...], self.rhot, self.options['KparDiff'], withPointSet = self.fv0Fine.vertices)
#                for t in range(self.Tsize+1):
#                    f.updateVertices(zt[t,...])
#                    f.saveVTK(self.options['outputDir'] +'/'+self.saveFile+'CorrectedCheckBis'+str(t)+'.vtk')
                 

                f = PointSet(data = self.fv1)
                U = la.inv(X[0][-1])
                yyt = f.vertices
                yyt = np.dot(yyt - X[1][-1, ...], U.T)
                f.updateVertices(yyt)
                savePoints(self.options['outputDir'] +'/TargetCorrected.vtk', f)

            self.saveEvolution(self.fv0, self.state)
            # fvDef = self.createObject(self.x0)
            # #v = self.v[0,...]
            # displ = np.zeros(self.npt)
            # # dt = 1.0 /self.Tsize
            # v = self.options['KparDiff'].applyK(self.x0, self.state['at'][0,...])
            # for kk in range(self.Tsize+1):
            #     fvDef.updateVertices(self.state['xt'][-1, :self.nvert:, :])
            #     vf = vtkFields('POINT_DATA', self.fv0.vertices.shape[0])
            #     scalars = dict()
            #     scalars['Jacobian'] = self.state['Jt'][kk, :self.nvert]
            #     scalars['displacement'] = displ[:self.nvert]
            #     if self.Tsize > 0:
            #         displ += np.sqrt((v ** 2).sum(axis=-1))
            #     if self.match_landmarks:
            #         pp = PointSet(data=self.state['xt'][kk, self.nvert:, :])
            #         pp.saveVTK(self.outputDir + '/' + self.options['saveFile']+str(kk+self.Tsize) + '_lmk.vtk')
            #     if kk < self.Tsize:
            #         v = self.options['KparDiff'].applyK(self.state['xt'][kk,...], self.state['at'][kk,...])
            #         #v = self.v[kk,...]
            #         kkm = kk
            #     else:
            #         kkm = kk-1
            #     vectors = dict()
            #     vectors['velocity'] = v[:self.nvert, :]
            #     savePoints(self.options['outputDir'] +'/'+ self.options['saveFile']+str(kk+self.Tsize)+'.vtk', fvDef,
            #                vectors=vectors, scalars=scalars)
        else:
            obj1, self.state = self.objectiveFunDef(self.control, withTrajectory=True, display=True)
            self.fvDef.updateVertices(self.state['xt'][-1, :, :])


    # def optimizeMatching(self):
    #     #print 'dataterm', self.dataTerm(self.fvDef)
    #     #print 'obj fun', self.objectiveFun(), self.obj0
    #     grd = self.getGradient(self.gradCoeff)
    #     [grd2] = self.dotProduct(grd, [grd])
    #
    #     self.gradEps = max(0.001, np.sqrt(grd2) / 10000)
    #     logging.info('Gradient lower bound: {0:f}'.format(self.gradEps))
    #     #print 'x0:', self.x0
    #     #print 'y0:', self.y0
    #     self.cgBurnIn = self.affBurnIn
    #
    #     if self.options['algorithm'] == 'cg':
    #         cg.cg(self, verb=self.options['verb'], maxIter=self.options['maxIter'],
    #               TestGradient=self.options['testGradient'], epsInit=.01,
    #               Wolfe=self.options['Wolfe'])
    #     elif self.options['algorithm'] == 'bfgs':
    #         bfgs.bfgs(self, verb=self.options['verb'], maxIter=self.options['maxIter'],
    #                   TestGradient=self.options['testGradient'], epsInit=1.,
    #                   Wolfe=self.options['Wolfe'], lineSearch=self.options['lineSearch'], memory=50)
    #     #return self.at, self.xt
