import os
import numpy as np
from copy import deepcopy
import logging
from . import curves, curveDistances as cd
from . import pointSets
from .pointSetMatching import PointSetMatching, Control, State
from . import conjugateGradient as cg, grid, bfgs
import matplotlib.pyplot as plt
from matplotlib import cm
from functools import partial


# Main class for curve matching
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
class CurveMatching(PointSetMatching):
    def set_parameters(self):
        super().set_parameters()
        self.gradEps = -1
        self.lineSearch = "Weak_Wolfe"
        self.randomInit = False
        self.iter = 0
        self.reset = True

    def set_template_and_target(self, Template, Target, misc=None):
        if Template is None:
            logging.info('Please provide a template curve')
            return
        else:
            self.fv0 = curves.Curve(curve=Template)

        if Target is None:
            logging.info('Please provide a target curve')
            return
        else:
            self.fv1 = curves.Curve(curve=Target)

        self.fv0.saveVTK(self.outputDir + '/Template.vtk')
        self.fv1.saveVTK(self.outputDir + '/Target.vtk')
        self.npt = self.fv0.vertices.shape[0]
        self.dim = self.fv0.vertices.shape[1]

    def initialize_variables(self):
        self.x0 = self.fv0.vertices
        self.fvDef = deepcopy(self.fv0)
        self.Tsize = int(round(1.0/self.options['timeStep']))
        self.control = Control()
        self.controlTry = Control()
        self.state = State()

        self.control['at'] = np.zeros([self.Tsize, self.x0.shape[0],
                                       self.x0.shape[1]])
        self.controlTry['at'] = np.zeros([self.Tsize, self.x0.shape[0],
                                          self.x0.shape[1]])
        self.control['Afft'] = np.zeros([self.Tsize, self.affineDim])
        self.controlTry['Afft'] = np.zeros([self.Tsize, self.affineDim])
        self.state['xt'] = np.tile(self.fv0.vertices, [self.Tsize+1, 1, 1])

        self.v = np.zeros([self.Tsize+1, self.npt, self.dim])
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

    def set_fun(self, errorType, vfun=None):
        self.options['errorType'] = errorType
        if errorType == 'current':
            print('Running Current Matching')
            weight = None
            self.fun_obj0 = partial(cd.currentNorm0,
                                    KparDist=self.options['KparDist'],
                                    weight=weight)
            self.fun_obj = partial(cd.currentNormDef,
                                   KparDist=self.options['KparDist'],
                                   weight=weight)
            self.fun_objGrad = partial(cd.currentNormGradient,
                                       KparDist=self.options['KparDist'],
                                       weight=weight)
        elif errorType == 'measure':
            print('Running Measure Matching')
            self.fun_obj0 = partial(cd.measureNorm0,
                                    KparDist=self.options['KparDist'])
            self.fun_obj = partial(cd.measureNormDef,
                                   KparDist=self.options['KparDist'])
            self.fun_objGrad = partial(cd.measureNormGradient,
                                       KparDist=self.options['KparDist'])
        elif errorType == 'varifold':
            self.fun_obj0 = partial(cd.varifoldNorm0,
                                    KparDist=self.options['KparDist'],
                                    weight=1.)
            self.fun_obj = partial(cd.varifoldNormDef,
                                   KparDist=self.options['KparDist'],
                                   weight=1.)
            self.fun_objGrad = partial(cd.varifoldNormGradient,
                                       KparDist=self.options['KparDist'],
                                       weight=1.)
        elif errorType == 'varifoldComponent':
            self.fun_obj0 = partial(cd.varifoldNormComponent0,
                                    KparDist=self.options['KparDist'])
            self.fun_obj = partial(cd.varifoldNormComponentDef,
                                   KparDist=self.options['KparDist'])
            self.fun_objGrad = partial(cd.varifoldNormComponentGradient,
                                       KparDist=self.options['KparDist'])
        elif errorType == 'landmarks':
            self.fun_obj0 = cd.L2Norm0
            self.fun_obj = cd.L2NormDef
            self.fun_objGrad = cd.L2NormGradient
            if self.fv1.vertices.shape[0] != self.fvDef.vertices.shape[0]:
                sdef = self.fvDef.arclength()
                s1 = self.fv1.arclength()
                x1 = np.zeros(self.fvDef.vertices.shape)
                x1[:, 0] = np.interp(sdef, s1, self.fv1.vertices[:, 0])
                x1[:, 1] = np.interp(sdef, s1, self.fv1.vertices[:, 1])
                self.fv1 = curves.Curve(curve=(self.fvDef.faces, x1))
            bestk = 0
            minL2 = cd.L2Norm(self.fvDef, self.fv1)
            fvTry = curves.Curve(curve=self.fv1)
            for k in range(1, self.fv1.vertices.shape[0]):
                fvTry.updateVertices(np.roll(self.fv1.vertices, k, axis=0))
                L2 = cd.L2Norm(self.fvDef, fvTry)
                if L2 < minL2:
                    bestk = k
                    minL2 = L2
            if bestk > 0:
                self.fv1.updateVertices(np.roll(self.fv1.vertices,
                                                bestk, axis=0))
        else:
            print('Unknown error Type: ', self.options['errorType'])

        if self.options['internalCost'] == 'h1':
            self.extraTerm = {}
            self.extraTerm['fun'] = partial(curves.normGrad,
                                            faces=self.fv0.faces,
                                            component=self.fv0.component,
                                            weight=0.0)
            self.extraTerm['grad'] = partial(curves.diffNormGrad,
                                             faces=self.fv0.faces,
                                             component=self.fv0.component,
                                             weight=0.0)
            self.extraTerm['coeff'] = self.options['internalWeight']
        elif self.options['internalCost'] == 'h1Alpha':
            self.extraTerm = {}
            self.extraTerm['fun'] = partial(curves.h1AlphaNorm,
                                            faces=self.fv0.faces,
                                            component=self.fv0.component)
            self.extraTerm['grad'] = partial(curves.diffH1Alpha,
                                             faces=self.fv0.faces,
                                             component=self.fv0.component)
            self.extraTerm['coeff'] = self.options['internalWeight']
        elif self.options['internalCost'] == 'h1AlphaInvariant':
            self.extraTerm = {}
            self.extraTerm['fun'] = partial(curves.h1AlphaNormInvariant,
                                            faces=self.fv0.faces,
                                            component=self.fv0.component)
            self.extraTerm['grad'] = partial(curves.diffH1AlphaInvariant,
                                             faces=self.fv0.faces,
                                             component=self.fv0.component)
            self.extraTerm['coeff'] = self.options['internalWeight']
        elif self.options['internalCost'] == 'h1Invariant':
            self.extraTerm = {}
            self.extraTerm['coeff'] = self.options['internalWeight']
            if self.fv0.vertices.shape[1] == 2:
                self.extraTerm['fun'] = partial(curves.normGradInvariant,
                                                faces=self.fv0.faces,
                                                component=self.fv0.component)
                self.extraTerm['grad'] = partial(curves.diffNormGradInvariant,
                                                 faces=self.fv0.faces,
                                                 component=self.fv0.component)
            else:
                self.extraTerm['fun'] = partial(curves.normGradInvariant3D,
                                                faces=self.fv0.faces,
                                                component=self.fv0.component)
                self.extraTerm['grad'] = partial(curves.diffNormGradInvariant3D,
                                                 faces=self.fv0.faces,
                                                 component=self.fv0.component)
        else:
            if self.options['internalCost'] is not None:
                logging.warning("Internal cost not recognized: "
                                + self.options['internalCost'])
            self.extraTerm = None

    def initial_plot(self):
        self.cmap = cm.get_cmap('hsv', self.fvDef.faces.shape[0])
        self.cmap1 = cm.get_cmap('hsv', self.fv1.faces.shape[0])
        self.lw = 3
        if self.dim == 2:
            fig = plt.figure(2)
            fig.clf()
            ax = fig.gca()
            for kf in range(self.fvDef.faces.shape[0]):
                ax.plot(self.fvDef.vertices[self.fvDef.faces[kf, :], 0],
                        self.fvDef.vertices[self.fvDef.faces[kf, :], 1],
                        color='r', linewidth=self.lw)
            for kf in range(self.fv1.faces.shape[0]):
                ax.plot(self.fv1.vertices[self.fv1.faces[kf, :], 0],
                        self.fv1.vertices[self.fv1.faces[kf, :], 1],
                        color=[0, 0, 1], linewidth=self.lw)
            plt.axis('equal')
            plt.axis('off')
            self.axis = plt.axis()
            plt.savefig(self.outputDir + '/Template_Target.png')
            fig = plt.figure(3)
            fig.clf()
            ax = fig.gca()
            for kf in range(self.fvDef.faces.shape[0]):
                ax.plot(self.fvDef.vertices[self.fvDef.faces[kf, :], 0],
                        self.fvDef.vertices[self.fvDef.faces[kf, :], 1],
                        color=self.cmap(kf), linewidth=self.lw)
            plt.axis('equal')
        elif self.dim == 3:
            fig = plt.figure(2)
            fig.clf()
            ax = fig.gca(projection='3d')
            for kf in range(self.fv1.faces.shape[0]):
                ax.plot(self.fv1.vertices[self.fv1.faces[kf, :], 0],
                        self.fv1.vertices[self.fv1.faces[kf, :], 1],
                        self.fv1.vertices[self.fv1.faces[kf, :], 2],
                        color=[0, 0, 1])
            fig = plt.figure(3)
            fig.clf()
            ax = fig.gca(projection='3d')
            for kf in range(self.fvDef.faces.shape[0]):
                ax.plot(self.fvDef.vertices[self.fvDef.faces[kf, :], 0],
                        self.fvDef.vertices[self.fvDef.faces[kf, :], 1],
                        self.fvDef.vertices[self.fvDef.faces[kf, :], 2],
                        color=[1, 0, 0], marker='*')
            # plt.axis('equal')
        plt.pause(0.1)

    def dataTerm(self, _fvDef, var=None):
        obj = self.fun_obj(_fvDef, self.fv1) / (self.options['sigmaError']**2)
        return obj


    def _objectiveFun(self, control, withTrajectory=False):
        (obj, st) = self.objectiveFunDef(control, withTrajectory=True)
        self.fvDef.updateVertices(st['xt'][-1, :, :])
        obj0 = self.dataTerm(self.fvDef)

        if withTrajectory:
            return obj+obj0, st
        else:
            return obj+obj0

    def objectiveFun(self):
        if self.obj is None:
            self.obj0 \
                = self.fun_obj0(self.fv1) / (self.options['sigmaError']**2)
            (self.obj, self.state) \
                = self.objectiveFunDef(self.control, withTrajectory=True)
            self.fvDef.updateVertices(np.squeeze(self.state['xt'][-1, :, :]))
            self.obj += self.obj0 + self.dataTerm(self.fvDef)
        return self.obj

    def getVariable(self):
        return self.control

    def makeTryInstance(self, state):
        ff = curves.Curve(curve=self.fvDef)
        ff.updateVertices(state['xt'][-1, :, :])
        return ff

    def endPointGradient(self, endPoint=None):
        if endPoint is None:
            endPoint = self.fvDef
        px = self.fun_objGrad(endPoint, self.fv1)
        return px / self.options['sigmaError']**2

    def testEndpointGradient(self):
        ff = curves.Curve(curve=self.fvDef)
        dff = np.random.normal(size=ff.vertices.shape)
        eps = 1e-6
        ff.updateVertices(ff.vertices+eps*dff)
        c1 = self.dataTerm(ff)
        ff.updateVertices(ff.vertices-2*eps*dff)
        c2 = self.dataTerm(ff)
        grd = self.endPointGradient()
        logging.info("test endpoint gradient: {0:.5f} {1:.5f}".format((c1-c2)/(2*eps), (grd*dff).sum()) )

    def setUpdate(self, update):
        control = Control()
        for k in update[0].keys():
            if update[0][k] is not None:
                control[k] = self.control[k] - update[1] * update[0][k]
        st = self.solveStateEquation(control)
        xt = st['xt']
        endPoint = curves.Curve(curve=self.fv0)
        endPoint.updateVertices(xt[-1, :, :])

        return control, st, endPoint

    def endOfIteration(self, endP=False):
        (obj1, self.state) = self.objectiveFunDef(self.control,
                                                  withTrajectory=True,
                                                  withJacobian=True)
        self.iter += 1
        if self.options['testGradient']:
            self.testEndpointGradient()

        if self.extraTerm is not None and self.options['testGradient']:
            Phi = np.random.normal(size=self.x0.shape)
            dPhi1 = np.random.normal(size=self.x0.shape)
            dPhi2 = np.random.normal(size=self.x0.shape)
            eps = 1e-10
            fv22 = curves.Curve(curve=self.fvDef)
            fv22.updateVertices(self.fvDef.vertices+eps*dPhi2)
            e11 = self.extraTerm['fun'](self.fvDef.vertices, Phi-eps*dPhi1)
            e22 = self.extraTerm['fun'](self.fvDef.vertices-eps*dPhi2, Phi)
            e1 = self.extraTerm['fun'](self.fvDef.vertices, Phi+eps*dPhi1)
            e2 = self.extraTerm['fun'](self.fvDef.vertices+eps*dPhi2, Phi)
            # e2 = self.internalCost(fv22, Phi)
            grad = self.extraTerm['grad'](self.fvDef.vertices, Phi)
            logging.info(f"Laplacian: {(e1-e11)/(2*eps):.5f} {(grad['phi']*dPhi1).sum():.5f};  "+
                         f"Gradient: {(e2-e22)/(2*eps):.5f} {(grad['x']*dPhi2).sum():.5f}\n")

        if self.saveRate > 0 and self.iter % self.saveRate == 0:
            if self.dim == 2:
                st = self.solveStateEquation(init_state=self.fv0.vertices,
                                             options={'withPointSet': self.gridxy})
                xt = st['xt']
                yt = st['yt']
            if self.options['saveTrajectories']:
                pointSets.saveTrajectories(self.outputDir + '/'
                                           + self.options['saveFile']
                                           + 'curves.vtk',
                                           self.state['xt'])

            for kk in range(self.Tsize+1):
                self.fvDef.updateVertices(self.state['xt'][kk, :, :])
                if self.pplot and self.dim == 2:
                    fig = plt.figure(4)
                    fig.clf()
                    ax = fig.gca()
                    for kf in range(self.fvDef.faces.shape[0]):
                        ax.plot(self.fvDef.vertices[self.fvDef.faces[kf, :], 0],
                                self.fvDef.vertices[self.fvDef.faces[kf, :], 1],
                                color=self.cmap(kf), linewidth=self.lw)
                    plt.axis('off')
                    plt.axis(self.axis)
                    plt.savefig(self.outputDir + '/' 
                                + self.options['saveFile']+str(kk)+'.png')
                    fig.canvas.flush_events()

                self.fvDef.saveVTK(self.outputDir + '/'
                                   + self.options['saveFile']+str(kk)+'.vtk')
                if self.dim == 2:
                    self.gridDef.vertices = np.copy(yt[kk, :, :])
                    self.gridDef.saveVTK(self.outputDir+'/grid'+str(kk)+'.vtk')
        else:
            self.fvDef.updateVertices(self.state['xt'][self.Tsize, :, :])

        if self.pplot:
            if self.dim == 2:
                if self.saveRate == 0 or self.iter % self.saveRate != 0:
                    fig = plt.figure(4)
                    fig.clf()
                    ax = fig.gca()
                    if self.options['errorType'] == 'landmarks':
                        for kv in range(self.fvDef.vertices.shape[0]):
                            ax.plot((self.fvDef.vertices[kv, 0],
                                     self.fv1.vertices[kv, 0]),
                                    (self.fvDef.vertices[kv, 1],
                                     self.fv1.vertices[kv, 1]),
                                    color=[0.7, 0.7, 0.7], linewidth=1)
                    for kf in range(self.fv1.faces.shape[0]):
                        ax.plot(self.fv1.vertices[self.fv1.faces[kf, :], 0],
                                self.fv1.vertices[self.fv1.faces[kf, :], 1],
                                color=self.cmap1(kf))
                    for kf in range(self.fvDef.faces.shape[0]):
                        ax.plot(self.fvDef.vertices[self.fvDef.faces[kf, :], 0],
                                self.fvDef.vertices[self.fvDef.faces[kf, :], 1],
                                color=self.cmap(kf), linewidth=self.lw)
                    plt.axis('equal')
            elif self.dim == 3:
                fig = plt.figure(4)
                fig.clf()
                ax = fig.gca(projection='3d')
                for kf in range(self.fv1.faces.shape[0]):
                    ax.plot(self.fv1.vertices[self.fv1.faces[kf, :], 0],
                            self.fv1.vertices[self.fv1.faces[kf, :], 1],
                            self.fv1.vertices[self.fv1.faces[kf, :], 2],
                            color=[0, 0, 1])
                for kf in range(self.fvDef.faces.shape[0]):
                    ax.plot(self.fvDef.vertices[self.fvDef.faces[kf, :], 0],
                            self.fvDef.vertices[self.fvDef.faces[kf, :], 1],
                            self.fvDef.vertices[self.fv1.faces[kf, :], 2],
                            color=[1, 0, 0], marker='*')
            fig.canvas.flush_events()

    def endOptim(self):
        if self.saveRate == 0 or self.iter % self.saveRate > 0:
            if self.dim == 2:
                st = self.solveStateEquation(init_state=self.fv0.vertices,
                                             options={'withPointSet': self.gridxy})
                yt = st['yt']
                for kk in range(self.Tsize+1):
                    self.gridDef.vertices = np.copy(yt[kk, :, :])
                    self.gridDef.saveVTK(self.outputDir + '/grid' + str(kk) + '.vtk')
            for kk in range(self.Tsize+1):
                self.fvDef.updateVertices(self.state['xt'][kk, :, :])
                self.fvDef.saveVTK(self.outputDir +'/'+ self.options['saveFile']+str(kk)+'.vtk')
        self.defCost = self.obj - self.obj0 - self.dataTerm(self.fvDef)

    def optimizeMatching(self):
        print('obj0', self.fun_obj0(self.fv1))
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = max(self.options['gradLB'], np.sqrt(grd2) / 10000)
        print('Gradient bound:', self.gradEps)
        kk = 0
        while os.path.isfile(self.outputDir +'/'+ self.options['saveFile']+str(kk)+'.vtk'):
            os.remove(self.outputDir +'/'+ self.options['saveFile']+str(kk)+'.vtk')
            kk += 1

        if self.options['algorithm'] == 'cg':
            cg.cg(self, verb=self.options['verb'],
                  maxIter=self.options['maxIter'],
                  TestGradient=self.options['testGradient'], epsInit=.01,
                  Wolfe=self.options['Wolfe'])
        elif self.options['algorithm'] == 'bfgs':
            bfgs.bfgs(self, verb=self.options['verb'],
                      maxIter=self.options['maxIter'],
                      TestGradient=self.options['testGradient'], epsInit=1.,
                      Wolfe=self.options['Wolfe'],
                      lineSearch=self.options['lineSearch'], memory=50)
