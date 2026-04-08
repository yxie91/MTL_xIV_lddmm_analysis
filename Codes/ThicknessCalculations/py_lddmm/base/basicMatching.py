import os
import glob
from copy import deepcopy
import numpy as np
import keopscore
import logging


class Control(dict):
    def __init__(self):
        super().__init__()


class State(dict):
    def __init__(self):
        super().__init__()


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
#        testGradient: evaluates gradient accuracy over random Control (debug)
#        outputDir: where results are saved
#        saveFile: generic name for saved surfaces
#        affine: 'affine', 'similitude', 'euclidean', 'translation' or 'none'
#        maxIter: max iterations in conjugate gradient
class BasicMatching(object):
    def __init__(self, Template=None, Target=None, options = None):
        self.setInitialOptions(options)

        if self.options['algorithm'] == 'cg':
             self.euclideanGradient = False
        else:
            self.euclideanGradient = True

        self.maxT = self.options['maxT']
        self.Tsize = self.options['Tsize']
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
        self.set_template_and_target(Template, Target, misc=self.options)
        self.burnIn = self.options['burnIn']
        self.gpuAvailable = keopscore.config.config.use_cuda

        self.reset = False
        self.Kdiff_dtype = self.options['pk_dtype']
        self.Kdist_dtype = self.options['pk_dtype']
        self.unreduced = self.options['unreduced']

            #print np.fabs(self.fv1.surfel-self.fv0.surfel).max()
        self.set_parameters()
        self.set_fun(self.options['errorType'], vfun=self.options['vfun'])
        self.setDotProduct(self.options['unreduced'])

        self.initialize_variables()
        self.gradCoeff = 1.
        self.set_passenger(self.options['passenger'])
        self.pplot = self.options['pplot']
        # if self.pplot:
        #     self.initial_plot()
        self.initialSave()


    def getDefaultOptions(self):
        options = {
            'timeStep': 0.1,
            'maxT': 1.0,
            'algorithm': 'bfgs',
            'unreduced': False,
            'Wolfe': True,
            'burnIn': 10,
            'epsInit': 1.,
            'KparDiff': None,
            'KparDist': None,
            'pk_dtype': 'float32',
            'sigmaError': 1.0,
            'errorType': 'measure',
            'vfun': None,
            'maxIter': 1000,
            'regWeight': 1.0,
            'unreducedWeight': 0.0,
            'affineKernel': False,
            'affineWeight': 10.0,
            'rotWeight': None,
            'randomInit': False,
            'scaleWeight': None,
            'transWeight': None,
            'testGradient': False,
            'subsampleTargetSize': -1,
            'affineOnly': False,
            'passenger': None,
            'Landmarks': None,
            'saveFile': 'evolution',
            'saveRate': 10,
            'mode': 'normal',
            'verb': True,
            'pplot': False,
            'saveTrajectories': False,
            'affine': None,
            'outputDir': '.',
            'symmetric': False,
            'internalCost': None,
            'internalWeight': 1.0,
            'lineSearch': 'Weak_Wolfe',
            'bfgs_memory': 50,
            'gradTol': 10.0,
            'cgRestartRate': 100,
            'nresetMax': 10,
            'gradLB': 0.0001,
            'gradLBCoeff': 0.0001
        }
        return options

    def setInitialOptions(self, options):
        self.options = self.getDefaultOptions()
        if options is not None:
            for k in options.keys():
                self.options[k] = options[k]
        if 'Tsize' in options:
            self.options['timeStep'] = 1/options['Tsize']
        else:
            self.options['Tsize'] =  int(round(1/self.options['timeStep']))


    def initialize_variables(self):
        pass

    def setDotProduct(self, unreduced=False):
        pass
    
    def set_passenger(self, passenger):
        pass

    def set_landmarks(self, landmarks):
        pass

    def initial_plot(self):
        pass

    def initialSave(self):
        pass

    def set_template_and_target(self, Template, Target, misc=None):
        pass

    def set_fun(self, errorType, vfun=None):
        pass

    def set_parameters(self):
        if self.options['mode'] == 'debug':
            self.options['verb'] = True
            self.options['testGradient'] = True
            self.options['pk_dtype'] = 'float64'
        elif self.options['mode'] == 'quiet':
            self.options['verb'] = False
            self.options['testGradient'] = False
        else:
            self.options['testGradient'] = False
            self.options['verb'] = True

        self.saveRate = self.options['saveRate']
        # self.gradEps = -1
        self.randomInit = self.options['randomInit']
        self.iter = 0
        self.nreset = 0
        self.restartRate = self.options['cgRestartRate']
        self.lineSearch = self.options['lineSearch']

        self.obj = None
        self.objTry = None
        # self.saveFile = saveFile
        self.coeffAff1 = 1.
        if self.options['algorithm'] == 'cg':
            self.coeffAff2 = 100.
        else:
            self.coeffAff2 = 1.
        self.coeffAff = self.coeffAff1
        self.affBurnIn = 25
        self.forceLineSearch = False
        self.varCounter = 0
        self.trajCounter = 0
        self.smoothKernel = None



    def setOutputDir(self, outputDir, clean=True):
        self.outputDir = outputDir
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                logging.error('Cannot save in ' + outputDir)
                return
            else:
                os.makedirs(outputDir)

        logging.info('Saving results in ' + outputDir)
        if clean:
            #logging.info(outputDir + '/*.vtk')
            fileList = glob.glob(outputDir + '/*.vtk')
            #logging.info(fileList)
            for f in fileList:
                os.remove(f)


    def dataTerm(self, _fvDef, var = None):
        pass
    

    def makeTryInstance(self, pts):
        pass
    
    def objectiveFun(self):
        pass

    def getVariable(self):
        pass
    
    def updateTry(self, dr, eps, objRef=None):
        pass

    def getGradient(self, coeff=1.0, update=None):
        pass
    
    def startOfIteration(self):
        pass

    def addProd(self, dir1, dir2, beta):
        # dr = Control()
        dr = dict()
        for k in dir1.keys():
            if dir1[k] is not None and dir2[k] is not None:
                dr[k] = dir1[k] + beta * dir2[k]
            else:
                dr[k] = None
        return dr

    def prod(self, dir1, beta):
        dr = dict() #Control()
        for k in dir1.keys():
            if dir1[k] is not None:
                dr[k] = beta * dir1[k]
            else:
                dr[k] = None
        return dr


    def copyDir(self, dir0):
        return deepcopy(dir0)


    def randomDir(self):
        return None

    def testEndpointGradient(self):
        pass

    def dotProduct(self, g1, g2):
        pass

    def testGradientFun(self, obj, grd, gradCoeff, opt=None, dotProduct=None):
        self.testEndpointGradient()
        dirfoo =self.randomDir()
        if dirfoo is not None:
            epsfoo = 1e-8
            objfoo1 = self.updateTry(dirfoo, epsfoo, obj - 1e10)
            [grdfoo] = self.dotProduct(grd, [dirfoo])
            objfoo2 = self.updateTry(dirfoo, -epsfoo, obj - 1e10)
            logging.info('Test Gradient: %.6f %.6f' % ((objfoo1 - objfoo2) / (2.0 * epsfoo), -grdfoo * gradCoeff))


    def dotProduct_Riemannian(self, g1, g2):
        pass

    def dotProduct_euclidean(self, g1, g2):
        pass

    def acceptVarTry(self):
        pass

    def endOfIteration(self, endP=False):
        pass

    def optimizeMatching(self):
        pass



