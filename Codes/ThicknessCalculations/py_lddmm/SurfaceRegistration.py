#!/usr/bin/env python
# -*- coding: utf-8 -*-
# CIS cluster header
#$ -S /usr/local/python/intelpython27/bin/python
#$ -cwd
#$ -j y
#$ -pe orte 8
#$ -v OMP_NUM_THREADS=8
#$ -o log.txt

import sys, os, time
sys.path.append('ThicknessCalculations/py_lddmm/')
print(sys.path)
print('1\n')
import multiprocessing as mp
from multiprocessing import Pool
import ntpath
from numba import jit, prange, int64, cuda

import torch
import numpy as np
import numpy.matlib

import os, psutil
import matplotlib
from matplotlib import pyplot as plt
if 'DISPLAY' in os.environ:
    matplotlib.use('qt5Agg')
else:
    matplotlib.use("Agg")
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import imageio.v3 as iio

from pykeops.torch import LazyTensor
#import pykeops.config

np_dtype = "float32"
dtype = torch.cuda.FloatTensor 

import pykeops
#import socket
#pykeops.set_build_folder("~/.cache/keops"+pykeops.__version__ + "_" + (socket.gethostname()))


torch.cuda.empty_cache()
cuda.current_context().deallocations.clear()
import GPUtil
GPUtil.showUtilization()

from base.surfaceMatchingNormalExtremities import *

print('3\n')

def main():
    outDirPath = ""
    flipValue = 1
    sigmaDist = 5
    subDivide = 0
    
    print('2\n')

    try:
        if len(sys.argv) == 1:
            templatePath = os.path.abspath(os.environ['templatePath'])
            targetPath = os.path.abspath(os.environ['targetPath'])

            if 'outDirPath' in os.environ.keys():
                outDirPath = os.path.abspath(os.environ['outDirPath'])
                
            if 'flipValue' in os.environ.keys():
                flipValue = int(os.environ['flipValue'])

            if 'sigmaDist' in os.environ.keys():
                sigmaDist = float(os.environ['sigmaDist'])

            if 'subDivide' in os.environ.keys():
                subDivide = int(os.environ['subDivide'])

        elif len(sys.argv) > 3:
            templatePath = os.path.abspath(sys.argv[1])
            targetPath = os.path.abspath(sys.argv[2])
            outDirPath = os.path.abspath(sys.argv[3])
            if len(sys.argv) > 4: flipValue = int(sys.argv[4])
            if len(sys.argv) > 5: sigmaDist = float(sys.argv[5])
            if len(sys.argv) > 6: subDivide = int(sys.argv[6])
        else:
            raise Exception()
                
    except:
        printUsage()


    runSurfaceMatching(templatePath, targetPath, outDirPath=outDirPath, flip=flipValue, sigmaDist=sigmaDist, subDivide=subDivide)
    print('3\n')

def printUsage():
    print("Usage: ")
    print("\t" + sys.argv[0] + " templatePath targetPath outDirPath [flipValue] [sigmaDist] [subDivide]")
    print("\n\t Where [flipValue] is an optional parameter that is 0 if template normals should be flipped and 1 otherwise.")
    sys.exit(1)    

def runSurfaceMatching(templatePath, targetPath, outDirPath="", flip=False, sigmaDist=5, subDivide=False):
    if outDirPath == "":
        outDirPath = os.path.splitext(templatePath)[0]+"/"
    if not os.path.exists(outDirPath): os.makedirs(outDirPath)

    fvTarget = surfaces.Surface(targetPath)
    fvTemplate = surfaces.Surface(templatePath)

    fvTarget.removeIsolated()
    fvTarget.edgeRecover()
    if subDivide:
        fvTarget.subDivide(1)
        fvTemplate.subDivide(1)
        
    fvTemplate.removeIsolated()
    fvTemplate.edgeRecover()
    if flip: fvTemplate.flipFaces()
    logFileName = "log.txt"
    loggingUtils.setup_default_logging(outDirPath, fileName=logFileName, stdOutput=True)
    K1 = kfun.Kernel(name='laplacian', sigma=0.5,
                     order=3)
    K2 = kfun.Kernel(name='gaussian',sigma=sigmaDist)
    timeStep = 0.05
    sm = surfaceMatching.SurfaceMatchingParam(timeStep=timeStep, KparDiff=K1, KparDist=K2,
                                              sigmaError=.1, errorType='varifold', internalCost='h1')

            
    f = SurfaceMatching(Template=fvTemplate, Target=fvTarget, outputDir=outDirPath, param=sm, regWeight=1.,
                        saveTrajectories=True, symmetric=False, pplot=False,
                        affine='none', internalWeight=100., affineWeight=1e3, maxIter_cg=50,
                        maxIter_al=50, mu=1e-3)
    f.saveRate = 5
    startTime = time.time()

    f.optimizeMatching()

    runTime = time.time() - startTime
    logPath = outDirPath+"/"+logFileName
    logFile = open(logPath,"a")
    logFile.write("Run Time = {0}m = {1}s\n".format(runTime/60, runTime))
    logFile.close()

if __name__== "__main__": main()
