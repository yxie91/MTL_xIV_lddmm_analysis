import os
import pdb
from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
import numpy as np
import logging
import h5py
import matplotlib
if 'DISPLAY' in os.environ:
    matplotlib.use('qt5Agg')
else:
    matplotlib.use("Agg")
from matplotlib import pyplot as plt, cm
from scipy.optimize import nnls
from base import pointSets
from base import loggingUtils
from base.pointSets import PointSet
from base.kernelFunctions import Kernel
from base.pointSetExamples import TwoDiscs, Rbf
from base.pointSetExamples import Circle, Flower, ShrinkCircle, BumpyEllipse, SimplifiedHuman, Fluc
from base.affineRegistration import rigidRegistration
from base.mspointSetMatching import MultiscalePointSetMatching
from base.mspointEvolution import mslandmarkPassengerEvolutionEuler, mslandmarkPassengerEvolutionReverse
from base.vtk_fields import vtkFields

def getInput(problem):
    input = {}
    input['base_scale'] = [19]
    # input['base_scale'] = [19]
    # input['base_scale'] = [0]
    input['sigmaError'] = 0.01
    input['fv0'] = []
    input['fv1'] = []
    nbase_scale = len(input['base_scale'])
    if problem == 'flower':
        # same target
        for ii in range(len(input['base_scale'])):
            input['fv0'].append(Circle(r=3., N=100))
            input['fv1'].append(Flower(N=100, rsmall=.1))
    elif problem == 'flowers':
        input['fv0'].append(Flower(r=2., N=100, rsmall=1))
        input['fv0'].append(Circle(r=3., N=100))
        input['fv1'].append(Flower(N=100, rsmall=.1, shift=.25*np.pi))
        input['fv1'].append(Flower(N=100, rsmall=.1))

    elif problem == 'rotate1':
        for ii in range(len(input['base_scale'])):
            input['fv0'].append(Circle(r=2., N=10))
            input['fv1'].append(Circle(r=2., N=10, shift=2 * np.pi / 10))
    elif problem == 'rotate2':
        input['fv0'].append(Circle(r=1.5, N=100))
        input['fv0'].append(Circle(r=3.5, N=100))
        input['fv1'].append(Circle(r=2, N=100, shift=2*np.pi/10))
        input['fv1'].append(Circle(r=3, N=100, shift=-2*np.pi/10))
    elif problem == 'complicated_rotate':
        for ii in range(len(input['base_scale'])):
            input['fv0'].append(Circle(r=2., N=100))
            input['fv1'].append(ShrinkCircle(r1=1., r2=4., N=100, shift=8 * np.pi / 10, offset= (1.0, 1.0)))
    elif problem=='combined':
        for ii in range(len(input['base_scale'])):
            input['fv0'].append(Circle(r=2., N=100))
            input['fv1'].append(BumpyEllipse(a=4., N=100))
    elif problem =='Fluc':
        for ii in range(len(input['base_scale'])):
            input['fv0'].append(Circle(r=2., N=100))
            input['fv1'].append(Fluc(a=4., N=100))
    elif problem == 'flower_circle':
        input['fv0'].append(Circle(r=2., N=100))
        input['fv0'].append(Circle(r=2., N=100))
        input['fv1'].append(Flower(r=2., N=100, rsmall=.1))
        input['fv1'].append(Circle(r=3., N=100))
    elif problem == 'flower_rotate':
        input['fv0'].append(Circle(r=2., N=100))
        input['fv0'].append(Circle(r=2., N=100))
        input['fv1'].append(Flower(r=2., N=100, rsmall=.1, shift=np.pi/4))
        input['fv1'].append(Flower(r=2., N=100, rsmall=.1))
    elif problem == 'human':
        for ii in range(len(input['base_scale'])):
            input['fv0'].append(SimplifiedHuman(N_head=15, N_arm=24, N_body=28, N_bottom=6))
            input['fv1'].append(SimplifiedHuman(N_head=15, N_arm=24, N_body=28, N_bottom=6,
                                                armAngles=[np.pi/4, np.pi/4], heada=.3, headb=.6))

        #
        # input['fv0'] = Circle(r=2., N=100)
        # input['fv1'] = BumpyEllipse(a=4., N=100)
    else:
        logging.error(f'Unknown problem: {problem}')
        
    return input



logging.info("Multi-scale landmark matching.\nThis work is part of Oscar Liu's Ph.D. dissertation")

#problem = 'complicated_rotate'
problem = 'flower'
# problem = 'combined'
#probelm = 'flowers'
# problem = 'Fluc'
#problem = 'rotate1'
#problem = 'rotate2'
#problem = 'flower_circle'
#problem = 'flower_rotate'
# problem = 'human'
use_positive = False

input = getInput(problem)
base_scale = input['base_scale']
try:
    fin = h5py.File('./MSKernel.h5', 'r')
except:
    logging.error('File MSKernel.h5 not found. Run make_mskernel.py to generate it')
freq = np.array(fin['frequencies'])
scales_precise = np.array(fin['scales'])
scales = np.array(range(len(scales_precise)))
widths = np.array(fin['widths'])
lbd = np.array(fin['scales'])
kappa = np.array(fin['kappa'])
# kappapv = np.array(fin['positive_values'])
nscales = len(scales)
nbs = len(base_scale)# number of base scales

# if use_positive:
#     kappa = kappapc
# else:
#     kappa = kappapv
fin.close()

sigma = 1

t = np.linspace(-5, 5, 201)
Kpc = np.zeros((nbs, kappa.shape[0], t.shape[0]))
Kpv = np.zeros((nbs, kappa.shape[0], t.shape[0]))
dKpc = np.zeros((nbs, kappa.shape[0], t.shape[0]))
dKpv = np.zeros((nbs, kappa.shape[0], t.shape[0]))
K = np.zeros((nbs, kappa.shape[0], t.shape[0]))
kappa_base = np.zeros((nbs, nbs, kappa.shape[2]))
for bs in range(nbs):
    for ts in range(nbs):
        kappa_base[bs, ts] = kappa[base_scale[bs], base_scale[ts], :]

loggingUtils.setup_default_logging('../Output', stdOutput = True)

# multiscale kernel (for optimization)
Kms = []
ksigma = []
kweight = []
for bs in range(nbs):
    sigma2append = []
    weight2append = []
    Kms2append = []
    for ts in range(nbs):
        sigma2append.append([])
        weight2append.append([])
        Kms2append.append([])
    Kms.append(Kms2append)
    ksigma.append(sigma2append)
    kweight.append(weight2append)

for bs in range(nbs):
    for ts in range(nbs):
        for j in range(widths.shape[0]):
            if np.abs(kappa_base)[bs, ts, j] > 1e-6:
                ksigma[bs][ts].append(widths[j])
                kweight[bs][ts].append(kappa_base[bs, ts, j])
# pdb.set_trace()

## Object kernel
for ii in range(nbs):
    for jj in range(nbs):
        # what are ksigma and kweight?
        Kms[ii][jj] = Kernel(name='gauss', sigma=ksigma[ii][jj], weight=kweight[ii][jj])

# at the end create full kernel

logging.info(f'Number of selected scales: {len(ksigma)}')

# # sm = PointSetMatchingParam(timeStep=0.1, )
# sm.KparDiff.pk_dtype = 'float64'
# sm.KparDist.pk_dtype = 'float64'
if use_positive:
    ktype = 'pos_coeff'
else:
    ktype = 'pos_values'
if len(base_scale) > 1:
    outputDir = f'../Output/Multi_Scale/{problem}_{base_scale}_{ktype}'
else:
    outputDir = f'../Output/Single_Scale/{problem}_{base_scale}_{ktype}'
options = {
    'outputDir': outputDir,
    'timeStep': 1/20,
    'base_scale': base_scale,
    'nbase_scales': len(base_scale),
    'scales': scales_precise,
    'mode': 'normal',
    # 'mode': 'debug',
    'maxIter': 30000,
    'burnIn': 10,
    'epsInit': .1,
    'randomInit': False,
    # 'randomInit': True,
    'affine': 'none',
    'rotWeight': 100,
    'transWeight': 100.,
    'scaleWeight': 10.,
    'affineWeight': 100.,
    'KparDiff': Kms,
    'sigmaError': input['sigmaError'],
    'errorType': 'L2',
    'unreducedResetRate': 50,
    'unreduced': False,
    'match_landmarks': False,
    'algorithm': 'bfgs',
    'unreducedWeight': 0.0,
    'saveRate': 500,
    'gradLBCoeff': 1e-5,
    'saveTrajectories': True,
    'pk_dtype': 'float64',
    'kappa_base': kappa_base
}

f = MultiscalePointSetMatching(Template=input['fv0'], Target=input['fv1'], options=options)

f.optimizeMatching()

# Compute the full kernels
Kms_ = []
ksigma_ = []
kweight_ = []

for bs in range(nscales):
    Kms_2append = []
    ksigma_2append = []
    kweight_2append = []
    for ts in range(nscales):
        Kms_2append.append([])
        ksigma_2append.append([])
        kweight_2append.append([])
    Kms_.append(Kms_2append)
    ksigma_.append(ksigma_2append)
    kweight_.append(kweight_2append)
kappa_base_ = np.zeros((nscales, nscales, kappa.shape[2]))
for bs in range(nscales):
    for ts in range(nscales):
        kappa_base_[bs, ts] = kappa[bs, ts, :]
for bs in range(nscales):
    for ts in range(nscales):
        for j in range(widths.shape[0]):
            if np.abs(kappa_base_)[bs, ts, j] > 1e-6:
                ksigma_[bs][ts].append(widths[j])
                kweight_[bs][ts].append(kappa_base_[bs, ts, j])
for ii in range(nscales):
    for jj in range(nscales):
        Kms_[ii][jj] = Kernel(name='gauss', sigma=ksigma_[ii][jj], weight=kweight_[ii][jj])

logging.info('Saving original grids')
for ii in range(nscales):
    f.gridDef[ii].vertices = np.copy(f.gridxy[0])
    f.gridDef[ii].saveVTK(outputDir + f'/original_grid_{ii}.vtk')

logging.info('Saving multiscale grids')
oldDef = np.zeros(f.gridxy[0].shape)
oldJ = np.zeros(f.gridxy[0].shape[0])
psnger = mslandmarkPassengerEvolutionEuler(f.gridxy[0], f.state['xt'], f.control['at'], Kms_, scales, base_scales=base_scale,
                                        options={'withJacobian':True}, T=f.maxT)
for ii in range(nscales):
    f.gridDef[ii].vertices = np.copy(psnger['yt'][ii][-1]) # diffeomorphism at all scales (ii here).
    f.gridDef[ii].saveVTK(outputDir + f'/grid_scale_{ii}.vtk',
                      vtkFields=vtkFields('POINT_DATA', f.gridDef[ii].vertices.shape[0],
                                          scalars = {'logJacobian':psnger['Jyt'][ii][-1, :, 0]}))

logging.info('Saving landmarks')
lmk_psnger = []
for ii in range(nbs):
    lmk_psnger.append(mslandmarkPassengerEvolutionEuler(f.fv0[ii].vertices, f.state['xt'], f.control['at'], Kms_, scales, base_scales=base_scale,
                                         options={'withJacobian': True}, T=f.maxT))
lmk_set = []
for ii in range(nbs):
    lmkset2append = []
    for jj in range(nscales):
        lmkset2append.append(PointSet(data=lmk_psnger[ii]['yt'][jj][-1, :, :]))
        lmkset2append[jj].save(outputDir + f'/landmarkSet_{ii}_Scale_{jj}.vtk')
    lmk_set.append(lmkset2append)

logging.info('Saving multiscale grid residuals')
resinv = mslandmarkPassengerEvolutionReverse(f.gridxy[0], f.state['xt'], f.control['at'],
                                      Kms_, scales=scales, base_scales=base_scale,
                                      options={'withJacobian': True}, T=f.maxT)
restest = []
resinv_last = []
for ii in range(nscales):
    resinv_last.append(resinv['yt'][ii][-1])
restest = mslandmarkPassengerEvolutionEuler(resinv_last, f.state['xt'], f.control['at'],
                                                Kms_, scales, base_scales=base_scale, options={'withJacobian': True}, T=f.maxT)
for ii in range(nscales):
    # for kk in range()
    f.gridDef[ii].vertices = np.copy(restest['yt'][ii][-1, :, :])
    f.gridDef[ii].saveVTK(outputDir + f'/grid_residual_test_scale_{ii}.vtk')

res = mslandmarkPassengerEvolutionEuler(resinv_last, f.state['xt'], f.control['at'],
                                   Kms_, scales, base_scales=base_scale, options={'withJacobian':True}, T=f.maxT)

for ii in range(nscales):
    res['Jyt'][ii] += resinv['Jyt'][ii]
    f.gridDef[ii].vertices = np.copy(res['yt'][ii][-1, :, :])
    f.gridDef[ii].saveVTK(outputDir + f'/grid_residual_scale_{ii}.vtk',
                      vtkFields=vtkFields('POINT_DATA', f.gridDef[ii].vertices.shape[0],
                                          scalars = {'Jacobian':res['Jyt'][ii][-1, :, 0]}))
    f.gridDef[ii].vertices = np.copy(f.gridxy[0])
    f.gridDef[ii].saveVTK(outputDir + f'/original_wJaco_{ii}.vtk',
                          vtkFields=vtkFields('POINT_DATA', f.gridDef[ii].vertices.shape[0],
                                              scalars = {'logJacobian': psnger['Jyt'][ii][-1,:,0]
                                                         ,'Jacobian': np.exp(psnger['Jyt'][ii][-1,:,0])}))

print('done')


