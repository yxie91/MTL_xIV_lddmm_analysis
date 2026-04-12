from sys import path as sys_path
import os

from os.path import split, splitext, isfile
import glob
import numpy as np
from scipy.spatial import KDTree
import pandas as pd
import pyvista as pv
import h5py
from base.affineRegistration import rigidRegistration
from base.surfaces import Surface
from base.kernelFunctions import Kernel
from base.surfaceTemplate import SurfaceTemplate
from base.surfaceMatching import SurfaceMatching
from base.secondOrderSurfaceMatching import SecondOrderSurfaceMatching
from base import loggingUtils



df = pd.read_csv('Data/ADNI_Surface/demographic_data.csv', index_col=0)

inputDir = 'Data/ADNI_Surface/thickness'
outputDir = 'Data/ADNI_Surface/Population_Surfaces'
outputH5 = 'Data/ADNI_Surface/Population_Surfaces/ADNIthickness.h5'

forceRedoInitialData = False
forceRedoTemplate = False
forceRedoToBaseline = False
forceRedoToFollowups = False


loggingUtils.setup_default_logging(outputDir, fileName=f'info.txt', stdOutput = True)

doInitialData = True
doTemplate = True
doToBaseline = True
doToFollowups = True
existf5 = isfile(outputH5)
if existf5:
    f5 = h5py.File(outputH5)
    if 'initial_data' in f5 and not forceRedoInitialData:
        doInitialData = False
    if 'template' in f5 and not forceRedoTemplate:
        doTemplate = False
    if 'to_baseline' in f5 and not doTemplate and not forceRedoToBaseline:
        doToBaseline = False
    if 'to_followups' in f5 and not forceRedoToFollowups:
        doToFollowups = False
        
    f5.close()

doInitialData = True
if doInitialData:
    files = glob.glob(inputDir + '/*.vtk')#'/*/*.vtk'
    #print(files)
    data = {'id':[], 'months':[], 'group':[]}
    for name in files:
        dirs = name.split('/')
        d1 = dirs[-1].split('_')
        d2 = dirs[-2].split('_')
        data['id'].append(d1[0])
        data['months'].append(d1[1])
        data['group'].append(d2[0])
        #print(data['id'][-1], data['months'][-1], data['group'][-1])
    for k in data:
        data[k] = np.array(data[k])
        
    # Mappings to baseline
    ids = np.unique(data['id'])
    dataset = {'id':[], 'months':[], 'group':[], 'age':[], 'pts':[], 'faces':[], 'thickness':[]}#build subject-wise dataset
    for idt in ids:
        if (len(idt) < 4):
            id_ = f'{int(idt):04d}'
        else:
            id_ = idt
        dataset['id'].append(id_)
        J = np.nonzero(np.array(data['id']) == idt)[0]
        dataset['group'].append(data['group'][J[0]])
        ages = df.loc[int(idt), 'Age']
        ages = ages[1:-1].split(' ')
        #print(id, ages)
        mo = np.sort(data['months'][J])
        dataset['months'].append(list(mo))
        ages2 = []
        for j in range(len(J)):
            agediff = float(mo[j])/12
            if agediff <= 4.1:
                ages2.append(float(ages[0]) + agediff)
            else:
                ages2.append(float(mo[j]))
        dataset['age'].append(ages2)
        
    ## reading thickness
    for k,idt in enumerate(dataset['id']):
        dataset['thickness'].append([])
        dataset['pts'].append([])
        dataset['faces'].append([])
        for j in range(len(dataset['months'][k])):
            ffile = inputDir +'/'  + idt + '_' + dataset['months'][k][j] + '_thickness.vtk'#inputDir +'/' + dataset['group'][k] + '_group/' + idt + '_' + dataset['months'][k][j] + '_thickness.vtk'
            pvf = pv.read(ffile)
            pts = np.array(pvf.points)
            faces_ = np.array(pvf.faces, dtype=int)
            nf = faces_.shape[0] // 4
            #print(pts,faces_,nf)
            faces = np.zeros((nf, 3), dtype=int)
            for k_ in range(nf):
                faces[k_, :] = faces_[4*k_+1:4*k_+4]
            dataset['pts'][k].append(pts)
            dataset['faces'][k].append(faces)
            disp = pvf.point_data['displacement']
            dataset['thickness'][k].append(np.array(disp))

    with h5py.File(outputH5, 'a') as f5:
        if 'initial_data' in f5:
            del f5['initial_data']
        f5.create_group('initial_data') #create hdf5 group
        for k,idt in enumerate(dataset['id']):
            f5['initial_data'].create_group(idt)
            #f5['initial_data'][id]['group'] = [dataset['group'][k]]
            f5['initial_data'][idt].attrs.create('group', data = dataset['group'][k], dtype=h5py.string_dtype())
            for j in range(len(dataset['months'][k])):
                m = idt + '_' + dataset['months'][k][j]
                f5['initial_data'][idt].create_group(m)
                f5['initial_data'][idt][m]['vertices'] = dataset['pts'][k][j]
                f5['initial_data'][idt][m]['faces'] = dataset['faces'][k][j]
                f5['initial_data'][idt][m]['thickness'] = dataset['thickness'][k][j]
                f5['initial_data'][idt][m].attrs['age'] = dataset['age'][k][j]
                f5['initial_data'][idt][m].attrs.create('month', data = dataset['months'][k][j], dtype=h5py.string_dtype())
else:
    print('Reading data')
    dataset = {'id':[], 'months':[], 'group':[], 'age':[], 'pts':[], 'thickness':[]}
    with h5py.File(outputH5, 'r') as f5:
        for idt in f5['initial_data']:
            dataset['id'].append(idt)
            dataset['group'].append(f5['initial_data'][idt].attrs['group'])
            dataset['months'].append([])
            dataset['age'].append([])
            dataset['thickness'].append([])
            dataset['pts'].append([])
            for m in f5['initial_data'][idt].keys():
                dataset['thickness'][-1].append(np.array(f5['initial_data'][idt][m]['thickness']))
                dataset['pts'][-1].append(np.array(f5['initial_data'][idt][m]['pts']))
                dataset['age'][-1].append(f5['initial_data'][idt][m].attrs['age'])
                dataset['months'][-1].append(f5['initial_data'][idt][m].attrs['month'])

#for k, idt in enumerate(dataset['id']):
#    tree = KDTree(dataset['pts'][k][0])
#    dd, J = tree.query(dataset['pts'][0][0])
#    print(idt, J.min(), J.max(), dataset['thickness'][k][0].shape)
#    x = dataset['thickness'][k][0][J]


## Define kernels
secondOrder = False
K1 = Kernel(name='laplacian', sigma = 5.0)
K2 = Kernel(name='gauss', sigma = 5.0)

if doTemplate:
    ## Computing template on baselines
    options = {
        'mode': 'normal',
        'timeStep': 0.1,
        'KparDiff': K1,
        'KparDist': K2,
        'sigmaError': 1.,
        'errorType': 'current',
        'outputDir': outputDir+ '/surfaceTemplate',
        'testGradient':False,
        'lambdaPrior': 1.,
        'maxIter': 1000,
        'affine': 'none',
        'rotWeight': 10.,
        'sgd': None,
        'transWeight': 1.,
        'scaleWeight': 10.,
        'affineWeight': 100.,
        'updateTemplate': True,
        'pk_dtype': 'float32',
        'verb': True
    }

    dataset['betweenBaselinesRigid'] = []
    bfile = inputDir + '/' + dataset['id'][0] + '_' + dataset['months'][0][0] + '_thickness.vtk'#inputDir + '/' + dataset['group'][0] + '_group/' + dataset['id'][0] + '_' + dataset['months'][0][0] + '_thickness.vtk'
    print(bfile)
    surf0 = Surface(bfile)
    surf = []
    for k,id in enumerate(dataset['id']):
        bfile = inputDir + '/' + id + '_' + dataset['months'][k][0] + '_thickness.vtk'#inputDir + '/' + dataset['group'][k] + '_group/' + id + '_' + dataset['months'][k][0] + '_thickness.vtk'
        s = Surface(bfile)
        R0, T0 = rigidRegistration(surfaces = (s.vertices, surf0.vertices),  rotWeight=0., verb=False, temperature=10., annealing=True)# Rigid alignment between baselines
        dataset['betweenBaselinesRigid'].append([R0, T0])
        #print(R0, T0)
        s.updateVertices(s.vertices @ R0.T + T0)
        #print(bfile)
        surf.append(s)
    f = SurfaceTemplate(Template=None, Target=surf, options=options) #Compute population average surface
    f.computeTemplate()
    template = Surface(f.fvTmpl)

    with h5py.File(outputH5, 'a') as f5:#save template
        if 'template' in f5:
            del f5['template']
        f5.create_group('template')
        f5['template'].create_dataset('vertices', data=template.vertices)
        f5['template'].create_dataset('faces', data=template.faces)
        
else:
    print('Reading template')
    with h5py.File(outputH5, 'r') as f5:
        template = Surface(surf = (np.array(f5['template']['faces']), np.array(f5['template']['vertices'])))


options = {
    'mode': 'normal',
    'timeStep': 0.1,
    'KparDiff': K1,
    'KparDist': K2,
    'internalCost': [['elastic', 50]],
    'internalWeight': 1.,
    'sigmaError': .25,
    'errorType': 'varifold',
    'maxIter': 1000,
    'affine': 'none',
    'rotWeight': 10.,
    'saveRate': 50,
    'transWeight': 1.,
    'algorithm': 'bfgs',
    'scaleWeight': 10.,
    'affineWeight': 100.,
    'verb': False,
    'pk_dtype': 'float32'
}


#template to baselines
dataset['toBaselineRigid'] = []
dataset['toBaselineTransforms'] = []
if doToBaseline:
    for k,idt in enumerate(dataset['id']):
        ffile = inputDir +'/'  + idt + '_' + dataset['months'][k][0] + '_thickness.vtk'#inputDir +'/' + dataset['group'][k] + '_group/' + idt + '_' + dataset['months'][k][0] + '_thickness.vtk'
        surf = Surface(ffile)
        R0, T0 = rigidRegistration(surfaces = (surf.vertices, template.vertices),  rotWeight=0., verb=False, temperature=10., annealing=True)
        dataset['toBaselineRigid'].append(np.concatenate((R0, T0)))
        surf.updateVertices(surf.vertices @ R0.T + T0)
        options['outputDir'] = outputDir + f"/to_baseline/{idt}_{dataset['months'][k][0]}/"
        if secondOrder:
            f = SecondOrderSurfaceMatching(Template=template, Target=surf, options=options)
        else:
            f = SurfaceMatching(Template=template, Target=surf, options=options)#Register template to subject baseline
        f.optimizeMatching()
        dataset['toBaselineTransforms'].append(f.fvDef.vertices)#deformed subject to each subject
            
    with h5py.File(outputH5, 'a') as f5:
        f5.create_group('to_baseline')
        for k,idt in enumerate(dataset['id']):
            f5['to_baseline'].create_group(idt)
            f5['to_baseline'][idt].create_dataset('rigid', data = dataset['toBaselineRigid'][k])
            f5['to_baseline'][idt].create_dataset('vertices', data = dataset['toBaselineTransforms'][k])
else:
    print('Reading t2b')
    with h5py.File(outputH5, 'r') as f5:
        for k,idt in enumerate(dataset['id']):
            dataset['toBaselineRigid'].append(np.array(f5['to_baseline'][idt]['rigid']))
            dataset['toBaselineTransforms'].append(np.array(f5['to_baseline'][idt]['vertices']))



#baseline to follow-ups
dataset['inSubjectRigid'] = []
dataset['inSubjectTransforms'] = []
if doToFollowups:
    for k,idt in enumerate(dataset['id']):
        bfile = inputDir + '/' + idt + '_' + dataset['months'][k][0] + '_thickness.vtk'#inputDir + '/' + dataset['group'][k] + '_group/' + idt + '_' + dataset['months'][k][0] + '_thickness.vtk'
        print(bfile)
        surf0 = Surface(bfile)
        dataset['inSubjectRigid'].append([None])
        dataset['inSubjectTransforms'].append([surf0.vertices])
        for j in range(1,len(dataset['months'][k])):
            ffile = inputDir +'/' + idt + '_' + dataset['months'][k][j] + '_thickness.vtk'#inputDir +'/' + dataset['group'][k] + '_group/' + idt + '_' + dataset['months'][k][j] + '_thickness.vtk'
            print(ffile)
            surf = Surface(ffile)
            R0, T0 = rigidRegistration(surfaces = (surf.vertices, surf0.vertices),  rotWeight=0., verb=False, temperature=10., annealing=True)
            dataset['inSubjectRigid'][k].append(np.concatenate((R0, T0)))
            surf.updateVertices(surf.vertices @ R0.T + T0)

            options['outputDir'] = outputDir + f"/to_follow_ups/{idt}_{dataset['months'][k][0]}_{dataset['months'][k][j]}/"
            print(options['outputDir'])
            if secondOrder:
                f = SecondOrderSurfaceMatching(Template=surf0, Target=surf, options=options)
            else:
                f = SurfaceMatching(Template=surf0, Target=surf, options=options)
            f.optimizeMatching()
            dataset['inSubjectTransforms'][k].append(f.fvDef.vertices)
            

    with h5py.File(outputH5, 'a') as f5:
        f5.create_group('to_followups')
        for k,idt in enumerate(dataset['id']):
            f5['to_followups'].create_group(idt)
            for j in range(1,len(dataset['months'][k])):
                m = idt + '_' + dataset['months'][k][j]
                f5['to_followups'][idt].create_group(m)
                f5['to_followups'][idt][m].create_dataset('rigid', data = dataset['inSubjectRigid'][k][j])
                f5['to_followups'][idt][m].create_dataset('vertices', data = dataset['inSubjectTransforms'][k][j])
else:
    print('reading b2f')
    with h5py.File(outputH5, 'r') as f5:
        for k,idt in enumerate(dataset['id']):
            dataset['inSubjectRigid'].append([None])
            dataset['inSubjectTransforms'].append([np.array(dataset['pts'][k][0])])
            for j in range(1,len(dataset['months'][k])):
                m = idt + '_' + dataset['months'][k][j]
                dataset['inSubjectRigid'][k].append(np.array(f5['to_followups'][idt][m]['rigid']))
                dataset['inSubjectTransforms'][k].append(np.array(f5['to_followups'][idt][m]['vertices']))


### Aligning thickness: LDDMM ARE NOT USED, ONLY ROTATION AND TRANSLATION ARE USED

#From follow-ups to baselines
dataset['f2b_thickness'] = []
for k, idt in enumerate(dataset['id']):
    dataset['f2b_thickness'].append([dataset['thickness'][k][0]])
    for j in range(1,len(dataset['months'][k])):
        R0 = dataset['inSubjectRigid'][k][j][:-1, :]
        T0 = dataset['inSubjectRigid'][k][j][-1, :]
        tree = KDTree(dataset['pts'][k][j]@R0.T + T0)#maps follow-up thickness onto baseline surface using nn interpolation
        dd, J = tree.query(dataset['pts'][k][0])#at the baseline's vertexes
        dataset['f2b_thickness'][k].append(dataset['thickness'][k][j][J])


#From baselines to template
dataset['b2t_thickness'] = []
for k, idt in enumerate(dataset['id']):
    dataset['b2t_thickness'].append([])
    for j in range(len(dataset['months'][k])):
        R0 = dataset['toBaselineRigid'][k][:-1, :]
        T0 = dataset['toBaselineRigid'][k][-1, :]
        tree = KDTree(dataset['pts'][k][0]@R0.T + T0)
        dd, J = tree.query(template.vertices)#at the subject-generated vertexes
        dataset['b2t_thickness'][k].append(dataset['f2b_thickness'][k][j][J])#stores the subjects' thickness at different tp above interpolated on the template coordinates

print('Saving thickness maps')
with h5py.File(outputH5, 'a') as f5:
    if 'thickness_maps' in f5:
        del f5['thickness_maps']
    f5.create_group('thickness_maps')
    for k, idt in enumerate(dataset['id']):
        f5['thickness_maps'].create_group(idt)
        f5['thickness_maps'][idt].create_group('f2b')
        f5['thickness_maps'][idt].create_group('b2t')
        for j in range(len(dataset['months'][k])):
            m = idt + '_' + dataset['months'][k][j]
            f5['thickness_maps'][idt]['b2t'].create_dataset(m, data = dataset['b2t_thickness'][k][j]) 
            f5['thickness_maps'][idt]['f2b'].create_dataset(m, data = dataset['f2b_thickness'][k][j]) 
            
#Can be used to compute mean_thickness = np.mean(all_subjects_all_timepoints, axis=0) for different groups and compare