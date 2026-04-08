import logging

import numpy as np
import pandas as pd
from os.path import splitext
from copy import deepcopy
from vtk import vtkOBJReader
from scipy.spatial.distance import squareform, pdist
from numpy.random import default_rng
from meshpy.geometry import GeometryBuilder
import meshpy.tet as tet
import meshpy.triangle as tri
from .curves import Curve
from .surfaces import Surface
from .vtk_fields import vtkFields


class PointSet:
    def __init__(self, data=None, weights=None, volumeRatio=1000.,
                 maxPoints=None, labels=None):
        if type(data) in (list, tuple):
            if isinstance(data[0], PointSet):
                self.concatenate(data)
            elif type(data[0]) is str:
                fvl = []
                for name in data:
                    fvl.append(PointSet(data=name))
                self.concatenate(fvl)
            else:
                self.vertices = np.copy(data[1])
                self.faces = np.copy(data[0])
                self.dim = self.vertices.shape[1]
                self.labels = deepcopy(labels)
                self.updateWeights(weights)
        elif isinstance(data, np.ndarray):
            self.vertices = data.copy()
            self.faces = np.arange(self.vertices.shape[0])[:, None]
            self.updateWeights(weights)
            self.labels = deepcopy(labels)
            self.dim = data.shape[1]
        elif type(data) is str:
            self.read(data, maxPoints=maxPoints)
            if labels is not None:
                self.labels = deepcopy(labels)
        elif issubclass(type(data), PointSet):
            self.vertices = np.copy(data.vertices)
            self.faces = np.arange(self.vertices.shape[0])[:, None]
            if labels is not None:
                self.labels = deepcopy(labels)
            else:
                self.labels = deepcopy(data.labels)

            if weights is not None:
                self.updateWeights(weights)
            else:
                self.vertex_weights = np.copy(data.vertex_weights)
                self.face_weights = np.copy(data.face_weights)
            self.dim = data.dim
        elif issubclass(type(data), Curve):
            g = GeometryBuilder()
            g.add_geometry(data.vertices, data.faces)
            mesh_info = tri.MeshInfo()
            g.set(mesh_info)
            vol = data.enclosedArea()
            f = tri.build(mesh_info, verbose=False,
                          max_volume=vol / volumeRatio)
            self.vertices = np.array(f.points)
            self.faces = np.arange(self.vertices.shape[0])[:, None]
            self.labels = deepcopy(labels)
            self.dim = 2
            self.updateWeights(weights)
        elif issubclass(type(data), Surface):
            g = GeometryBuilder()
            g.add_geometry(data.vertices, data.faces)
            mesh_info = tet.MeshInfo()
            g.set(mesh_info)
            vol = data.surfVolume()
            f = tet.build(mesh_info, options=tet.Options(switches='q1.2/10'),
                          verbose=True, max_volume=vol / volumeRatio)
            self.vertices = np.array(f.points)
            self.faces = np.arange(self.vertices.shape[0])[:, None]
            self.labels = deepcopy(labels)
            self.dim = 3
            self.updateWeights(weights)
            print(f'Point Set: {self.vertices.shape[0]} vertices')
        else:
            self.vertices = np.empty(0)
            self.faces = np.empty(0)
            self.vertex_weights = np.empty(0)
            self.face_weights = np.empty(0)
            self.labels = np.empty(0)
            self.dim = 0

        self.pointdata = None

        # if type(data) is str:
        #     self.read(data, maxPoints=maxPoints)
        # elif issubclass(type(data), PointSets):
        #     self.vertices = np.copy(data.vertices)
        #     if weights is None:
        #         self.vertex_weights = np.copy(data.vertex_weights)
        #     else:
        #         self.vertex_weights = weights
        # elif isinstance(data, np.ndarray):
        #     self.vertices = data.copy()
        #     if weights is None:
        #         self.vertex_weights = np.ones(data.shape[0])
        #     else:
        #         self.vertex_weights = weights
        # else:
        #     self.vertices = np.empty(0)
        #     self.vertex_weights = np.empty(0)

    def updateVertices(self, pts):
        self.vertices = np.copy(pts)

    def updateWeights(self, weights= None):
        if self.vertices is not None:
            if weights is None:
                self.vertex_weights = np.ones(self.vertices.shape[0])
            elif np.isscalar(weights):
                self.vertex_weights = weights * np.ones(self.vertices.shape[0])
            else:
                self.vertex_weights = np.copy(weights)
            self.face_weights = np.copy(self.vertex_weights)

    def addToPlot(self, ax, ec = 'b', fc = 'r', al=.5, lw=1):
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        z = self.vertices[:, 2]
        ax.scatter(x,y,z, alpha=al)
        xlim = [x.min(),x.max()]
        ylim = [y.min(),y.max()]
        zlim = [z.min(),z.max()]
        return [xlim, ylim, zlim]

    def concatenate(self, fvl):
        nv = 0
        for fv in fvl:
            nv += fv.vertices.shape[0]
        self.dim = fvl[0].dim
        self.vertices = np.zeros([nv,self.dim])
        self.vertex_weights = np.zeros(nv)

        nv0 = 0
        for fv in fvl:
            nv = nv0 + fv.vertices.shape[0]
            self.vertices[nv0:nv, :] = fv.vertices
            self.vertex_weights[nv0:nv] = fv.vertex_weights
            nv0 = nv
        self.faces = np.arange(self.vertices.shape[0])[:, None]

    def set_weights(self, knn=5):
        d = squareform(pdist(self.vertices))
        d = np.sort(d, axis=1)
        eps = d[:, knn].mean() + 1e-10
        self.vertex_weights = np.pi * eps ** 2 / (d < eps).sum(axis=1)


    def read(self, filename, maxPoints=None):
        head, tail = splitext(filename)
        if tail in ('.obj', '.OBJ'):
            self.readOBJ(filename, maxPoints=maxPoints)
        elif tail == '.csv':
            self.vertices = pd.read_csv(filename).to_numpy()
            self.faces = np.arange(self.vertices.shape[0])[:, None]
            self.vertex_weights = np.ones(self.vertices.shape[0])
            self.face_weights = np.copy(self.vertex_weights)
            self.labels = None
        else:
            self.vertices, self.vertex_weights = readVector(filename)
            self.faces = np.arange(self.vertices.shape[0])[:, None]
            self.face_weights = np.copy(self.vertex_weights)
            self.labels = None

    def readOBJ(self, fileName, maxPoints=None):
        u = vtkOBJReader()
        u.SetFileName(fileName)
        u.Update()
        v = u.GetOutput()
        # print v
        npoints = int(v.GetNumberOfPoints())
        V = np.zeros([npoints, 3])
        for kk in range(npoints):
            V[kk, :] = np.array(v.GetPoint(kk))

        if maxPoints is not None and maxPoints < npoints:
            rng = default_rng()
            select = rng.choice(npoints, maxPoints, replace=False)
            V = V[select, :]

        self.vertices = V
        self.faces = np.arange(self.vertices.shape[0])[:, None]
        self.set_weights(5)
        self.labels = None


    def select_faces_(self, select):
        faces = self.faces[select, :]
        selectv = np.zeros(self.vertices.shape[0], dtype=bool)
        for j in range(faces.shape[0]):
            selectv[faces[j,:]] = True
        vertices = self.vertices[selectv, :]
        newindx = - np.ones(self.vertices.shape[0], dtype=int)
        newindx[selectv] = np.arange(selectv.sum(), dtype=int)
        faces = newindx[faces]
        return faces, vertices, selectv

    def select_faces(self, select):
        faces, vertices, selectv = self.select_faces_(select)
        res = PointSet(data=(faces, vertices), weights=self.face_weights[select])
        # res.updateWeights(self.face_weights)
        return res, np.nonzero(selectv)[0]


    def saveVTK(self, filename):
        self.save(filename)
    def save(self, filename, vtk_fields=None):
        if vtk_fields is None:
            vtk_fields = vtkFields('POINT_DATA', self.vertices.shape[0])
        if not 'weights' in vtk_fields.scalars.keys():
            vtk_fields.scalars['weights'] = self.vertex_weights
        if not 'labels' in vtk_fields.scalars.keys() and self.labels is not None:
            vtk_fields.scalars['labels'] = self.labels
        savePoints(filename, self.vertices, vtk_fields)

def readVector(filename):
    try:
        with open(filename, 'r') as fn:
            ln0 = fn.readline().split()
            N = int(ln0[0])
            dim = int(ln0[1])
            #print 'reading ', filename, ':', N, ' landmarks'
            v = np.zeros([N, dim])
            w = np.zeros([N,1])

            for i in range(N):
                ln0 = fn.readline().split()
                #print ln0
                for k in range(3):
                    v[i,k] = float(ln0[k])
                w[i] = ln0[3]
    except IOError:
        print('cannot open ', filename)
        raise
    return v,w




def loadlmk(filename, dim=3):
# [x, label] = loadlmk(filename, dim)
# Loads 3D landmarks from filename in .lmk format.
# Determines format version from first line in file
#   if version number indicates scaling and centering, transform coordinates...
# the optional parameter s in a 3D scaling factor

    try:
        with open(filename, 'r') as fn:
            ln0 = fn.readline()
            versionNum = 1
            versionStrs = ln0.split("-")
            if len(versionStrs) == 2:
                try:
                    versionNum = int(float(versionStrs[1]))
                except:
                    pass

            #print fn
            ln = fn.readline().split()
            #print ln0, ln
            N = int(ln[0])
            #print 'reading ', filename, ':', N, ' landmarks'
            x = np.zeros([N, dim])
            label = []

            for i in range(N):
                ln = fn.readline()
                label.append(ln) 
                ln0 = fn.readline().split()
                #print ln0
                for k in range(dim):
                    x[i,k] = float(ln0[k])
            if versionNum >= 6:
                lastLine = ''
                nextToLastLine = ''
                # read the rest of the file
                # the last two lines contain the center and the scale variables
                while 1:
                    thisLine = fn.readline()
                    if not thisLine:
                        break
                    nextToLastLine = lastLine
                    lastLine = thisLine
                    
                centers = nextToLastLine.rstrip('\r\n').split(',')
                scales = lastLine.rstrip('\r\n').split(',')
                if len(scales) == dim and len(centers) == dim:
                    if scales[0].isdigit and scales[1].isdigit and scales[2].isdigit and centers[0].isdigit \
                            and centers[1].isdigit and centers[2].isdigit:
                        x[:, 0] = x[:, 0] * float(scales[0]) + float(centers[0])
                        x[:, 1] = x[:, 1] * float(scales[1]) + float(centers[1])
                        x[:, 2] = x[:, 2] * float(scales[2]) + float(centers[2])
                
    except IOError:
        print('cannot open ', filename)
        raise
    return x, label




def  savelmk(x, filename):
# savelmk(x, filename)
# save landmarks in .lmk format.

    with open(filename, 'w') as fn:
        str = 'Landmarks-1.0\n {0: d}\n'.format(x.shape[0])
        fn.write(str)
        for i in range(x.shape[0]):
            str = '"L-{0:d}"\n'.format(i)
            fn.write(str)
            str = ''
            for k in range(x.shape[1]):
                str = str + '{0: f} '.format(x[i,k])
            str = str + '\n'
            fn.write(str)
        fn.write('1 1 \n')

        
# Saves in .vtk format
def savePoints(fileName, x, vtk_fields=None):
    if x.shape[1] <3:
        x = np.concatenate((x, np.zeros((x.shape[0],3-x.shape[1]))), axis=1)
    with open(fileName, 'w') as fvtkout:
        fvtkout.write('# vtk DataFile Version 3.0\nSurface Data\nASCII\nDATASET UNSTRUCTURED_GRID\n') 
        fvtkout.write('\nPOINTS {0: d} float'.format(x.shape[0]))
        for ll in range(x.shape[0]):
            fvtkout.write('\n{0: f} {1: f} {2: f}'.format(x[ll,0], x[ll,1], x[ll,2]))

        if vtk_fields is not None and vtk_fields.data_type == 'POINT_DATA':
            vtk_fields.write(fvtkout)

# Saves in .vtk format
def saveTrajectories(fileName, xt):
    with open(fileName, 'w') as fvtkout:
        fvtkout.write('# vtk DataFile Version 3.0\ncurves \nASCII\nDATASET POLYDATA\n')
        npt = xt.shape[0]*xt.shape[1]
        fvtkout.write('\nPOINTS {0: d} float'.format(npt))
        if xt.shape[2] == 2:
            xt = np.concatenate((xt, np.zeros([xt.shape[0],xt.shape[1], 1])), axis=2)
        for t in range(xt.shape[0]):
            for ll in range(xt.shape[1]):
                fvtkout.write('\n{0: f} {1: f} {2: f}'.format(xt[t,ll,0], xt[t,ll,1], xt[t,ll,2]))
        nlines = (xt.shape[0]-1)*xt.shape[1]
        fvtkout.write('\nLINES {0:d} {1:d}'.format(nlines, 3*nlines))
        for t in range(xt.shape[0]-1):
            for ll in range(xt.shape[1]):
                fvtkout.write('\n2 {0: d} {1: d}'.format(t*xt.shape[1]+ll, (t+1)*xt.shape[1]+ll))

        fvtkout.write(('\nPOINT_DATA {0: d}').format(npt))
        fvtkout.write('\nSCALARS time int 1\nLOOKUP_TABLE default')
        for t in range(xt.shape[0]):
            for ll in range(xt.shape[1]):
                fvtkout.write(f'\n{t}')

        fvtkout.write('\n')



def epsilonNet(x, rate):
    #print 'in epsilon net'
    n = x.shape[0]
    dim = x.shape[1]
    inNet = np.zeros(n, dtype=int)
    inNet[0]=1
    net = np.nonzero(inNet)[0]
    survivors = np.ones(n, dtype=np.int)
    survivors[0] = 0 ;
    dist2 = ((x.reshape([n, 1, dim]) -
              x.reshape([1,n,dim]))**2).sum(axis=2)
    d2 = np.sort(dist2, axis=0)
    i = np.int_(1.0/rate)
    eps2 = (np.sqrt(d2[i,:]).sum()/n)**2
    #print n, d2.shape, i, np.sqrt(eps2)
    

    i1 = np.nonzero(dist2[net, :] < eps2)
    survivors[i1[1]] = 0
    i2 = np.nonzero(survivors)[0]
    while len(i2) > 0:
        closest = np.unravel_index(np.argmin(dist2[net.reshape([len(net),1]), i2.reshape([1, len(i2)])].ravel()), [len(net), len(i2)])
        inNet[i2[closest[1]]] = 1 
        net = np.nonzero(inNet)[0]
        i1 = np.nonzero(dist2[net, :] < eps2)
        survivors[i1[1]] = 0
        i2 = np.nonzero(survivors)[0]
        #print len(net), len(i2)
    idx = - np.ones(n, dtype=np.int)
    for p in range(n):
        closest = np.unravel_index(np.argmin(dist2[net, p].ravel()), [len(net), 1])
        #print 'p=', p, closest, len(net)
        idx[p] = closest[0]
        
        #print idx
    return net, idx


