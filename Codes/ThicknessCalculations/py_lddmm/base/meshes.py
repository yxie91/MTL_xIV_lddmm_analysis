import os
import numpy as np
from numba import jit, int64
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
import scipy as scp
from .pointSets import PointSet
from .curves import Curve
from .surfaces import Surface
from .vtk_fields import vtkFields
from meshpy.geometry import GeometryBuilder
import meshpy.tet as tet
import meshpy.triangle as tri
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.csgraph import connected_components
import pyvista as pv

try:
    from vtk import vtkCellArray, vtkPoints, vtkPolyData, vtkVersion, \
        vtkLinearSubdivisionFilter, vtkQuadricDecimation, \
        vtkWindowedSincPolyDataFilter, vtkImageData, VTK_FLOAT, \
        vtkDoubleArray, vtkContourFilter, vtkPolyDataConnectivityFilter, \
        vtkCleanPolyData, vtkPolyDataReader, vtkUnstructuredGridReader, \
        vtkOBJReader, vtkSTLReader, \
        vtkDecimatePro, VTK_UNSIGNED_CHAR, vtkPolyDataToImageStencil, \
        vtkImageStencil
    from vtk.util.numpy_support import vtk_to_numpy
    gotVTK = True
except ImportError:
    v2n = None
    print('could not import VTK functions')
    gotVTK = False

import logging
from .pointSets_util import det2D, det3D


def twelve_vertexes(dimension=3):
    if dimension == 2:
        t = np.linspace(0, 2*np.pi, 12)
        ico = np.zeros((12, 2))
        ico[:, 0] = np.cos(t)
        ico[:, 1] = np.sin(t)
    else:
        phi = (1+np.sqrt(5))/2

        ico = np.array([
            [phi, 1, 0],
            [phi, -1, 0],
            [-phi, -1, 0],
            [-phi, 1, 0],
            [1, 0, phi],
            [-1, 0, phi],
            [-1, 0, -phi],
            [1, 0, -phi],
            [0, phi, 1],
            [0, phi, -1],
            [0, -phi, -1],
            [0, -phi, 1]]
        )

    return ico


# @jit(nopython=True)
def computeCentersVolumesNormals__(faces, vertices, weights,
                                   checkOrientation=False):
    dim = vertices.shape[1]
    if dim == 2:
        xDef1 = vertices[faces[:, 0], :]
        xDef2 = vertices[faces[:, 1], :]
        xDef3 = vertices[faces[:, 2], :]
        centers = (xDef1 + xDef2 + xDef3) / 3
        x12 = xDef2-xDef1
        x13 = xDef3-xDef1
        volumes = det2D(x12, x13)/2
        if checkOrientation:
            if volumes.min() < -1e-12:
                if volumes.max() > 1e-12:
                    print('Warning: mesh has inconsistent orientation',
                          (volumes < 0).sum(), (volumes > 0).sum())
                else:
                    f_ = np.copy(faces[:, 1])
                    faces[:, 1] = np.copy(faces[:, 2])
                    faces[:, 2] = f_
                    xDef2 = vertices[faces[:, 1], :]
                    xDef3 = vertices[faces[:, 2], :]
                    x12 = xDef2-xDef1
                    x13 = xDef3-xDef1
                    volumes = (x12[:, 0] * x13[:, 1] - x12[:, 1]*x13[:, 0])/2
        J = np.array([[0., -1.], [1., 0.]])
        normals = np.zeros((3, faces.shape[0], 2))
        normals[0, :, :] = (xDef3 - xDef2) @ J.T
        normals[1, :, :] = (xDef1 - xDef3) @ J.T
        normals[2, :, :] = (xDef2 - xDef1) @ J.T
    elif dim == 3:
        xDef1 = vertices[faces[:, 0], :]
        xDef2 = vertices[faces[:, 1], :]
        xDef3 = vertices[faces[:, 2], :]
        xDef4 = vertices[faces[:, 3], :]
        centers = (xDef1 + xDef2 + xDef3 + xDef4) / 4
        x12 = xDef2-xDef1
        x13 = xDef3-xDef1
        x14 = xDef4-xDef1
        volumes = det3D(x12, x13, x14)/6
        if checkOrientation:
            if volumes.min() < -1e-12:
                if volumes.max() > 1e-12:
                    print('Warning: mesh has inconsistent orientation',
                          (volumes < 0).sum(), (volumes > 0).sum())
                else:
                    f_ = np.copy(faces[:, 2])
                    faces[:, 2] = np.copy(faces[:, 3])
                    faces[:, 3] = f_
                    xDef3 = vertices[faces[:, 2], :]
                    xDef4 = vertices[faces[:, 3], :]
                    x13 = xDef3 - xDef1
                    x14 = xDef4 - xDef1
                    volumes = det3D(x12, x13, x14) / 6
        normals = np.zeros((4, faces.shape[0], 3))
        normals[0, :, :] = np.cross(xDef4 - xDef2, xDef3 - xDef2)
        normals[1, :, :] = np.cross(xDef2 - xDef1, xDef4 - xDef1)
        normals[2, :, :] = np.cross(xDef4 - xDef1, xDef2 - xDef1)
        normals[3, :, :] = np.cross(xDef2 - xDef1, xDef3 - xDef1)

    vertex_weights = np.zeros(vertices.shape[0])
    face_per_vertex = np.zeros(vertices.shape[0])
    for k in range(faces.shape[0]):
        for j in range(faces.shape[1]):
            vertex_weights[faces[k, j]] += weights[k] * volumes[k]
            face_per_vertex[faces[k, j]] += volumes[k]
    mv = volumes.sum()/volumes.shape[0]
    for k in range(face_per_vertex.shape[0]):
        if np.fabs(face_per_vertex[k]) > 1e-10*mv:
            vertex_weights[k] /= face_per_vertex[k]

    return centers, volumes, normals, vertex_weights


@jit(nopython=True)
def get_edges_(faces, parallel=True):
    int64 = "int64"
    dim = faces.shape[1] - 1
    nf = faces.shape[0]

    edgi = dict()
    ne = 0
    edges = np.zeros(((dim+1)*nf, dim), dtype=int64)
    for k in range(faces.shape[0]):
        for j in range(dim+1):
            indx = list(faces[k, :])
            indx.remove(faces[k, j])
            indx.sort()
            inds = ''
            for s in indx:
                inds += str(s) + '-'
            if inds not in edgi:
                edgi[inds] = ne
                edges[ne, :] = indx
                ne += 1
    edges = edges[:ne, :]

    edgesOfFaces = np.zeros(faces.shape, dtype=int64)
    for k in range(faces.shape[0]):
        for j in range(dim+1):
            indx = list(faces[k, :])
            indx.remove(faces[k, j])
            indx.sort()
            inds = ''
            for s in indx:
                inds += str(s) + '-'
            edgesOfFaces[k, j] = edgi[inds]

    facesOfEdges = - np.ones((ne, 2), dtype=int64)
    if dim == 2:
        for k in range(faces.shape[0]):
            i0 = faces[k, 0]
            i1 = faces[k, 1]
            i2 = faces[k, 2]
            for f in ([i0, i1], [i1, i2], [i2, i0]):
                kk = edgi[str(min(f[0], f[1]))+'-'+str(max(f[0], f[1])) + '-']
                if facesOfEdges[kk, 0] >= 0:
                    facesOfEdges[kk, 1] = k
                else:
                    facesOfEdges[kk, 0] = k
    else:
        for k in range(faces.shape[0]):
            i0 = faces[k, 0]
            i1 = faces[k, 1]
            i2 = faces[k, 2]
            i3 = faces[k, 3]
            for f in ([i0, i1, i2], [i2, i1, i3], [i0, i2, i3], [i1, i0, i3]):
                f.sort()
                kk = edgi[str(f[0])+'-'+str(f[1])+'-'+str(f[2])+'-']
                if facesOfEdges[kk, 0] >= 0:
                    facesOfEdges[kk, 1] = k
                else:
                    facesOfEdges[kk, 0] = k

    bdry = np.zeros(edges.shape[0], dtype=int64)
    for k in range(edges.shape[0]):
        if facesOfEdges[k, 1] < 0:
            bdry[k] = 1
    return edges, facesOfEdges, edgesOfFaces, bdry


class Mesh(PointSet):
    def __init__(self, data=None, weights=None, image=None, imNames=None,
                 volumeRatio=1000., checkOrientation=False, maxPoints=None):
        if type(data) in (list, tuple):
            if isinstance(data[0], Mesh):
                self.concatenate(data)
            elif type(data[0]) is str:
                fvl = []
                for name in data:
                    fvl.append(Mesh(data=name))
                self.concatenate(fvl)
            else:
                self.vertices = np.copy(data[1])
                self.faces = np.int_(np.copy(data[0]))
                self.dim = self.vertices.shape[1]
                self.component = np.zeros(self.faces.shape[0], dtype=int)
                if weights is None:
                    self.face_weights = np.ones(self.faces.shape[0], dtype=int)
                elif np.isscalar(weights):
                    self.face_weights = weights*np.ones(self.faces.shape[0],
                                                        dtype=int)
                else:
                    self.face_weights = weights

                self.computeCentersVolumesNormals(checkOrientation=checkOrientation)

                if image is None:
                    self.image = np.ones((self.faces.shape[0], 1))
                    self.imNames = ['0']
                    self.imageDim = 1
                else:
                    self.image = np.copy(image)
                    self.imageDim = self.image.shape[1]
                    self.imNames = []
                    if imNames is None:
                        for k in range(self.imageDim):
                            self.imNames.append(str(k))
                    else:
                        self.imNames = imNames
        elif type(data) is str:
            self.read(data)
        elif issubclass(type(data), Mesh):
            self.vertices = np.copy(data.vertices)
            self.volumes = np.copy(data.volumes)
            self.normals = np.copy(data.normals)
            self.faces = np.copy(data.faces)
            self.centers = np.copy(data.centers)
            self.component = np.copy(data.component)
            if weights is None:
                self.face_weights = np.copy(data.face_weights)
                self.vertex_weights = np.copy(data.vertex_weights)
            else:
                self.updateWeights(weights)
            self.dim = data.dim
            self.image = np.copy(data.image)
            self.imageDim = data.imageDim
            self.imNames = data.imNames
        elif issubclass(type(data), Curve):
            g = GeometryBuilder()
            g.add_geometry(data.vertices, data.faces)
            mesh_info = tri.MeshInfo()
            g.set(mesh_info)
            vol = data.enclosedArea()
            f = tri.build(mesh_info, verbose=False, max_volume=vol/volumeRatio)
            self.vertices = np.array(f.points)
            self.faces = np.array(f.elements, dtype=int)
            self.dim = 2
            self.component = np.zeros(self.faces.shape[0], dtype=int)
            if weights is None:
                w = 1
            else:
                w = weights
            self.vertex_weights = w * np.ones(self.vertices.shape[0],
                                              dtype=int)
            self.face_weights = w * np.ones(self.faces.shape[0], dtype=int)
            self.computeCentersVolumesNormals(checkOrientation=checkOrientation)
            self.image = np.ones((self.faces.shape[0], 1))
            self.imageDim = 1
            self.imNames = ['0']
        elif issubclass(type(data), Surface):
            g = GeometryBuilder()
            g.add_geometry(data.vertices, data.faces)
            mesh_info = tet.MeshInfo()
            g.set(mesh_info)
            vol = data.surfVolume()
            f = tet.build(mesh_info,
                          options=tet.Options(switches='q1.2/10'),
                          verbose=True, max_volume=vol/volumeRatio)
            self.vertices = np.array(f.points)
            self.faces = np.array(f.elements, dtype=int)
            self.dim = 3
            self.component = np.zeros(self.faces.shape[0], dtype=int)
            if weights is None:
                w = 1
            else:
                w = weights
            self.face_weights = w * np.ones(self.faces.shape[0], dtype=int)
            self.computeCentersVolumesNormals()
            self.image = np.ones((self.faces.shape[0], 1))
            self.imageDim = 1
            self.imNames = ['0']
            self.computeCentersVolumesNormals(checkOrientation=checkOrientation)
            print(f'Mesh: {self.vertices.shape[0]} vertices, {self.faces.shape[0]} cells')
        else:
            self.vertices = np.empty(0)
            self.centers = np.empty(0)
            self.faces = np.empty(0)
            self.volumes = np.empty(0)
            self.normals = np.empty(0)
            self.component = np.empty(0)
            self.face_weights = np.empty(0)
            self.vertex_weights = np.empty(0)
            self.image = np.empty(0)
            self.imageDim = 0
            self.imNames = []
            self.dim = 0

        self.edges = None
        self.facesOfEdges = None
        self.edgesOfFaces = None
        self.bdry_indices = None
        self.bdry = None

    def from_netgen(self, nmesh):
        p_ = nmesh.Points()
        dim = len(list(p_[1]))
        if dim == 2:
            m = nmesh.Elements2D().NumPy()
        else:
            m = nmesh.Elements3D().NumPy()
        self.dim = dim
        p = np.zeros((len(p_), dim))
        for k, pt in enumerate(p_):
            p[k, :] = pt.p
        self.vertices = p
        self.faces = m['nodes'][:,:(dim+1)]-1
        self.component = np.zeros(self.faces.shape[0], dtype=int)
        self.vertex_weights = np.ones(self.vertices.shape[0], dtype=int)
        self.face_weights = np.ones(self.faces.shape[0], dtype=int)
        self.computeCentersVolumesNormals(checkOrientation=True)
        # if targetSize is not None:
        #     self.Simplify(target=targetSize)


    def read(self, filename):
        (mainPart, ext) = os.path.splitext(filename)
        if ext == '.vtk':
            self.readVTK(filename)
        else:
            self.vertices = np.empty(0)
            self.centers = np.empty(0)
            self.faces = np.empty(0)
            self.volumes = np.empty(0)
            self.normals = np.empty(0)
            self.component = np.empty(0)
            self.face_weights = np.empty(0)
            self.vertex_weights = np.empty(0)
            self.dim = 0
            self.imNames = None
            raise NameError('Unknown Mesh Extension: '+filename)

    # face centers and area weighted normal
    def computeCentersVolumesNormals(self, checkOrientation=False):
        self.centers, self.volumes, self.normals, self.vertex_weights =\
            computeCentersVolumesNormals__(self.faces, self.vertices,
                                           self.face_weights,
                                           checkOrientation=checkOrientation)

    def select_faces(self, select):
        faces, vertices, selectv = self.select_faces_(select)
        res = Mesh(data=(faces, vertices), weights=self.face_weights[select])
        # res.updateWeights(self.vertex_weights[selectv])
        res.updateImage(self.image[select, :])
        return res, np.nonzero(selectv)[0]

    def forcePositiveOrientation(self):
        self.computeCentersVolumesNormals(checkOrientation=True)

    def updateWeights(self, weights=None):
        if weights is None:
            self.face_weights = np.ones(self.faces.shape[0])
        if np.isscalar(weights):
            self.face_weights = weights * np.ones(self.faces.shape[0])
        else:
            self.face_weights = np.copy(weights)
        self.vertex_weights = np.zeros(self.vertices.shape[0])
        face_per_vertex = np.zeros(self.vertices.shape[0], dtype=int)
        for k in range(self.faces.shape[0]):
            for j in range(self.faces.shape[1]):
                self.vertex_weights[self.faces[k, j]] +=\
                    self.face_weights[k] * self.volumes[k]
                face_per_vertex[self.faces[k, j]] += self.volumes[k]
        mv = self.volumes.sum()/self.volumes.shape[0]
        for k in range(face_per_vertex.shape[0]):
            if face_per_vertex[k] > 1e-10*mv:
                self.vertex_weights[k] /= face_per_vertex[k]

    def shrinkTriangles(self, ratio=0.5):
        newv = np.zeros((self.faces.shape[0]*self.faces.shape[1],
                         self.vertices.shape[1]))
        newf = np.zeros(self.faces.shape)
        for k in range(self.faces.shape[0]):
            for j in range(self.faces.shape[1]):
                newf[k, j] = self.faces.shape[1]*k + j
                newv[3*k+j, :] = self.centers[k, :] \
                    + ratio*(self.vertices[self.faces[k, j], :] - self.centers[k, :])
        neww = self.face_weights / (ratio**self.dim)
        newm = Mesh(data=(newf, newv), weights=neww, image=self.image,
                    imNames=self.imNames)
        return newm

    def rescaleUnits(self, scale):
        self.face_weights /= scale**self.dim
        self.updateVertices(self.vertices*scale)

    # modify vertices without toplogical change
    def updateVertices(self, x0, checkOrientation=False):
        self.vertices = np.copy(x0)
        if self.bdry is not None:
            self.bdry.updateVertices(x0[self.bdry_indices])
        self.computeCentersVolumesNormals(checkOrientation=checkOrientation)

    def updateImage(self, img, imNames = None):
        self.image = np.copy(img)
        self.imageDim = img.shape[1]
        if imNames is not None:
            self.imNames = imNames
        else:
            if len(self.imNames) != img.shape[1]:
                self.imNames = []
                for i in range(img.shape[1]):
                    self.imNames.append(str(i))

    def computeSparseMatrices(self):
        self.v2f0 = scp.sparse.csc_matrix((np.ones(self.faces.shape[0]),
                                          (range(self.faces.shape[0]),
                                           self.faces[:, 0]))).transpose(copy=False)
        self.v2f1 = scp.sparse.csc_matrix((np.ones(self.faces.shape[0]),
                                          (range(self.faces.shape[0]),
                                           self.faces[:, 1]))).transpose(copy=False)
        self.v2f2 = scp.sparse.csc_matrix((np.ones(self.faces.shape[0]),
                                          (range(self.faces.shape[0]),
                                           self.faces[:, 2]))).transpose(copy=False)

    def computeVertexVolume(self):
        V = self.vertices
        F = self.faces
        nv = V.shape[0]
        nf = F.shape[0]
        AV = np.zeros(nv)
        df = F.shape[1]
        vol = self.volumes/df
        for k in range(nf):
            for j in range(df):
                AV[F[k, j]] += vol[k]
        return AV

    def summary(self):
        out = f'Number of vertices: {self.vertices.shape[0]}\n'
        out += f'Number of simplices: {self.faces.shape[0]}\n'
        out += f'Min-Max-Mean volume: {self.volumes.min():.6f} {self.volumes.max():.6f} {self.volumes.mean():.6f}\n'
        out += f'Min-Max-Mean weight: {self.face_weights.min():.6f} {self.face_weights.max():.6f} {self.face_weights.mean():.6f}\n'
        out += f'Min-Max-Mean vertex weight: {self.vertex_weights.min():.6f} {self.vertex_weights.max():.6f} {self.vertex_weights.mean():.6f}\n'
        vw = self.face_weights * self.volumes
        out += f'Min-Max-Mean cells per simplex: {vw.min():.4f} {vw.max():.4f} {vw.mean():.4f}\n'

        if self.imNames is not None:
            out += f'\n{len(self.imNames)} image fields:\n'
            for nm in self.imNames:
                out += nm + ' '
            out += '\n'
        else:
            out += 'No image names\n'
        return out

    # Computes edges from vertices/faces
    def getEdges(self):
        self.edges, self.facesOfEdges, self.edgesOfFaces, bdry = get_edges_(self.faces)
        J0 = np.nonzero(bdry)[0]
        J = np.zeros(self.vertices.shape[0], dtype=int)
        for k in J0:
            for j in range(self.edges.shape[1]):
                J[self.edges[k, j]] = 1

        J = np.nonzero(J)[0]
        self.bdry_indices = J
        newindx = np.zeros(self.vertices.shape[0], dtype=int)
        newindx[J] = np.arange(len(J))
        V = self.vertices[J, :]
        F = newindx[self.edges[J0, :]]
        if self.dim == 3:
            self.bdry = Surface(surf=(F, V))
        else:
            self.bdry = Curve(curve=(F, V))

    def toPolyData(self):
        if gotVTK:
            points = vtkPoints()
            for k in range(self.vertices.shape[0]):
                if self.dim == 3:
                    points.InsertNextPoint(self.vertices[k, 0],
                                           self.vertices[k, 1],
                                           self.vertices[k, 2])
                else:
                    points.InsertNextPoint(self.vertices[k, 0],
                                           self.vertices[k, 1], 0)
            polys = vtkCellArray()
            df = self.faces.shape[1]
            for k in range(self.faces.shape[0]):
                polys.InsertNextCell(df)
                for kk in range(df):
                    polys.InsertCellPoint(self.faces[k, kk])
            polydata = vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetPolys(polys)
            return polydata
        else:
            raise Exception('Cannot run toPolyData without VTK')

    def fromUnstructuredGridData(self, g, scales=None):
        npoints = int(g.GetNumberOfPoints())
        nfaces = int(g.GetNumberOfCells())
        logging.info(f'Dimensions: {npoints}, {nfaces}, {g.GetNumberOfCells()}')
        V = np.zeros([npoints, 3])
        for kk in range(npoints):
            V[kk, :] = np.array(g.GetPoint(kk))
        F = np.zeros([nfaces, 4], dtype=int)
        for kk in range(g.GetNumberOfCells()):
            c = g.GetCell(kk)
            npt = c.GetNumberOfPoints()
            if kk == 0:
                self.dim = npt - 1
            for ll in range(npt):
                F[kk, ll] = c.GetPointId(ll)

        if self.dim == 2:
            F = F[:, :3]
            V = V[:, :2]
        if scales is None:
            self.vertices = V
        else:
            self.vertices = V * scales
        self.faces = F
        self.component = np.zeros(self.faces.shape[0], dtype=int)
        self.face_weights = np.ones(self.faces.shape[0])
        self.computeCentersVolumesNormals()

    def fromPyvista(self, g, scales=None):
        npoints = int(g.n_points)
        nfaces = int(g.n_cells)
        logging.info(f'Dimensions: {npoints}, {nfaces}')
        V = g.points
        F = np.reshape(g.cells, (nfaces, 4))
        F = F[:, 1:]

        if V.shape[1] == 3:
            if np.abs(V[:, 2]).max() < 1e-10:
                V = V[:, :2]

        self.dim = V.shape[1]
        if scales is None:
            self.vertices = V
        else:
            self.vertices = V * scales
        self.faces = F

        if 'vertex_weights' in g.point_data:
            self.vertex_weights = np.array(g.point_data['vertex_weights'])
        else:
            self.vertex_weights = np.ones(self.vertices.shape[0])

        # Image
        n_im = len(g.cell_data)
        if 'component' in g.cell_data:
            self.component = np.array(g['component'])
            n_im -= 1
        else:
            self.component = np.zeros(self.faces.shape[0], dtype=int)
        if 'weights' in g.cell_data:
            self.face_weights = np.array(g['weights'])
            n_im -= 1
        else:
            self.face_weights = np.ones(self.faces.shape[0])
        if 'image_magnitude' in g.cell_data:
            n_im -= 1
        if 'volumes' in g.cell_data:
            n_im -= 1

        self.computeCentersVolumesNormals()

        self.imNames = []
        self.imageDim = n_im
        self.image = np.zeros((nfaces, n_im))
        k = 0
        for nm in g.cell_data:
            if nm not in ('component', 'weights', 'image_magnitude',
                          'volumes'):
                self.imNames.append(nm)
                self.image[:, k] = np.array(g.cell_data[nm])

    def flipFaces(self):
        if self.dim == 2:
            self.faces = self.faces[:, [0, 2, 1]]
        else:
            self.faces = self.faces[:, [0, 1, 3, 2]]
        self.computeCentersVolumesNormals()

    def meshVolume(self):
        # Computes mesh volume
        return self.volumes.sum()

    def toImage(self, resolution=1, background=0, margin=10, index=None,
                bounds=None):
        interp = LinearNDInterpolator(self.centers,
                                      self.face_weights[:, None]*self.image,
                                      fill_value=background)

        if bounds is None:
            imin = self.vertices.min(axis=0) - margin
            imax = self.vertices.max(axis=0) + margin
        else:
            imin = bounds[0]
            imax = bounds[1]

        imDim = np.ceil((imax - imin)/resolution).astype(int)

        if np.prod(imDim) * self.image.shape[1] > 1e7:
            print('Image to big to create', imDim, self.image.shape[1])
            return

        if self.dim > 3:
            print('Image cannot be created: dimensions too large', self.dim)
            return

        X = np.linspace(imin[0], imax[0], imDim[0])
        if self.dim == 2:
            Y = np.linspace(imin[1], imax[1], imDim[1])
            X, Y = np.meshgrid(X, Y)
            outIm = interp(X, Y)
            outIm = np.flip(outIm, axis=0)
        elif self.dim == 3:
            Y = np.linspace(imin[1] - margin, imax[1] + margin, imDim[1])
            Z = np.linspace(imin[2] - margin, imax[2] + margin, imDim[2])
            X, Y, Z = np.meshgrid(X, Y, Z)
            outIm = interp(X, Y, Z)
        else:
            outIm = interp(X)

        if index is None:
            return outIm
        else:
            return outIm[..., index]

    # Saves in .vtk format
    def saveVTK(self, fileName, vtk_fields=()):
        F = self.faces
        V = self.vertices

        with open(fileName, 'w') as fvtkout:
            fvtkout.write('# vtk DataFile Version 3.0\nMesh Data\nASCII\nDATASET UNSTRUCTURED_GRID\n')
            fvtkout.write('\nPOINTS {0: d} float'.format(V.shape[0]))
            for ll in range(V.shape[0]):
                fvtkout.write('\n')
                for kk in range(self.dim):
                    fvtkout.write(f'{V[ll, kk]: f} ')
                if self.dim == 2:
                    fvtkout.write('0')
            fvtkout.write('\nCELLS {0:d} {1:d}'.format(F.shape[0],
                                                       (self.dim+2)*F.shape[0]))
            for ll in range(F.shape[0]):
                fvtkout.write(f'\n{self.dim+1} ')
                for kk in range(self.dim+1):
                    fvtkout.write(f'{F[ll, kk]: d} ')

            if self.dim == 2:
                ctype = 5
            else:
                ctype = 10
            fvtkout.write('\nCELL_TYPES {0:d}'.format(F.shape[0]))
            for ll in range(F.shape[0]):
                fvtkout.write(f'\n{ctype} ')

            cell_data = False
            point_data = False
            for v in vtk_fields:
                if v.data_type == 'CELL_DATA':
                    cell_data = True
                    if 'image' not in v.fields.keys():
                        v.fields['image'] = {}
                        for k, nm in enumerate(self.imNames):
                            v.fields['image'][nm] = self.image[:, [k]]
                        v.scalars['image_magnitude'] = self.image.sum(axis=1)
                    if 'weights' not in v.scalars.keys():
                        v.scalars['weights'] = self.face_weights
                    if 'volumes' not in v.scalars.keys():
                        v.scalars['volumes'] = self.volumes
                    v.write(fvtkout)
                elif v.data_type == 'POINT_DATA':
                    point_data = True
                    if 'vertex_weights' not in v.scalars.keys():
                        v.scalars['vertex_weights'] = self.vertex_weights
                    v.write(fvtkout)
            if not cell_data:
                fld = {}
                for k, nm in enumerate(self.imNames):
                    fld[nm] = self.image[:, [k]]
                v = vtkFields('CELL_DATA', self.faces.shape[0],
                              fields={'image': fld},
                              scalars={'weights': self.face_weights,
                                       'volumes': self.volumes,
                                       'image_magnitude': self.image.sum(axis=1)
                                       })
                v.write(fvtkout)
            if not point_data:
                v = vtkFields('POINT_DATA', self.vertices.shape[0],
                              scalars={'vertex_weights': self.vertex_weights})
                v.write(fvtkout)

    def save(self, fileName, vtkFields=()):
        self.saveVTK(fileName, vtk_fields=vtkFields)

    # Reads .vtk file
    def readVTK(self, fileName):
        mesh = pv.read(fileName)
        self.fromPyvista(mesh)

    def readVTK_legacy(self, fileName):
        if gotVTK:
            u = vtkUnstructuredGridReader()
            u.ReadAllVectorsOn()
            u.ReadAllScalarsOn()
            u.ReadAllFieldsOn()
            u.SetFileName(fileName)
            u.Update()
            v = u.GetOutput()
            w = v.GetCellData().GetScalars('weights')
            lab = v.GetCellData().GetScalars('labels')
            image = v.GetCellData().GetArray('image')
            image2 = v.GetFieldData()
            print('vtk', image, image2, image2.GetNumberOfTuples())
            self.fromUnstructuredGridData(v)
            nfaces = self.faces.shape[0]

            if lab:
                Lab = np.zeros(nfaces, dtype=int)
                for kk in range(nfaces):
                    Lab[kk] = lab.GetTuple(kk)[0]
            else:
                Lab = np.zeros(nfaces, dtype=int)

            if image:
                nt = image.GetNumberOfTuples()
                if nt == nfaces:
                    self.imageDim = image.GetNumberOfComponents()
                    IM = np.zeros((nfaces, self.imageDim))
                    kj = 0
                    for k in range(nfaces):
                        for j in range(self.imageDim):
                            IM[k, j] = image.GetValue(kj)
                            kj += 1
                else:
                    IM = np.ones(nfaces)
            else:
                IM = np.ones(nfaces)
            if w:
                W = np.zeros(nfaces)
                for kk in range(nfaces):
                    W[kk] = w.GetTuple(kk)[0]
            else:
                W = np.ones(nfaces)

            self.face_weights = W
            self.image = IM
            self.computeCentersVolumesNormals()
            self.imNames = None
            self.component = Lab
        else:
            raise Exception('Cannot run readVTK without VTK')

    def concatenate(self, fvl):
        nv = 0
        nf = 0
        for fv in fvl:
            nv += fv.vertices.shape[0]
            nf += fv.faces.shape[0]
        self.dim = fvl[0].dim
        self.vertices = np.zeros([nv, self.dim])
        self.face_weights = np.zeros(nv)
        self.faces = np.zeros([nf, self.dim+1], dtype='int')
        self.component = np.zeros(nf, dtype='int')

        nv0 = 0
        nf0 = 0
        c = 0
        for fv in fvl:
            nv = nv0 + fv.vertices.shape[0]
            nf = nf0 + fv.faces.shape[0]
            self.vertices[nv0:nv, :] = fv.vertices
            self.face_weights[nv0:nv] = fv.face_weights
            self.faces[nf0:nf, :] = fv.faces + nv0
            self.component[nf0:nf] = fv.component + c
            nv0 = nv
            nf0 = nf
            c = self.component[:nf].max() + 1
        self.computeCentersVolumesNormals()


def split_prism(bottom, top):
    # bottom: v1, v2, v3; top:v4, v5, v6
    all = np.concatenate((bottom, top))
    j = np.argmin(all)
    if j == 0:
        i1, i2, i3, i4, i5, i6 = (0, 1, 2, 3, 4, 5)
    elif j == 1:
        i1, i2, i3, i4, i5, i6 = (1, 2, 0, 4, 5, 3)
    elif j == 2:
        i1, i2, i3, i4, i5, i6 = (2, 0, 1, 5, 3, 4)
    elif j == 3:
        i1, i2, i3, i4, i5, i6 = (3, 5, 4, 0, 2, 1)
    elif j == 4:
        i1, i2, i3, i4, i5, i6 = (4, 3, 5, 1, 0, 2)
    else:
        i1, i2, i3, i4, i5, i6 = (5, 4, 3, 2, 1, 0)

    res = np.zeros((3, 4), dtype=int)
    if min(all[i2], all[i6]) < min(all[i3], all[i5]):
        res[0, :] = (all[i1], all[i2], all[i3], all[i6])
        res[1, :] = (all[i1], all[i2], all[i6], all[i5])
        res[2, :] = (all[i1], all[i5], all[i6], all[i4])
    else:
        res[0, :] = (all[i1], all[i2], all[i3], all[i5])
        res[1, :] = (all[i1], all[i5], all[i3], all[i6])
        res[2, :] = (all[i1], all[i5], all[i6], all[i4])

    return res


def build3DMeshFromLayers(layers, triangles, image=None, weights=None):
    nl = layers.shape[0]
    nv = layers.shape[1]
    nf = triangles.shape[0]
    vert = np.zeros((nl*nv, 3))
    faces = np.zeros((3*(nl-1)*nf, 4), dtype=int)
    for l in range(nl):
        vert[l*nv:(l+1)*nv, :] = layers[l, :, :]
    if image is not None:
        im3D = np.zeros((3*(nl-1)*nf, image.shape[1]))
    else:
        im3D = None
    if weights is not None:
        w3D = np.zeros(3*(nl-1)*nf)
    else:
        w3D = None
    kf = 0
    for l in range(nl-1):
        for j in range(nf):
            bottom = triangles[j, :] + l*nv
            top = triangles[j, :] + (l + 1)*nv
            faces[kf:kf + 3, :] = split_prism(bottom, top)
            if image is not None:
                im3D[kf:kf+3, :] = image[j, :]
            if weights is not None:
                w3D[kf:kf+3] = weights[j]
            kf += 3

    res = Mesh(data=(faces, vert), image=im3D, weights=w3D)
    return res


def thick2D(mesh, delta=1., z0=0.):
    layers = np.zeros((2, mesh.vertices.shape[0], 3))
    layers[0, :, :2] = mesh.vertices
    layers[0, :, 2] = z0
    layers[1, :, :2] = mesh.vertices
    layers[1, :, 2] = z0 + delta
    return build3DMeshFromLayers(layers, mesh.faces, image=mesh.image,
                                 weights=mesh.face_weights)


@jit(nopython=True)
def count__(g, sps, inv):
    g_ = g
    logging.info(f'{inv.shape}')
    for k in range(sps.shape[0]):
        g_[sps[k], inv[k]] += 1
    return g_


@jit(nopython=True)
def select_faces__(g, points, simplices, threshold=1e-10):
    keepface = np.nonzero(np.fabs(g).sum(axis=1) > threshold)[0]
    newf_ = np.zeros((keepface.shape[0], simplices.shape[1]), dtype=int64)
    for k in range(keepface.shape[0]):
        for j in range(simplices.shape[1]):
            newf_[k, j] = simplices[keepface[k], j]
    keepvert = np.zeros(points.shape[0], dtype=int64)
    for k in range(newf_.shape[0]):
        for j in range(simplices.shape[1]):
            keepvert[newf_[k, j]] = 1
    keepvert = np.nonzero(keepvert)[0]
    newv = np.zeros((keepvert.shape[0], points.shape[1]))
    for k in range(keepvert.shape[0]):
        for j in range(points.shape[1]):
            newv[k, j] = points[keepvert[k], j]
    newI = - np.ones(points.shape[0], dtype=int64)
    for k in range(keepvert.shape[0]):
        newI[keepvert[k]] = k
    newf = np.zeros(newf_.shape, dtype=int64)
    for k in range(newf_.shape[0]):
        for j in range(newf_.shape[1]):
            newf[k, j] = newI[newf_[k, j]]
    # newf = newI[newf]
    g = np.copy(g[keepface, :])
    return newv, newf, g, keepface


def select_faces2__(points, simplices, threshold=1e-10, g=None,
                    removeBackground=True, small=0):
    edges, facesOfEdges, edgesOfFaces, bdry = get_edges_(simplices)
    if g is None:
        g = np.ones((simplices.shape[0], 1))
    gsum = np.fabs(g).sum(axis=1) > threshold
    N = simplices.shape[0]
    A = lil_matrix((N, N), dtype=int)
    for k in range(facesOfEdges.shape[0]):
        f0 = facesOfEdges[k, 0]
        f1 = facesOfEdges[k, 1]
        if (f0 >= 0 and f1 >= 0 and
           ((gsum[f0] and gsum[f1]) or (not gsum[f0] and not gsum[f1]))):
            A[f0, f1] = 1
            A[f1, f0] = 1
    nc, labels = connected_components(A, directed=False)
    rd = np.random.permutation(nc)
    labels = rd[labels]
    logging.info(f'found {nc} connected components')
    newv, newf, g, keepface = subselect_(points, simplices, g, nc, labels,
                                         small, removeBackground)
    return newv, newf, g, keepface


@jit(nopython=True)
def subselect_(points, simplices, g, nc, labels, small, removeBackground):
    centers = np.zeros((simplices.shape[0], points.shape[1]))
    for j in range(simplices.shape[1]):
        centers += points[simplices[:, j], :]
    centers /= simplices.shape[1]
    if removeBackground:
        diag = centers.sum(axis=1)
        k = np.argmin(diag)
        background = labels[k]
    else:
        background = -1

    nperlab = np.zeros(nc)
    for j in range(labels.shape[0]):
        nperlab[labels[j]] += 1
    for j in range(labels.shape[0]):
        if nperlab[labels[j]] < small:
            labels[j] = background

    keepface = np.nonzero(np.fabs(labels - background))[0]
    newf_ = np.zeros((keepface.shape[0], simplices.shape[1]), dtype=int64)
    for k in range(keepface.shape[0]):
        for j in range(simplices.shape[1]):
            newf_[k, j] = simplices[keepface[k], j]
    keepvert = np.zeros(points.shape[0], dtype=int64)
    for k in range(newf_.shape[0]):
        for j in range(simplices.shape[1]):
            keepvert[newf_[k, j]] = 1
    keepvert = np.nonzero(keepvert)[0]
    newv = np.zeros((keepvert.shape[0], points.shape[1]))
    for k in range(keepvert.shape[0]):
        for j in range(points.shape[1]):
            newv[k, j] = points[keepvert[k], j]
    newI = - np.ones(points.shape[0], dtype=int64)
    for k in range(keepvert.shape[0]):
        newI[keepvert[k]] = k
    newf = np.zeros(newf_.shape, dtype=int64)
    for k in range(newf_.shape[0]):
        for j in range(newf_.shape[1]):
            newf[k, j] = newI[newf_[k, j]]
    g = np.copy(g[keepface, :])
    return newv, newf, g, keepface


def buildMeshFromFullListHR(x0, y0, genes, radius=20, threshold=1e-10):
    dx = (x0.max() - x0.min())/20
    minx = x0.min() - dx
    maxx = x0.max() + dx
    dy = (y0.max() - y0.min())/20
    miny = y0.min() - dy
    maxy = y0.max() + dy
    ugenes, inv = np.unique(genes, return_inverse=True)
    logging.info(f'{x0.shape[0]} input points, {ugenes.shape[0]} unique genes')

    spacing = radius/2

    ul = np.array((minx, miny))
    ur = np.array((maxx, miny))
    ll = np.array((minx, maxy))
    v0 = ur - ul
    v1 = ll - ul

    nv0 = np.sqrt((v0 ** 2).sum())
    nv1 = np.sqrt((v1 ** 2).sum())
    npt0 = int(np.ceil(nv0 / spacing))
    npt1 = int(np.ceil(nv1 / spacing))

    t0 = np.linspace(0, 1, npt0)

    t1 = np.linspace(0, 1, npt1)
    x, y = np.meshgrid(t0, t1)
    x = np.ravel(x)
    y = np.ravel(y)
    pts = ul[None, :] + x[:, None] * v0[None, :] + y[:, None] * v1[None, :]

    tri = Delaunay(pts)
    vert = np.zeros((tri.points.shape[0], 3))
    vert[:, :2] = tri.points
    centers = np.zeros((x0.shape[0], 2))
    centers[:, 0] = x0
    centers[:, 1] = y0

    g = np.zeros((tri.simplices.shape[0], ugenes.shape[0]))
    sps = tri.find_simplex(centers)
    logging.info('count')
    g = count__(g, sps, inv)
    logging.info('counts done')
    logging.info(f'Creating {centers.shape[0]} faces')

    logging.info('face selection')
    newv, newf, newg, foo = select_faces2__(tri.points, tri.simplices,
                                            threshold=threshold, g=g)

    logging.info(f'mesh construction: {newv.shape[0]} vertices {newf.shape[0]} faces')
    fv0 = Mesh(data=(newf, newv), image=newg)
    return fv0


@jit(nopython=True)
def buildImageFromFullListHR(x0, y0, genes, radius=20.):
    dx = (x0.max() - x0.min())/20
    minx = x0.min() - dx
    maxx = x0.max() + dx
    dy = (y0.max() - y0.min())/20
    miny = y0.min() - dy
    maxy = y0.max() + dy
    ng = genes.max() + 1

    spacing = radius/2

    ul = np.array((minx, miny))
    ur = np.array((maxx, miny))
    ll = np.array((minx, maxy))
    v0 = ur - ul
    v1 = ll - ul

    nv0 = np.sqrt((v0 ** 2).sum())
    nv1 = np.sqrt((v1 ** 2).sum())
    npt0 = int(np.ceil(nv0 / spacing))
    npt1 = int(np.ceil(nv1 / spacing))

    img = np.zeros((npt0, npt1, ng))
    ik = np.floor((x0 - minx)/spacing).astype(int64)
    jk = np.floor((y0 - miny)/spacing).astype(int64)

    for k in range(x0.shape[0]):
        img[ik[k], jk[k], genes[k]] += 1

    return img, (minx, miny, spacing)


def buildMeshFromFullList(x0, y0, genes, resolution=100, HRradius=20,
                          HRthreshold=0.5):
    logging.info('Building High-resolution mesh')
    fvHR = buildMeshFromFullListHR(x0, y0, genes, radius=HRradius,
                                   threshold=HRthreshold)
    if np.isscalar(resolution):
        resolution = (resolution,)
    fv0 = [fvHR]
    for r in resolution:
        logging.info(f'Buiding meshes at resolution {r:.0f}')
        fv0.append(buildMeshFromCentersCounts(fvHR.centers, fvHR.image,
                                              resolution=r,
                                              weights=fvHR.volumes))
    return fv0


def get_cubes(dim, pts, npt, centers):
    if dim == 2:
        cube = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype='float')
        tri = Delaunay(cube, qhull_options='Qc Qz')
        ns0 = tri.simplices.shape[0]
        ncubes = np.prod(npt - 1)
        cubes = np.zeros((ncubes, 4), dtype=int)
        u1 = 1
        u0 = npt[1]
        template = np.array([0, u1, u0, u1 + u0], dtype=int)
        c = 0
        for i0 in range(npt[0]-1):
            for i1 in range(npt[1]-1):
                cubes[c, :] = i0*u0 + i1 + template
                c += 1

        dx = pts[npt[1], 0] - pts[0, 0]
        dy = pts[1, 1] - pts[0, 1]
        alld = np.array([dx, dy])
        rk = np.floor((centers-pts[0, :])/alld[None, :]).astype(int)
        resid = (centers-pts[0, :])/alld[None, :] - rk
        sp0 = tri.find_simplex(resid)
        sp = ns0 * (rk[:, 0] * (u0-1) + rk[:, 1] * u1) + sp0

        faces = np.zeros((ns0*ncubes, 3), dtype=int)
        cc = np.arange(ncubes, dtype=int)
        for j in range(tri.simplices.shape[0]):
            x0 = tri.points[tri.simplices[j, 0], :]
            x1 = tri.points[tri.simplices[j, 1], :]
            x2 = tri.points[tri.simplices[j, 2], :]
            vol = np.cross(x1 - x0, x2 - x0)
            if vol < 0:
                k = tri.simplices[j, -2]
                tri.simplices[j, -2] = tri.simplices[j, -1]
                tri.simplices[j, -1] = k
        for j in range(ns0):
            for q in range(dim+1):
                faces[ns0 * cc + j, q] = cubes[cc, tri.simplices[j, q]]
    else:
        cube = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1],
                         [1, 0, 1], [1, 1, 0], [1, 1, 1]],
                        dtype='float')
        tri = Delaunay(cube, qhull_options='Qc Qz')
        ns0 = tri.simplices.shape[0]
        ncubes = np.prod(npt-1)
        cubes = np.zeros((ncubes, 8), dtype=int)
        u2 = 1
        u1 = npt[2]
        u0 = npt[1]*npt[2]
        uu1 = npt[2]-1
        uu0 = (npt[1]-1)*(npt[2]-1)
        template = np.array([0, u2, u1, u0, u2+u1, u2+u0, u1+u0, u2+u1+u0],
                            dtype=int)
        c = 0
        for i0 in range(npt[0]-1):
            for i1 in range(npt[1]-1):
                for i2 in range(npt[2]-1):
                    cubes[c, :] = i0*u0 + i1*u1 + i2 + template
                    c += 1

        dx = pts[npt[1] * npt[2], 0] - pts[0, 0]
        dy = pts[npt[2], 1] - pts[0, 1]
        dz = pts[1, 2] - pts[0, 2]
        alld = np.array([dx, dy, dz])
        rk = np.floor((centers-pts[0, :])/alld[None, :]).astype(int)
        resid = (centers-pts[0, :])/alld[None, :] - rk
        sp0 = tri.find_simplex(resid)
        u2 = 1
        sp = ns0 * (rk[:, 0] * uu0 + rk[:, 1] * uu1 + rk[:, 2] * u2) + sp0

        faces = np.zeros((ns0*ncubes, 4), dtype=int)
        cc = np.arange(ncubes, dtype=int)
        for j in range(ns0):
            x0 = tri.points[tri.simplices[j, 0], :]
            x1 = tri.points[tri.simplices[j, 1], :]
            x2 = tri.points[tri.simplices[j, 2], :]
            x3 = tri.points[tri.simplices[j, 3], :]
            vol = ((x1-x0)*np.cross(x2 - x0, x3 - x0)).sum()
            if vol < 0:
                k = tri.simplices[j, -2]
                tri.simplices[j, -2] = tri.simplices[j, -1]
                tri.simplices[j, -1] = k
        for j in range(ns0):
            for q in range(dim+1):
                faces[ns0 * cc + j, q] = cubes[cc, tri.simplices[j, q]]

    return faces, sp


# Builds a mesh structure from cell centers and multivariate counts
# If radius is not none: creates small triangles around each center
def buildMeshFromCentersCounts(centers, cts, resolution=100, radius=None,
                               weights=None, minCounts=1e-10,
                               minComponentSize=1, threshold=None):

    if threshold is not None:
        logging.warning('buildMeshFromCentersCounts: threshold is deprecated. Use minCounts instead')
        minCounts = threshold

    # Create extended domain
    dim = centers.shape[1]
    logging.info(f'dim = {dim} {centers.shape} {cts.shape}')
    deltax = np.zeros(dim)
    minx = np.zeros(dim)
    maxx = np.zeros(dim)
    for i in range(dim):
        deltax[i] = (centers[:, i].max() - centers[:, i].min())/20
        minx[i] = centers[:, i].min() - deltax[i]
        maxx[i] = centers[:, i].max() + deltax[i]

    if radius is None:
        if np.isscalar(resolution):
            spacing = [resolution] * dim
        else:
            spacing = resolution
    else:
        spacing = [radius/2] * dim

    if weights is None:
        weights = np.ones(centers.shape[0])

    # building grid
    v = np.zeros((dim, dim))
    npt = np.zeros(dim, dtype=int)
    t = []
    for i in range(dim):
        v[i, i] = maxx[i] - minx[i]
        npt[i] = int(np.ceil(v[i, i] / spacing[i]))
        t.append(np.linspace(0, 1, npt[i]))

    # ntotal = npt.prod()
    allts = np.copy(t[0])
    for i in range(1, dim):
        allts = np.column_stack((np.repeat(allts, t[i].shape[0], axis=0),
                                 np.tile(t[i], allts.shape[0])))
    pts = np.zeros(allts.shape)
    pts[:, :] = minx[None, :]
    for i in range(dim):
        pts += np.outer(allts[:, i], v[i, :])
    logging.info(f'{pts.shape}')

    faces, sp = get_cubes(dim, pts, npt, centers)

    vert = np.zeros((pts.shape[0], 3))
    vert[:, :dim] = pts

    logging.info(f'Grid with {pts.shape[0]} points; {centers.shape[0]}')
    csc = csc_matrix((np.ones(centers.shape[0]), sp, np.arange(centers.shape[0]+1)),
                     shape=(faces.shape[0], centers.shape[0]))
    g = csc @ cts
    nc = csc.sum(axis=1)
    wgts = csc @ weights
    print(g.shape, csc.shape, nc.shape)

    g /= np.maximum(1e-10, nc)
    if not isinstance(g, np.ndarray):
        g = g.toarray()

    # First cleaning pass: remove background
    logging.info('first selection')
    newv, newf, newg, keepface = select_faces2__(pts, faces,
                                                 threshold=minCounts,
                                                 g=g, removeBackground=True,
                                                 small=0)
    wgts = wgts[keepface]

    # Second cleaning pass: remove small components
    logging.info('second selection')
    newv, newf, newg2, keepface = select_faces2__(newv, newf,
                                                  threshold=minCounts,
                                                  removeBackground=False,
                                                  small=minComponentSize)
    wgts = wgts[keepface]
    newg = np.copy(newg[keepface, :])
    logging.info(f'Mesh with {newv.shape[0]} vertices and {newf.shape[0]} faces; {wgts.sum()} cells')
    fv0 = Mesh(data=(newf, newv), image=newg, weights=wgts)
    fv0.updateWeights(wgts/fv0.volumes)
    return fv0


def buildMeshFromImage(img, resolution=25, bounding_box=None, histogram=False,
                       minCounts=1e-10, minComponentSize=1):

    if bounding_box is None:
        bounding_box = []
        for j in range(img.ndim):
            bounding_box += [0, img.shape[j]]
    xi = np.linspace(bounding_box[0], bounding_box[1], img.shape[0])
    yi = np.linspace(bounding_box[2], bounding_box[3], img.shape[1])
    if img.ndim == 3:
        ndim = 3
        zi = np.linspace(bounding_box[4], bounding_box[5], img.shape[2])
        (x, y, z) = np.meshgrid(yi, xi, zi)
    else:
        ndim = 2
        (x, y) = np.meshgrid(yi, xi)
    if histogram:
        labels = np.unique(img)
        cts = img.reshape((x.size, 1)) == labels[None, :]
    else:
        cts = img.reshape((x.size, 1))

    centers = np.zeros((x.size, ndim))
    centers[:, 0] = np.ravel(x)
    centers[:, 1] = np.ravel(y)
    if ndim == 3:
        centers[:, 2] = np.ravel(z)
    return buildMeshFromCentersCounts(centers, cts, resolution=resolution,
                                      minCounts=minCounts,
                                      minComponentSize=minComponentSize)


def buildMeshFromImageData(img, geneSet=None, resolution=25, radius=None,
                           bounding_box=(0, 1, 0, 1, 0, 1)):

    xi = np.linspace(bounding_box[0], bounding_box[1], img.shape[0])
    yi = np.linspace(bounding_box[2], bounding_box[3], img.shape[1])
    if img.ndim == 4:
        ndim = 3
        zi = np.linspace(bounding_box[4], bounding_box[5], img.shape[2])
        (x, y, z) = np.meshgrid(yi, xi, zi)
    else:
        ndim = 2
        (x, y) = np.meshgrid(yi, xi)
    ng = img.shape[ndim]
    cts = img.reshape((x.size, ng))
    if geneSet is not None:
        img = img[:, geneSet]
    centers = np.zeros((x.size, ndim))
    centers[:, 0] = np.ravel(x)
    centers[:, 1] = np.ravel(y)
    if ndim == 3:
        centers[:, 2] = np.ravel(z)
    return buildMeshFromCentersCounts(centers, cts, resolution=resolution,
                                      radius=radius)


def buildMeshFromMerfishData(fileCounts, fileData, geneSet=None, resolution=100,
                             radius=None,
                             coordinate_columns=('center_x', 'center_y')):
    counts = pd.read_csv(fileCounts)
    if geneSet is not None:
        counts = counts.loc[:, geneSet]
    data = pd.read_csv(fileData)
    centers = np.zeros((data.shape[0], 2))
    centers[:, 0] = data.loc[:, coordinate_columns[0]]
    centers[:, 1] = data.loc[:, coordinate_columns[1]]
    cts = counts.to_numpy()
    return buildMeshFromCentersCounts(centers, cts, resolution=resolution,
                                      radius=radius)
