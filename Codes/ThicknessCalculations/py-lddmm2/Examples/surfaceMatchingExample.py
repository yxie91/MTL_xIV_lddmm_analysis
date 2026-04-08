import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sys import path as sys_path
import pyvista as pv

from base import surfaces
from base import loggingUtils
from base.surfaces import Surface
from base.kernelFunctions import Kernel
from base.affineRegistration import rigidRegistration
from base.surfaceMatching import SurfaceMatching
from base.secondOrderSurfaceMatching import SecondOrderSurfaceMatching
from base.surfaceExamples import HalfSphere, Heart, Sphere_pygal as Sphere
import pykeops


sys_path.append('..')
sys_path.append('../base')
if 'DISPLAY' in os.environ:
    matplotlib.use('qt5Agg')
else:
    matplotlib.use("Agg")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
pykeops.clean_pykeops()
plt.ion()

model = 'Balls'
errorType = 'varifold'

secondOrder = False

if secondOrder:
    typeCost = 'LDDMM'
    order = '_SO_'
else:
    typeCost = 'LDDMM'
    order = ''


def compute(model):
    loggingUtils.setup_default_logging('../Output', stdOutput=True)
    if typeCost == 'LDDMM':
        sigmaKernel = 5
        internalCost = None
        internalWeight = 0.
        regweight = 1.
    else:
        sigmaKernel = .5
        internalWeight = 1.
        regweight = 0.1
        internalCost = [['elastic', 50]]
        # internalCost = [['elastic', 100.], ['displacement', 10.]]

    sigmaDist = 10.
    sigmaError = 1.
    # internalCost = 'h1'
    landmarks = None
    if model == 'Balls':
        s2 = np.sqrt(2)
        def eval1(x, y, z):
            return 0.06 - ((x - .50) ** 2 + 0.5 * y ** 4 + z ** 2)


        img = pv.ImageData(dimensions=(201, 201, 201), origin=(-1, -1, -1), spacing=(0.05, 0.05, 0.05))
        x, y, z = img.points.T
        values = eval1(x,y,z)
        mesh = img.contour([0], values, method='marching_cubes')
        # s1 = pv.Sphere(center=(.5, 0, 0), radius=np.sqrt(0.06))
        # v = np.array(s1.points)
        # v[:, 1] = np.sign(v[:,1]) * np.sqrt(np.abs(v[:,1])) * (2**.25)
        # s1 = pv.PolyData(v, s1.faces)
        # s1 = s1.subdivide(2)
        if mesh.points.shape[0] > 1000:
            mesh = mesh.decimate(1-1000 / mesh.points.shape[0])

        class Surf1:
            def f(self, x):
                return 0.06 - ((x[0] - .50) ** 2 + 0.5 * x[1] ** 4 + x[2] ** 2)

            def grad(self, x):
                res = np.zeros(x.shape)
                res[0] = -2*(x[0] - .5)
                res[1] = -2*x[1]**3
                res[2] = -2*x[2]
                return res


        points = np.array(mesh.points)
        faces = np.reshape(mesh.faces, (mesh.faces.size // 4, 4))
        faces = faces[:, 1:]
        # ms_ = Mesh(points, faces)
        # ms = MeshSet()
        # ms.create_implicit_surface(expr="0.06 - ((x - .50) * (x - .50) + 0.5 * y*y*y*y + z*z)")
        # # ms.add_mesh(ms_, "0")
        # # ms.apply_coord_laplacian_smoothing_surface_preserving(iterations=2)
        # ms_ = ms.current_mesh()
        # faces = ms_.face_matrix()
        # points = ms_.vertex_matrix()
        # print(points.shape)
        # print(faces.shape)
        # print(faces)

        fv1 = Surface(surf = (faces, points))
        fv1.saveVTK('ball_test.vtk')
        # fv1.from_pv(mesh, targetSize=None)

        # fv1 = Sphere(center=(.5, 0, 0), radius=np.sqrt(0.06))
        # v = fv1.vertices
        # v[:, 1] = np.sign(v[:,1]) * np.sqrt(np.abs(v[:,1])) * (2**.25)
        # fv1.updateVertices(v)

        s1 = pv.Sphere(center=(.7, 0, 0), radius=np.sqrt(0.05))
        s2 = pv.Sphere(center=(.5, 0, 0), radius=np.sqrt(0.05))
        s1s2 = s1.boolean_union(s2)
        if s1s2.points.shape[0] > 1000:
            s1s2 = s1s2.decimate(1-1000/s1s2.points.shape[0])
        # geo = CSGeometry()
        # s1 = Sphere_ng(Pnt(0.7, 0, 0), np.sqrt(0.05))
        # s2 = Sphere_ng(Pnt(0.5, 0, 0), np.sqrt(0.02))
        # geo.Add(s1 + s2)
        # mesh = Mesh(geo.GenerateMesh(maxh=0.01, perfstepsend=MeshingStep.MESHSURFACE))
        fv2 = Surface()
        fv2.from_pv(s1s2, targetSize=None)



        # d = Ball1()
        # mesh = pygalmesh.generate_surface_mesh(d, max_facet_distance=0.05, min_facet_angle=30.0,
        #                                        max_radius_surface_delaunay_ball=0.05, verbose=False)
        # fv1 = Surface(surf=(mesh.cells[0].data, mesh.points))
        fv1.updateVertices(fv1.vertices*100)

        # d = Ball2()
        # mesh = pygalmesh.generate_surface_mesh(d, max_facet_distance=0.05, min_facet_angle=30.0,
        #                                        max_radius_surface_delaunay_ball=0.05, verbose=False)
        # fv2 = Surface(surf=(mesh.cells[0].data, mesh.points))
        fv2.updateVertices(fv2.vertices*100)

        ftemp = fv1
        ftarg = fv2
        landmarks = (fv1.vertices[0:25, :], fv2.vertices[0:25, :], 10.)
        #errorType = 'LandmarksOnly'
        #internalCost = None
    elif model=='Hearts':
        # [x,y,z] = np.mgrid[0:200, 0:200, 0:200]/100.
        # ay = np.fabs(y-1)
        # az = np.fabs(z-1)
        # ax = np.fabs(x-0.5)
        # s2 = np.sqrt(2)
        # c1 = np.sqrt(0.06)
        # c2 = np.sqrt(0.045)
        # c3 = 0.1

        # I1 = np.minimum(c1**2 - (ax**2 + 0.5*ay**2 + az**2), np.minimum((ax**2 + 0.5*ay**2 + az**2)-c2**2, 1+c3-y)) 
        # fv1 = Surface()
        # fv1.Isosurface(I1, value = 0, target=1000, scales=[1, 1, 1], smooth=0.01)

        # p = 1.75
        # s1 = 1.2
        # s2 = 1.4
        # I1 = np.minimum(c1**p/s1 - ((ax**p + 0.5*ay**p + az**p)), np.minimum((s2*ax**p + s2*0.5*ay**p + s2*az**p)-c2**p/s1, 1+c3/s1-y))  
        # fv3 = Surface()
        # fv3.Isosurface(I1, value = 0, target=1000, scales=[1, 1, 1], smooth=0.01)

        # vrt = fv3.vertices
        # vrt[:, 1] += 15 - 15 / s1
        # fv3.updateVertices(vrt)

        # ftemp = fv1
        # ftarg = fv3
        ftarg = Heart(zoom=100, targetSize=10000)
        ftemp = Heart(p=1.5, scales=(1.1, 1.75), zoom = 100, targetSize=10000)
        sigmaKernel = 5.
        internalWeight = 10
        sigmaError = 1.
        sigmaDist = 10
        regweight = 1.
    elif model=='HalfSphere':
        ftemp = HalfSphere(radius=5)
        ftarg = HalfSphere(radius=10)
        sigmaDist = 2.5
        sigmaError = 0.1
    elif model=='KCrane':
        ftemp = surfaces.Surface(surf='../testData/Surfaces/KCrane/blub_triangulated_reduced.obj')
        ftarg = surfaces.Surface(surf='../testData/Surfaces/KCrane/spot_triangulated_reduced.obj')
        R0, T0 = rigidRegistration(surfaces = (ftarg.vertices, ftemp.vertices),  rotWeight=0., verb=False, temperature=10., annealing=True)
        ftarg.updateVertices(np.dot(ftarg.vertices, R0.T) + T0)
        sigmaKernel = 0.5
        sigmaDist = 5.
        sigmaError = 0.01
        internalWeight = 10.
    elif model=='snake':
        M=100
        [x,y,z] = np.mgrid[0:2*M, 0:2*M, 0:2*M]/float(M)
        x = x-1
        y = y-1
        z = z-1
        t = np.arange(-0.5, 0.5, 0.01)

        r = .3
        c = .95
        delta = 0.05
        h = 0.25
        f1 = np.zeros((t.shape[0],3))
        f1[:,0] = r*np.cos(2*np.pi*c*t) -r
        f1[:,1] = r*np.sin(2*np.pi*c*t)
        fig = plt.figure(4)
        f1[:,2] = h*t
        fig.clf()
        ax = fig.gca(projection='3d')
        ax.plot(f1[:,0], f1[:,1], f1[:,2])

        f2 = np.zeros((t.shape[0],3))
        f2[:,0] = r*np.cos(2*np.pi*c*t)-r
        f2[:,1] = r*np.sin(2*np.pi*c*t)
        f2[:,2] = -h*t
        ax.plot(f2[:,0], f2[:,1], f2[:,2])
#        ax.axis('equal')
        plt.pause((0.1))

        dst = (x[..., np.newaxis] - f1[:,0])**2 + (y[..., np.newaxis] - f1[:,1])**2 + (z[..., np.newaxis] - f1[:,2])**2
        dst = np.min(dst, axis=3)
        ftarg = Surface()
        ftarg.Isosurface((dst < delta**2), value=0.5)
        dst = (x[..., np.newaxis] - f2[:,0])**2 + (y[..., np.newaxis] - f2[:,1])**2 + (z[..., np.newaxis] - f2[:,2])**2
        dst = np.min(dst, axis=3)
        ftemp = Surface()
        ftemp.Isosurface((dst < delta**2), value=0.5)
        sigmaKernel = np.array([1,5,10])
        sigmaDist = 10.
        sigmaError = .1
        internalWeight = 5.
        internalCost = None
    else:
        return

    # Object kernel
    K1 = Kernel(name='laplacian', sigma=sigmaKernel)

    # sm = SurfaceMatchingParam(timeStep=0.1, algorithm='bfgs', KparDiff=K1, KparDist=('gauss', sigmaDist),
    #                           sigmaError=sigmaError, errorType='varifold', internalCost=internalCost)
    # sm.KparDiff.pk_dtype = 'float64'
    # sm.KparDist.pk_dtype = 'float64'
    options = {
        'outputDir': '../Output/surfaceMatchingExample/' + model + order,
        'mode': 'normal',
        'saveRate': 20,
        'maxIter': 2000,
        'burnIn': 10,
        'affine': 'none',
        'affineKernel': False,
        'regWeight': regweight,
        'Landmarks': landmarks,
        'rotWeight': 10.,
        'transWeight': 10.,
        'scaleWeight': 10.,
        'affineWeight': 10.,
        'KparDiff': K1,
        'KparDist': ('gauss', sigmaDist),
        'sigmaError': sigmaError,
        'errorType': errorType,
        'pk_dtype': 'float32',
        'algorithm': 'bfgs',
        'unreduced': False,
        'internalWeight': internalWeight,
        'internalCost': internalCost,
        'saveTrajectories': True,
        'unreducedResetRate': 10000,
        'unreducedWeight': 0.1
    }
    if secondOrder:
        f = SecondOrderSurfaceMatching(Template=ftemp, Target=ftarg, options=options)
    else:
        f = SurfaceMatching(Template=ftemp, Target=ftarg, options=options)
    f.optimizeMatching()
    # for k in range(20):
    #     f.options['unreducedWeight'] += 0.1 * f.ds
    #     f.reset = True
    #     f.optimizeMatching()
    plt.ioff()
    plt.show()

    return f


compute(model)
