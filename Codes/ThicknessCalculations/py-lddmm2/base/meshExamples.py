import numpy as np
import netgen.csg as ncsg
from netgen.meshing import Mesh as Meshng, FaceDescriptor, Element2D
from .curves import Curve
from .surfaces import Surface
from .curveExamples import Circle, Ellipse
from .surfaceExamples import Sphere, Sphere_pygal, Heart as surface_heart
from .meshes import Mesh, select_faces__, buildMeshFromCentersCounts
from .pointSetExamples import ringUniform


class TwoDiscs(Mesh):
    def __init__(self, largeRadius = 10., smallRadius = 4.5, targetSize=250):
        f0 = Circle(radius=largeRadius, targetSize=targetSize)
        f1 = Circle(center= f0.center, radius=smallRadius, targetSize=targetSize)
        f = Curve(curve=(f0,f1))
        super(TwoDiscs, self).__init__(f,volumeRatio=5000)
        imagef = np.array(((self.centers - np.array(f0.center)[None, :]) ** 2).sum(axis=1) < smallRadius**2, dtype=float)
        image = np.zeros((self.faces.shape[0], 2))
        image[:, 0] = imagef #(imagev[self.faces[:, 0]] + imagev[self.faces[:, 1]] + imagev[self.faces[:, 2]]) / 3
        image[:, 1] = 1 - image[:, 0]
        self.updateImage(image)

class TwoEllipses(Mesh):
    def __init__(self, Boundary_a = 10., Boundary_b = 10, smallRadius = 0.5, translation = (0,0), targetSize=250,
                 volumeRatio = 5000):
        f0 = Ellipse(a=Boundary_a, b=Boundary_b, targetSize=targetSize)
        f1 = Ellipse(center= np.array([f0.center[0] + translation[0]*Boundary_a, f0.center[1] + translation[1]*Boundary_b]),
                     a=Boundary_b*smallRadius, b=Boundary_a*smallRadius, targetSize=targetSize)
        f = Curve(curve=(f0,f1))
        super(TwoEllipses, self).__init__(f,volumeRatio=volumeRatio)
        # A = (((self.vertices[:,0] - f.center[0] - translation[0]*Boundary_a)/Boundary_b)**2
        #      + ((self.vertices[:,1] - f.center[1] - translation[1]*Boundary_b)/Boundary_a)**2) < smallRadius
        A = (((self.centers[:,0] - f0.center[0] - translation[0]*Boundary_a)/Boundary_b)**2
             + ((self.centers[:,1] - f0.center[1] - translation[1]*Boundary_b)/Boundary_a)**2) < smallRadius**2
        imagev = np.array(A, dtype=float)
        image = np.zeros((self.faces.shape[0], 2))
        image[:, 0] = imagev #(imagev[self.faces[:, 0]] + imagev[self.faces[:, 1]] + imagev[self.faces[:, 2]]) / 3
        image[:, 1] = 1 - image[:, 0]
        self.updateImage(image)

class TwoBalls(Mesh):
    def __init__(self, largeRadius = 10., smallRadius = 4.5, maxh = 1):
        # f0 = Sphere_pygal(radius=largeRadius,targetSize=targetSize)
        # f1 = Sphere_pygal(radius=smallRadius, targetSize=targetSize)
        # f = Surface(surf=(f0,f1))
        # super(TwoBalls, self).__init__(f,volumeRatio=volumeRatio)
        f0 = ncsg.Sphere(ncsg.Pnt(0,0,0), largeRadius)
        f1 = ncsg.Sphere(ncsg.Pnt(0,0,0), smallRadius)
        #fu = (f0-f1)
        #fu = fu + f1
        # f0 = pygalmesh.Ball([0.0, 0.0, 0.0], largeRadius)
        # f1 = pygalmesh.Ball([0.0, 0.0, 0.0], smallRadius)
        # f00 = pygalmesh.Difference(f0, f1)
        # fu = pygalmesh.Union((f00, f1))

        geo1 = ncsg.CSGeometry()
        geo1.Add(f0)
        m1 = geo1.GenerateMesh(maxh=maxh)
        #m1.Refine()
        geo2 = ncsg.CSGeometry()
        geo2.Add(f1)
        m2 = geo2.GenerateMesh(maxh=maxh)
        #m2.Refine()

        # create an empty mesh
        mesh = Meshng()

        # a face-descriptor stores properties associated with a set of surface elements
        # bc . boundary condition marker,
        # domin/domout . domain-number in front/back of surface elements (0 = void),
        # surfnr . number of the surface described by the face-descriptor

        fd_outside = mesh.Add(FaceDescriptor(bc=1, domin=1, surfnr=1))
        fd_inside = mesh.Add(FaceDescriptor(bc=2, domin=2, domout=1, surfnr=2))
        # copy all boundary points from first mesh to new mesh.
        # pmap1 maps point-numbers from old to new mesh

        pmap1 = {}
        for e in m1.Elements2D():
            for v in e.vertices:
                if (v not in pmap1):
                    pmap1[v] = mesh.Add(m1[v])

        # copy surface elements from first mesh to new mesh
        # we have to map point-numbers:

        for e in m1.Elements2D():
            mesh.Add(Element2D(fd_outside, [pmap1[v] for v in e.vertices]))

        # same for the second mesh:

        pmap2 = {}
        for e in m2.Elements2D():
            for v in e.vertices:
                if (v not in pmap2):
                    pmap2[v] = mesh.Add(m2[v])

        for e in m2.Elements2D():
            mesh.Add(Element2D(fd_inside, [pmap2[v] for v in e.vertices]))

        mesh.GenerateVolumeMesh()

        # f = pygalmesh.generate_mesh(
        #     fu, #pygalmesh.Ball([0.0, 0.0, 0.0], largeRadius),
        #     min_facet_angle=30.0,
        #     max_radius_surface_delaunay_ball=radius_ball,
        #     max_facet_distance=facet_distance,
        #     max_circumradius_edge_ratio=edge_ratio,
        #     max_cell_circumradius= circumradius,  # lambda x: abs(np.sqrt(np.dot(x, x)) - 0.5) / 5 + 0.025,
        #     verbose=False
        # )
        super(TwoBalls, self).__init__()
        self.from_netgen(mesh)
        c0 = np.array([0,0,0])
        A = ((self.centers - c0[None, :]) ** 2).sum(axis=1) < smallRadius**2
        # imagev = np.array(A, dtype=float)
        image = np.zeros((self.faces.shape[0], 2))
        image[:, 0] = np.array(A, dtype=float)
            # (imagev[self.faces[:, 0]] + imagev[self.faces[:, 1]] + imagev[self.faces[:, 2]]
            #                + imagev[self.faces[:, 3]]) / 4
        image[:, 1] = 1 - image[:, 0]
        self.updateImage(image)


class MoGCircle(Mesh):
    def __init__(self, largeRadius = 10., nregions = 5, ntypes = 5, ngenes=10, density = 10., centers=None, a=1,
                 targetSize=500, volumeRatio= 5000, cellTypes = True, typeProb = None, geneProb = None, alpha = None):
        f = Circle(radius=2*largeRadius, targetSize=targetSize)
        super(MoGCircle, self).__init__(f,volumeRatio=volumeRatio)

        if centers is None:
            self.nregions = nregions
            pts = np.random.normal(0, 1, (nregions, 2))
            pts = pts / np.sqrt((pts**2).sum(axis=1))[:,None]
            r = np.sqrt(np.random.uniform(0,1,(nregions,1)))
            self.GaussCenters = largeRadius*r*pts
        else:
            self.nregions = centers.shape[0]
            self.GaussCenters = centers

        ## prior on cell types and genes
        if typeProb is None:
            typeProb = np.random.dirichlet([a] * ntypes, nregions)
        if geneProb is None:
            geneProb = np.random.dirichlet([a] * ngenes, ntypes)

        self.label = np.zeros(self.faces.shape[0], dtype=int)
        if cellTypes:
            image = np.zeros((self.faces.shape[0], ntypes))
        else:
            image = np.zeros((self.faces.shape[0], ngenes))
        self.types = np.zeros((self.faces.shape[0], ntypes))
        if alpha is None:
            alpha = np.random.poisson(density, nregions)
        weights = np.zeros(self.faces.shape[0])
        for k in range(self.faces.shape[0]):
            distk = ((self.centers[k,:][None, :] - self.GaussCenters)**2).sum(axis=1)
            jk = np.argmin(distk)
            ## type composition of the simplex
            self.types[k,:] = np.random.dirichlet(a - 1 + typeProb[jk, :])
            self.label[k] = jk
            #weights[k] = np.random.poisson(alpha[self.label[k]]) * np.exp(-distk[jk]/(2*(largeRadius/nregions)**2))
            weights[k] = np.random.poisson(alpha[self.label[k]]) * np.exp(-distk[jk]/(2*(largeRadius/10)**2))
            if not cellTypes:
                for t in range(ntypes):
                    image[k, :] += np.random.choice(np.floor(ngenes*self.types[k,t]), p=geneProb[jk, :])

        # for k in range(self.faces.shape[0]):
        #     weights[k] = np.random.poisson(alpha[self.label[k]])
        self.typeProb = typeProb
        self.geneProb = geneProb
        self.alpha = alpha
        self.updateWeights(weights)
        if cellTypes:
            self.updateImage(self.types)
        else:
            self.updateImage(image)

        newv, newf, newg2, keepface = select_faces__(weights[:, None], self.vertices, self.faces,
                                                      threshold=.1*density)
        wgts = self.face_weights[keepface]
        newg = np.copy(self.image[keepface, :])
        self.vertices = newv
        self.faces = newf
        self.computeCentersVolumesNormals()
        self.updateWeights(wgts)
        self.image = newg

class Rbf(Mesh):
    def __init__(self, N, d=2, sigma=0.):
        nc = 100
        # centers = np.random.uniform(-1.5, 1.5, size=(nc, d))
        centers = sigma * np.random.normal(0,1, size=(nc, d))
        c = np.ones(nc)
        for k in range(nc):
            centers[k, k % d] = 3 * k / float(nc)
            c[k] = np.cos(6 * np.pi * k / nc)  # (2 * (k % 1.5) - 0.75)
        s = 0.25
        x0 = ringUniform(2000, d=d)  # np.random.normal(0, 1, (2000, d))
        K = np.exp(- (((x0[:, np.newaxis, :] - centers[np.newaxis, :, :]) / s) ** 2).sum(axis=2) / np.sqrt(d))
        m = np.median(np.sin(np.dot(K, c)))
        x0 = ringUniform(N, d=d)
        # x0Tr = np.random.normal(0, 1, (NTr, d))
        K = np.exp(- (((x0[:, np.newaxis, :] - centers[np.newaxis, :, :]) / s) ** 2).sum(axis=2) / np.sqrt(d))
        J = (np.sin(np.dot(K, c)) - m) > 0
        print(J.sum())
        f = x0[J, :]
        fv0 = buildMeshFromCentersCounts(f, np.ones((f.shape[0], 1)), resolution=0.01)
        super(Rbf, self).__init__(fv0)

class Heart(Mesh):
    def __init__(self, resolution = 100, targetSize = 1000, p=2., parameters = (0.25, 0.20, 0.1), scales=(1., 1.),
                 zoom = 1.):
        super().__init__()
        s = surface_heart(resolution=resolution, targetSize=targetSize, p=p, parameters=parameters,
                          scales=scales, zoom=zoom)
        #fv0 = buildMeshFromCentersCounts(s.centers, np.ones((s.centers.shape[0], 1)), resolution=0.01*zoom)
        super(Heart, self).__init__(s)


