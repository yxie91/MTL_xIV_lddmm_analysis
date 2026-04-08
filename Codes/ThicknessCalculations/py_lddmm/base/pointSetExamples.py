import numpy as np
import netgen.csg as ncsg
from netgen.meshing import Mesh as Meshng, FaceDescriptor, Element2D
from .curves import Curve
from .curveExamples import Circle as CE_Circle, Ellipse
from .pointSets import PointSet

def ringUniform(n, a=0., b=1., d=2):
    m = np.zeros(d)
    S = np.eye(d)
    X = np.random.multivariate_normal(m, S, size=n)
    nrm = np.sqrt((X**2).sum(axis=1))
    X /= nrm[:, np.newaxis]
    r = np.random.uniform(0,1, size=n)
    r = (a**d + r* (b**d-a**d))**(1/d)
    return X*r[:, np.newaxis]


class TwoDiscs(PointSet):
    def __init__(self, largeRadius = 10., smallRadius = 4.5, targetSize=250):
        f0 = CE_Circle(radius=largeRadius, targetSize=targetSize)
        f1 = CE_Circle(center= f0.center, radius=smallRadius, targetSize=targetSize)
        f = Curve(curve=(f0,f1))
        super(TwoDiscs, self).__init__(f,volumeRatio=5000)

class TwoEllipses(PointSet):
    def __init__(self, Boundary_a = 10., Boundary_b = 10, smallRadius = 0.5, translation = (0,0), targetSize=250,
                 volumeRatio = 5000):
        f0 = Ellipse(a=Boundary_a, b=Boundary_b, targetSize=targetSize)
        f1 = Ellipse(center= np.array([f0.center[0] + translation[0]*Boundary_a, f0.center[1] + translation[1]*Boundary_b]),
                     a=Boundary_b*smallRadius, b=Boundary_a*smallRadius, targetSize=targetSize)
        f = Curve(curve=(f0,f1))
        super(TwoEllipses, self).__init__(f,volumeRatio=volumeRatio)

class TwoBalls(PointSet):
    def __init__(self, largeRadius = 10., smallRadius = 4.5, circumradius = 0.5, facet_distance = 0.025, radius_ball = 0.5, edge_ratio=2.0):
        # f0 = Sphere_pygal(radius=largeRadius,targetSize=targetSize)
        # f1 = Sphere_pygal(radius=smallRadius, targetSize=targetSize)
        # f = Surface(surf=(f0,f1))
        # super(TwoBalls, self).__init__(f,volumeRatio=volumeRatio)
        #import pygalmesh
        f0 = ncsg.Sphere(ncsg.Pnt(0,0,0), largeRadius)
        f1 = ncsg.Sphere(ncsg.Pnt(0,0,0), smallRadius)
        geo1 = ncsg.CSGeometry()
        geo1.Add(f0)
        m1 = geo1.GenerateMesh(maxh=largeRadius/10)
        #m1.Refine()
        geo2 = ncsg.CSGeometry()
        geo2.Add(f1)
        m2 = geo2.GenerateMesh(maxh=largeRadius/10)

        mesh = Meshng()
        pmap1 = {}
        for e in m1.Elements2D():
            for v in e.vertices:
                if (v not in pmap1):
                    pmap1[v] = mesh.Add(m1[v])

        # copy surface elements from first mesh to new mesh
        # we have to map point-numbers:
        fd_outside = mesh.Add(FaceDescriptor(bc=1, domin=1, surfnr=1))
        fd_inside = mesh.Add(FaceDescriptor(bc=2, domin=2, domout=1, surfnr=2))

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
        #
        #
        #
        # f0 = pygalmesh.Ball([0.0, 0.0, 0.0], largeRadius)
        # f1 = pygalmesh.Ball([0.0, 0.0, 0.0], smallRadius)
        # f00 = pygalmesh.Difference(f0, f1)
        # fu = pygalmesh.Union((f00, f1))
        #
        # f = pygalmesh.generate_mesh(
        #     fu, #pygalmesh.Ball([0.0, 0.0, 0.0], largeRadius),
        #     min_facet_angle=30.0,
        #     max_radius_surface_delaunay_ball=radius_ball,
        #     max_facet_distance=facet_distance,
        #     max_circumradius_edge_ratio=edge_ratio,
        #     max_cell_circumradius= circumradius,  # lambda x: abs(np.sqrt(np.dot(x, x)) - 0.5) / 5 + 0.025,
        #     verbose=False
        # )
        p_ = mesh.Points()
        dim = len(list(p_[1]))
        # if dim == 2:
        #     m = mesh.Elements2D().NumPy()
        # else:
        #     m = nmesh.Elements3D().NumPy()
        self.dim = dim
        p = np.zeros((len(p_), dim))
        for k, pt in enumerate(p_):
            p[k, :] = pt.p

        super(TwoBalls, self).__init__(data=p)


class Rbf(PointSet):
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
        super(Rbf, self).__init__(f)

class Circle(PointSet):
    def __init__(self, N=40, r=1., shift=0.):
        theta = np.arange(N)*2*np.pi/N-shift
        target = np.zeros((N,2))
        target[:,0] = r * np.cos(theta)
        target[:,1] = r * np.sin(theta)
        super(Circle, self).__init__(target)

class Flower(PointSet):
    def __init__(self, N=40, r=4., rsmall = 0., a=3., shift = 0.):
        theta = np.arange(N)*2*np.pi/N - shift
        target = np.zeros((N,2))
        target[:,0] = r*(rsmall + np.abs(np.sin(theta*a))) * np.cos(theta)
        target[:,1] = r*(rsmall + np.abs(np.sin(theta*a))) * np.sin(theta)
        super(Flower, self).__init__(target)

class ShrinkCircle(PointSet):
    def __init__(self, N=40, r1=1.,r2=4., shift=0., offset=(0., 0.)):
        theta = np.arange(N) * 2 * np.pi / N - shift
        target = np.zeros((N, 2))
        L = np.zeros(N)
        L[:(N//2)] = np.linspace(r1, r2, N//2)
        L[(N//2):] = np.linspace(r2, r1, N-N//2)
        target[:, 0] = L * np.cos(theta) + offset[0]
        target[:, 1] = L * np.sin(theta) + offset[1]
        super(ShrinkCircle, self).__init__(target)

class Fluc(PointSet):
    def __init__(self, N=40, a=1.,b=1., fluctuation = .4, shift=0., offset=(0., 0.)):
        theta = np.arange(N) * 2 * np.pi / N - shift
        target = np.zeros((N, 2))
        target[:, 0] = a * np.cos(theta) * (1+fluctuation  * np.cos(8*theta)) + offset[0]
        target[:, 1] = b * np.sin(theta) * (1+fluctuation  * np.sin(8*theta))+ offset[1]
        super(Fluc, self).__init__(target)

class BumpyEllipse(PointSet):
    def __init__(self, N=40, a=1.,b=1., fluctuation = .4, shift=0., offset=(0., 0.)):
        theta = np.arange(N) * 2 * np.pi / N - shift
        target = np.zeros((N, 2))
        target[:, 0] = a * np.cos(theta) * (1+fluctuation  * np.cos(8*theta)) + offset[0]
        target[:, 1] = b * np.sin(theta) * (1+fluctuation  * np.cos(8*theta))+ offset[1]
        super(BumpyEllipse, self).__init__(target)

class SimplifiedHuman(PointSet):
    def __init__(self, N_head=30, headPos=[0, 2], heada = 0.5, headb = .5,
                 N_arm=40, armSize=0.25, armLen = 1.5, armCenter=[0, 1.1], armAngles = [0, 0],
                 N_body=40, N_bottom=10, bodySize=2.5, bodyWidth=0.75):
        theta = np.linspace(0, 2*np.pi, N_head+1)
        theta = theta[:-1]
        head = np.zeros((N_head, 2))
        head[:, 0] = heada*np.cos(theta) + headPos[0]
        head[:, 1] = headb*np.sin(theta) + headPos[1]
        leftarm = np.zeros((N_arm//2, 2))
        rightarm = np.zeros((N_arm-leftarm.shape[0], 2))
        leftEnd = armCenter[0] - bodyWidth / 2 - armLen * np.cos(armAngles[0])
        leftHeight = armCenter[1] - armLen * np.sin(armAngles[0])
        rightEnd = armCenter[0] + bodyWidth / 2 + armLen * np.cos(armAngles[1])
        rightHeight = armCenter[1] + armLen * np.sin(armAngles[1])
        l1 = np.linspace(armCenter[0]-bodyWidth/2, leftEnd-armSize/2*np.sin(armAngles[0]), leftarm.shape[0]//2)
        l2 = np.linspace(leftEnd+armSize/2*np.sin(armAngles[0]), armCenter[0]-bodyWidth/2, leftarm.shape[0]-len(l1))
        h1 = np.linspace(armCenter[1]+armSize/2, leftHeight+armSize/2*np.cos(armAngles[0]), leftarm.shape[0]//2)
        h2 = np.linspace(leftHeight-armSize/2*np.cos(armAngles[0]), armCenter[1]-armSize/2, leftarm.shape[0]-len(l1))
        leftarm[:, 0] = np.concatenate((l1, l2), 0)
        leftarm[:, 1] = np.concatenate((h1, h2), 0)
        L1 = np.linspace(armCenter[0]+bodyWidth/2, rightEnd-armSize/2*(np.sin(armAngles[1])), rightarm.shape[0]//2)
        L2 = np.linspace(rightEnd+armSize/2*(np.sin(armAngles[1])), armCenter[0]+bodyWidth/2, rightarm.shape[0]-len(L1))
        H1 = np.linspace(armCenter[1]+armSize/2, rightHeight+armSize/2*np.cos(armAngles[1]), rightarm.shape[0]//2)
        H2 = np.linspace(rightHeight-armSize/2*np.cos(armAngles[1]), armCenter[1]-armSize/2, rightarm.shape[0]-len(L1))
        rightarm[:, 0] = np.concatenate((L1, L2), 0)
        rightarm[:, 1] = np.concatenate((H1, H2), 0)
        bodyleft = np.zeros((N_body//2, 2))
        bodyright = np.zeros((N_body//2, 2))
        bodyleft[:, 0] = armCenter[0] - bodyWidth/2
        bdleft2 = np.linspace(armCenter[1]-armSize/2, armCenter[1]-armSize/2-bodySize, 1+bodyleft.shape[0])
        bodyleft[:, 1] = bdleft2[1:]
        bodyright[:, 1] = bdleft2[1:]
        bodyright[:, 0] = armCenter[0] + bodyWidth/2
        bodyhori = np.zeros((N_bottom, 2))
        bodyhori[:, 1] = armCenter[1]-armSize/2-bodySize
        hori2 = np.linspace(armCenter[0]-bodyWidth/2, armCenter[0]+bodyWidth/2, N_bottom+2)
        bodyhori[:, 0] = hori2[1:-1]
        target = np.concatenate((head, leftarm, rightarm, bodyleft, bodyright, bodyhori), 0)
        super(SimplifiedHuman, self).__init__(target)





        # case 'big'
        #     weight = 15*10;
        #     tau = 0.7;
        #     a0 = 0.25*10^3;
        #     nn = 10;
        #     theta = (1:nn)'*2*pi/nn;
        #     landmark = [cos(theta), sin(theta); 3*cos(theta), 3*sin(theta)];
        #     target = [0.5*cos(circshift(theta, 2)), 0.5*sin(circshift(theta, 2)); 4*cos(circshift(theta, -1)), 4*sin(circshift(theta, -1))];
        #     % outer counter clockwise
        #     % inner clockwise
        # case 'rotate1'
        #     tau = 0.7;
        #     a0 = 0.25*10^3;
        #     weight = 15;
        #     nn = 10;
        #     theta = (1:nn)'*2*pi/nn;
        #     landmark = 2*[cos(theta), sin(theta)];
        #     target = 2*[cos(circshift(theta,1)), sin(circshift(theta,1))];
        # case 'rotate2'
        #     tau = 0.7;
        #     a0 = 0.25*10^3;
        #     weight = 10;
        #     nn = 10;
        #     theta = (1:nn)'*2*pi/nn;
        #     landmark = 2*[cos(theta), sin(theta)];
        #     target = 2*1.5*[cos(circshift(theta,1)), sin(circshift(theta,1))];
        # case 'complicated_rotate'
        #     tau = 0.6;
        #     a0 = 0.025;
        #     weight = 10;
        #     nn = 10;
        #     theta = (1:nn)'*2*pi/nn;
        #     landmark = 2*[cos(theta), sin(theta)];
        #     len1 = linspace(1,4,5); len = [len1, flip(len1)]';
        #     target = repmat(len, 1,2).*[cos(circshift(theta,4)), sin(circshift(theta,4))]+1;
        # case 'simple_3_move'
        #     landmark = [-2 0; 1 1; 2 0];
        #     target = [-3 0; 1 2; 3 0];
        #     tau = 0.8;
        #     a0 = 0.08;
        #     weight = 10;
        # case 'combined_inv'
        #     weight = 15*10;
        #     tau = 0.7;
        #     a0 = 0.25*10^3;
        #     nn = 30;    fluctuation = .4;
        #     theta = (1:nn)'*2*pi/nn;
        #     landmark = [4*cos(theta), sin(theta)];
        #     target = 2*[cos(theta), sin(theta)]+fluctuation*[cos(theta*9), sin(theta*9)];
        # case 'combined'
        #     weight = 15*100;
        #     tau = 0.7;
        #     a0 = 0.25*10^3;
        #     nn = 30;    fluctuation = .4;
        #     theta = (1:nn)'*2*pi/nn;
        #     landmark = 2*[cos(theta), sin(theta)];
        #     target = [4*cos(theta), sin(theta)]+fluctuation*[cos(theta*9), sin(theta*9)];
        # case 'combined1'        % compress circle to ellipse + translation
        #     weight = 15;
        #     tau = 0.7;
        #     a0 = 0.25*10^3;
        #     nn = 30;    direction = [-1,1];     magnitude = 2;
        #     theta = (1:nn)'*2*pi/nn;
        #     landmark = 2*[cos(theta), sin(theta)];
        #     target = [4*cos(theta), 1*sin(theta)]+direction * magnitude;
        # case 'nofluc'
        #     weight = 15;
        #     tau = 0.7;
        #     a0 = 0.25*10^3;
        #     nn = 30;    fluctuation = 0;
        #     theta = (1:nn)'*2*pi/nn;
        #     landmark = 2*[cos(theta), sin(theta)];
        #     target = [4*cos(theta), sin(theta)]+fluctuation*[cos(theta*9), sin(theta*9)];
