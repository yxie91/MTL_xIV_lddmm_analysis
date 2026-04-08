import numpy as np
from .gridscalars import GridScalars

class Circle(GridScalars):
    def __init__(self, N=151, radius=.5):
        x = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(x, x)
        data = np.array(radius > (X**2 + Y**2)**(1/2))
        self.radius = radius
        super(Circle, self).__init__(grid=data.astype(float))


class TwoCircles(GridScalars):
    def __init__(self, N=151, radius=(.3,.3), offset=(.5, .5)):
        h = 2 / (N - 1)
        x = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(x, x)
        C1 = np.array(radius[0] > (X ** 2 + (Y -offset[0]) ** 2) ** (1 / 2))
        C2 = np.array(radius[1] > (X ** 2 + (Y + offset[0]) ** 2) ** (1 / 2))
        data = np.maximum(C1, C2).T
        self.radius = radius
        self.offset = offset
        super(TwoCircles, self).__init__(grid=data.astype(float))


class ThreeCircles(GridScalars):
    def __init__(self, N=151, radius=(.3, .3, .3), offset=(.6, 0., 0., .5, .5)):
        h = 2 / (N - 1)
        x = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(x, x)
        C0 = np.array(radius[0] > ((X-offset[0]) ** 2 + (Y-offset[1]) ** 2) ** (1 / 2))
        C1 = np.array(radius[1] > ((X - offset[2]) ** 2 + (Y - offset[3]) ** 2) ** (1 / 2))
        C2 = np.array(radius[2] > ((X - offset[2]) ** 2 + (Y + offset[4]) ** 2) ** (1 / 2))
        data = np.maximum(np.maximum(C1, C2), C0).T
        self.radius = radius
        self.offset = offset
        super(ThreeCircles, self).__init__(grid=data.astype(float))


class FourCircles(GridScalars):
    def __init__(self, N=151, radius=(.3,.3,.3, .3), offset=(.5, .5, .5, .5)):
        h = 2 / (N - 1)
        x = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(x, x)
        C0 = np.array(radius[0] > ((X-offset[0]) ** 2 + Y ** 2) ** (1 / 2))
        C1 = np.array(radius[1] > (X ** 2 + (Y -offset[1]) ** 2) ** (1 / 2))
        C2 = np.array(radius[2] > (X ** 2 + (Y + offset[2]) ** 2) ** (1 / 2))
        C3 = np.array(radius[3] > ((X+offset[3]) ** 2 + Y ** 2) ** (1 / 2))
        data = np.maximum(np.maximum(C1, C2), np.maximum(C0, C3)).T
        self.radius = radius
        self.offset = offset
        super(FourCircles, self).__init__(grid=data.astype(float))

class Donut(GridScalars):
    def __init__(self, N=151, radius=(.3,.6), offset=0):
        h = 2 / (N - 1)
        x = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(x, x)
        C0 = np.array(radius[0] < ((X-offset) ** 2 + Y ** 2) ** (1 / 2))
        C1 = np.array(radius[1] > (X ** 2 + (Y -offset) ** 2) ** (1 / 2))
        self.radius = radius
        self.offset = offset
        super(Donut, self).__init__(grid=((C0*C1).T).astype(float))


