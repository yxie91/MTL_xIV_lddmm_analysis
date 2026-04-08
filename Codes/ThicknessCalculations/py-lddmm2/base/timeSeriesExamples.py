import numpy as np
from .surfaceExamples import Ellipse


class Ellipses():
    def __init__(self, nsurf=10, a = (5, 10), b = None, c = None, withLandmarks = False):
        if len(a) == 2:
            self.a = np.linspace(a[0], a[1], nsurf)
        else:
            self.a = a

        if b is None:
            self.b = self.a
        elif len(b) == 2:
            self.b = np.linspace(b[0], b[1], nsurf)
        else:
            self.b = b

        if c is None:
            self.c = self.a
        elif len(c) == 2:
            self.c = np.linspace(c[0], c[1], nsurf)
        else:
            self.c = c

        self.fv = []
        for k in range(nsurf):
            self.fv.append(Ellipse(radius=(self.a[k], self.b[k], self.c[k]), withLandmarks=withLandmarks))

        if withLandmarks:
            self.lmk = []
            for k in range(nsurf):
                self.lmk += [self.fv[k].lmk]
        else:
            self.lmk = None
