import math

import torch
from pykeops.torch import Vi, Vj
from pykeops.torch.cluster import grid_cluster, cluster_ranges_centroids, from_matrix, sort_clusters

from varap.loss.rescale import rescale_loss
from varap import dtype


class ParticleLoss_ranges():

    # TODO: make it pure torch ??!!

    rangesZZ_ij = None
    rangesXX_ij = None
    rangesZX_ij = None

    def __init__(self, sig, X, nu_X, Z, nu_Z):
        self.sig = sig
        self.X = X
        self.nu_X = nu_X

        self.Z = Z
        self.nu_Z = nu_Z

        self.make_sorted_ranges()

    @staticmethod
    def makeEps(Z ,Nmax, Npart):
        '''
        Returns list of epsilons to use for making ranges

        params:
            Nmax: maximum number of cubes, default 2500

            Npart: minimun number of particles per cube, default 1000
        '''

        denZ = min(Z.shape[0] / Npart, Nmax)
        rangeOfData = torch.max(Z, axis=0).values - torch.min(Z, axis=0).values
        numDimNonzero = torch.sum(rangeOfData > 0)

        if rangeOfData[-1] == 0:
            volZ = torch.prod(rangeOfData[:-1])
            epsZ = (volZ / denZ) ** (1.0 / numDimNonzero)
        else: # assume 3D
            volZ = torch.prod(rangeOfData) # volume of bounding box  (avoid 0); 1 micron
            epsZ = torch.pow(volZ / denZ, 1.0/3.0)

        print("Z\tVol\tParts\tCubes\tEps")
        print("Z\t" + str(volZ) + "\t" + str(Z.shape[0]) + "\t" + str(denZ) + "\t" + str(epsZ))

        return epsZ


    def make_sorted_ranges(self):
        """
        Returns ranges for Z, X and Z and X. But clusters are depending only on x_i and y_j (point positions)
        and not feature (could be changed when genes expression is diffrenten accross space)
        """
        # Here X and Z are torch tensors
        a = torch.tensor(3.).sqrt()

        # Ranges for Z
        epsZ = self.makeEps(self.Z, 2500.0, 1000.0)
        Z_labels = grid_cluster(self.Z, epsZ)
        Z_ranges, Z_centroids, _ = cluster_ranges_centroids(self.Z, Z_labels)

        D = ((Z_centroids[:, None, :] - Z_centroids[None, :, :]) ** 2).sum(dim=2)
        keep = D <(a * epsZ + 4 * self.sig) ** 2
        self.rangesZZ_ij = from_matrix(Z_ranges, Z_ranges, keep)

        # Ranges for X
        epsX = self.makeEps(self.X, 2500.0, 1000.0)
        X_labels = grid_cluster(self.X, epsX)
        X_ranges, X_centroids, _ = cluster_ranges_centroids(self.X, X_labels)

        D = ((X_centroids[:, None, :] - X_centroids[None, :, :]) ** 2).sum(dim=2)
        keep = D < (a * epsX + 4 * self.sig) ** 2
        self.rangesXX_ij = from_matrix(X_ranges, X_ranges, keep)

        # Ranges for Z and X
        D = ((Z_centroids[:, None, :] - X_centroids[None, :, :]) ** 2).sum(dim=2)
        keep = D < (a * (epsZ / 2.0 + epsX / 2.0) + 4 * self.sig) ** 2
        self.rangesZX_ij = from_matrix(Z_ranges, X_ranges, keep)

        self.Z, _ = sort_clusters(self.Z, Z_labels)  # sorting the labels. TODO: check if it affects the results
        self.nu_Z, _ = sort_clusters(self.nu_Z, Z_labels)

        self.X, _ = sort_clusters(self.X, X_labels)
        self.nu_X, _ = sort_clusters(self.nu_X, X_labels)

        print("Ranges computations done.")




class ParticleLoss_full():

    rangesZZ_ij = None
    rangesXX_ij = None
    rangesZX_ij = None

    def __init__(self, sig, X, nu_X, bw=-1, ranges=None):

        self.dtype = dtype
        self.sig = sig

        self.X = X
        self.nu_X = nu_X

        if ranges is not None:
            self.rangesZZ_ij = ranges.rangesZZ_ij
            self.rangesXX_ij = ranges.rangesXX_ij
            self.rangesZX_ij = ranges.rangesZX_ij
            assert torch.allclose(ranges.X, X), "Ranges are not consistent with data"

        self.loss = self.make_loss(self.X, self.nu_X) if (bw < 0 or self.nu_X.shape[1] <= bw) else self.slice_it(bw)

    def make_loss(self, X0, nu_X0):
        tx = torch.tensor(X0).type(self.dtype).contiguous()
        LX_i, LX_j = Vi(tx), Vj(tx)

        tnu_X = torch.tensor(nu_X0).type(self.dtype).contiguous()
        Lnu_X_i, Lnu_X_j = Vi(tnu_X), Vj(tnu_X)

        D_ij = ((LX_i - LX_j) ** 2 / self.sig ** 2).sum(dim=2)
        K_ij = (- D_ij).exp() * (Lnu_X_i | Lnu_X_j)

        K_ij.ranges = self.rangesXX_ij
        c = K_ij.sum(dim=1).sum()
        print('c=', c)

        def loss(tZal_Z):
            LZ_i, LZ_j = Vi(tZal_Z[0]), Vj(tZal_Z[0])

            Lnu_Z_i = Vi(tZal_Z[1] ** 2)
            Lnu_Z_j = Vj(tZal_Z[1] ** 2)

            DZZ_ij = ((LZ_i - LZ_j) ** 2 / self.sig ** 2).sum(dim=2)
            KZZ_ij = (- DZZ_ij).exp() * (Lnu_Z_i | Lnu_Z_j)
            KZZ_ij.ranges = self.rangesZZ_ij

            DZX_ij = ((LZ_i - LX_j) ** 2 / self.sig ** 2).sum(dim=2)
            KZX_ij = (- DZX_ij).exp() * (Lnu_Z_i | Lnu_X_j)
            KZX_ij.ranges = self.rangesZX_ij

            E = KZZ_ij.sum(dim=1) - 2 * KZX_ij.sum(dim=1)
            L = E.sum() + c
            return L

        return loss

    def eval(self, *args):
        return self.loss(*args)

    def slice_it(self, bw):

        nb_bands = math.ceil(self.nu_X.shape[1] / bw)
        bands = [(i * bw, min((i + 1) * bw, self.nu_X.shape[1])) for i in range(nb_bands)]

        ltmploss = [self.make_loss(self.X,
                                   self.nu_X[:, bands[i][0]:bands[i][1]]) for i in range(nb_bands)]

        def uloss(xu):
            return sum([ltmploss[i]([xu[0], xu[1][:, bands[i][0]:bands[i][1]]]) for i in range(nb_bands)])

        return uloss



class ParticleLoss_restricted():

    rangesZZ_ij = None
    rangesXX_ij = None
    rangesZX_ij = None

    def __init__(self, sig, X, nu_X, Z, bw=-1, ranges=None):
        self.sig = sig

        self.X = X
        self.Z = Z
        self.nu_X = nu_X

        if ranges is not None:
            self.rangesZZ_ij = ranges.rangesZZ_ij
            self.rangesXX_ij = ranges.rangesXX_ij
            self.rangesZX_ij = ranges.rangesZX_ij
            assert torch.allclose(ranges.X, X) & torch.allclose(ranges.Z, Z), "Ranges are not consistent with data"

        self.loss = self.make_loss(self.X, self.nu_X, self.Z) if (bw < 0 or self.nu_X.shape[1] <= bw) else self.slice_it(bw)

    def make_loss(self, X0, nu_X0, Z):
        tx = torch.tensor(X0).type(dtype).contiguous()
        LX_i = Vi(tx)
        LX_j = Vj(tx)
        Lnu_X_i = Vi(torch.tensor(nu_X0).type(dtype).contiguous())
        Lnu_X_j = Vj(torch.tensor(nu_X0).type(dtype).contiguous())

        D_ij = ((LX_i - LX_j) ** 2 / self.sig ** 2).sum(dim=2)
        K_ij = (- D_ij).exp() * (Lnu_X_i | Lnu_X_j)
        K_ij.ranges = self.rangesXX_ij

        c = K_ij.sum(dim=1).sum()

        tz = torch.tensor(Z).type(dtype).contiguous()
        LZ_i, LZ_j = Vi(tz), Vj(tz)
        DZZ_ij = ((LZ_i - LZ_j) ** 2 / self.sig ** 2).sum(dim=2)

        print('c=', c)


        def loss(tal_Z):
            Lnu_Z_i, Lnu_Z_j = Vi(tal_Z[0] ** 2), Vj(tal_Z[0] ** 2)

            KZZ_ij = (- DZZ_ij).exp() * (Lnu_Z_i | Lnu_Z_j)
            KZZ_ij.ranges = self.rangesZZ_ij

            DZX_ij = ((LZ_i - LX_j) ** 2 / self.sig ** 2).sum(dim=2)
            KZX_ij = (- DZX_ij).exp() * (Lnu_Z_i | Lnu_X_j)
            KZX_ij.ranges = self.rangesZX_ij

            E = KZZ_ij.sum(dim=1) - 2 * KZX_ij.sum(dim=1)
            L = E.sum() + c

            return L

        return loss

    def eval(self, *args):
        return self.loss(*args)

    def slice_it(self, bw):
        nb_bands = math.ceil(self.nu_X.shape[1] / bw)
        bands = [(i * bw, min((i + 1) * bw, self.nu_X.shape[1])) for i in range(nb_bands)]

        ltmploss = [self.make_loss(self.X,
                               self.nu_X[:, bands[i][0]:bands[i][1]],
                               self.Z) for i in range(nb_bands)]

        def uloss(xu):
            return sum([ltmploss[i]([xu[0][:, bands[i][0]:bands[i][1]]]) for i in range(nb_bands)])

        return uloss



if __name__ == '__main__':

    X = torch.randn(300, 3).cuda()
    nu_X = torch.randn(300, 20).cuda()

    Z = torch.randn(100, 3).cuda()
    nu_Z = torch.randn(100, 20).cuda()

    sig = torch.tensor([1.]).cuda()

    L = ParticleLoss_restricted(sig, X, nu_X, Z)
    print(L.loss([nu_Z]))

    # L = ParticleLoss_full(sig, X, nu_X)
    # L.loss([Z, nu_Z])

    ranges = ParticleLoss_ranges(sig, X, nu_X, Z, nu_Z)

    L = ParticleLoss_restricted(sig, X, nu_X, Z, ranges=ranges)
    print(L.loss([nu_Z]))

    # L = ParticleLoss_full(sig, X, nu_X, ranges=ranges)
    # L.loss([Z, nu_Z])

    L = ParticleLoss_restricted(sig, X, nu_X, Z, bw=10)
    print(L.loss([nu_Z]))

    L = ParticleLoss_restricted(sig, X, nu_X, Z, bw=10, ranges=ranges)
    print(L.loss([nu_Z]))

    #L.range_it(Z, nu_Z)
    pass