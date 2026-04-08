import torch
import numpy as np
from varap.loss.rescale import rescale_loss

def optimize(uloss, xu_init, dxmax, nb_iter=20, callback=None):

    #  normalize the loss
    nuloss = lambda xu: uloss(xu) / uloss(xu_init)

    optloss, ufromopt, xopt = rescale_loss(nuloss, xu_init, dxmax)
    optimizer = torch.optim.LBFGS(xopt, max_iter=15, line_search_fn='strong_wolfe', history_size=10)

    def closure():
        optimizer.zero_grad()
        L = optloss(xopt)
        print("loss", L.detach().cpu().numpy())
        # print("nu_Z:", ufromopt(xopt)[-1]**2)
        # print("nu_Z max:", (ufromopt(xopt)[-1]**2).max())
        L.backward()
        return L

    for i in range(nb_iter):
        print("it ", i, ": ", end="")
        optimizer.step(closure)

        xu = ufromopt(xopt)
        if callback is not None:
            nZ, nnu_Z = callback(xu)


    return nZ, nnu_Z