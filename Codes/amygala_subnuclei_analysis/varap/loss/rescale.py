import torch
def rescale_loss(loss, x, dmax):
    # implement a change of variable of the loss for lbfg
    # x and dmax as suppose to be a list of tensors for each group of variables

    xr = []
    # Create a randomized initialization point
    for (xh, dmaxh) in zip(x, dmax):
        xr.append((xh + (2 * torch.rand(xh.shape) - 1) * dmaxh).requires_grad_(True))

    # Compute the gradient at randomized intial point
    L = loss(xr)
    L.backward()

    # Compute the scaling coefficient
    a, cp = [], 0.
    for (xrh, dmaxh) in zip(xr, dmax):
        ah = (dmaxh / xrh.grad.max()).sqrt()
        cp += ah * xrh.grad.abs().sum()
        a.append(ah)
    cp = torch.max(torch.tensor(1.), cp)

    a = [ah * cp for ah in a]

    def uvar_from_optvar(tx):
        # tranform optimization var into user variable
        return [xh * ah for (xh, ah) in zip(tx, a)]

    def loss_new(xopt_cur):
        # create the new rescaled using the optimization variable
        return loss(uvar_from_optvar(xopt_cur))

    xopt = []
    for (xh, ah) in zip(x, a):
        xopt.append((xh / ah).requires_grad_(True))
        print("ah:", ah)

    return loss_new, uvar_from_optvar, xopt
