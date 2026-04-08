import torch

# boundary weight regularization
class NoSupportRestriction:
    def __init__(self, eta0=None):
        """
        `lam` is a bandwith used to describe the transition zone between the ROI and the exterior.
        """
        self.eta0 = torch.sqrt(torch.tensor(0.1)) if eta0 is None else eta0
        self.weight = 1.0

    def get_params(self):
        return {"eta0": self.eta0, "weight": self.weight}

    def __call__(self, lamb_est=None):
        reg = 1.0
        return reg

