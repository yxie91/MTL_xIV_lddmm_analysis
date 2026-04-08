import torch


class PiRegularizationSystemDummy:
    """
        pi regularization (KL divergence)
    """
    def __init__(self, zeta_S, nu_T, norm=True):
        self.nuTconst = torch.sum(nu_T)
        # two alternatives for distribution to compare to
        if not norm:
            self.nu_Tprob = torch.sum(nu_T, dim=0) / torch.sum(
                nu_T
            )  # compare to overall distribution of features
        else:
            self.nu_Tprob = (torch.ones((1, nu_T.shape[-1])) / nu_T.shape[-1])

        self.zeta_S = zeta_S
        self.nu_T = nu_T

        self.weight = 1.0 #TODO: put normalizing cste here ?1.0 / torch.log(torch.tensor(nu_T.shape[-1]))

    def get_params(self):
        params_dict = {
            "weight": self.weight,
            "norm": True,
            "nuTconst": self.nuTconst
        }
        return params_dict

    def __call__(self, qw, pi_est):
        return 0.0*self.nuTconst
