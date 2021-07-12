import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class VIB(nn.Module):
    def __init__(self, dim_in, dim_out, use_bn=False, beta=1):
        super(VIB, self).__init__()
        self.use_bn = use_bn
        # print(dim_in, dim_out)
        self.proj_mean = nn.Sequential(nn.Linear(dim_in, dim_out),
                                       nn.BatchNorm1d(dim_out, 2e-5)) if use_bn else nn.Linear(dim_in, dim_out)
        self.proj_var = nn.Sequential(nn.Linear(dim_in, dim_out),
                                      nn.BatchNorm1d(dim_out, 2e-5)) if use_bn else nn.Linear(dim_in, dim_out)
        self.norm_dist = torch.distributions.Normal(0.0, 1.0)
        self.beta = beta
        self._init_weight()
        print("USING VIB")

    def _init_weight(self):
        if self.use_bn:
            for i in range(2):
                self._init_helper(self.proj_mean[i])
                self._init_helper(self.proj_var[i])
        else:
            self._init_helper(self.proj_mean)
            self._init_helper(self.proj_var)

    def _init_helper(self, module):
        init.normal_(module.weight, std=0.01)
        init.constant_(module.bias, 0.0)

    def forward(self, in_ft):
        info_loss = 0.0
        if self.training:
            ft_mean = self.proj_mean(in_ft)
            ft_var = self.proj_var(in_ft)
            ft_dist = torch.distributions.Normal(ft_mean, F.softplus(ft_var - 5))
            ft = ft_mean + ft_dist.sample()
            info_loss = torch.sum(
                torch.mean(torch.distributions.kl_divergence(ft_dist, self.norm_dist), dim=0)) * self.beta
        else:
            ft = self.proj_mean(in_ft)
        return ft, info_loss
