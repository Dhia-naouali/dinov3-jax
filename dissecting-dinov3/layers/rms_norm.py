import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim):
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = 1e-5
    
    def reset_params(self):
        nn.init.constant_(self.weight, 1)

    def norm_(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        # x: (b, seq, dim)
        # self.weight: (dim,)
        return self.weight * self.norm_(x.float()).type_as(x)