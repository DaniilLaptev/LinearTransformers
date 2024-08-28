
import torch 
import torch.nn as nn 
import math

def get_kernel(config):   
    KERNELS = {
        'based': BasedKernel,
        'rebased': ReBasedKernel,
        'elu': EluKernel,
        'learnable': LearnableKernel,
        'squared': SquaredKernel
    }
    return KERNELS[config.method](config)

def normalize(x):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + 1e-6)

class BasedKernel(nn.Module):
    def __init__(self, config):
        super(BasedKernel, self).__init__()
        
        self.feature_dim = config.dhat // config.attn_heads
        self.r2 = math.sqrt(2)
        self.rd = math.sqrt(self.feature_dim)
        self.rrd = math.sqrt(self.rd)
        
    def forward(self, x):
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2)
        return torch.cat([
            torch.ones(x[..., :1].shape).to(x.device),
            x / self.rrd,
            x2 / self.r2 / self.rd
        ], dim = -1)

class ReBasedKernel(nn.Module):
    def __init__(self, config):
        super(ReBasedKernel, self).__init__()

        self.feature_dim = config.dhat // config.attn_heads
        self.gamma = nn.Parameter(torch.rand(1, 1, 1, self.feature_dim))
        self.beta = nn.Parameter(torch.rand(1, 1, 1, self.feature_dim))
        self.rd = math.sqrt(config.hidden_dim)
        
    def forward(self, x):
        x = self.gamma * normalize(x) + self.beta
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2)
        return x2 / self.rd
    
class LearnableKernel(nn.Module):
    def __init__(self, config):
        super(LearnableKernel, self).__init__()

        self.feature_dim = config.dhat // config.attn_heads
        self.transformation = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
            )
        
    def forward(self, x):
        x = self.transformation(normalize(x))
        return x
    
class EluKernel(nn.Module):
    def __init__(self, config):
        super(EluKernel, self).__init__()

        self.feature_dim = config.dhat // config.attn_heads
        self.elu = nn.ELU(config.elu_alpha)
        
    def forward(self, x):
        x = self.elu(x) + 1
        return x
    
class SquaredKernel(nn.Module):
    def __init__(self, config):
        super(SquaredKernel, self).__init__()

        self.feature_dim = config.dhat // config.attn_heads
        self.gamma = nn.Parameter(torch.rand(1, 1, 1, self.feature_dim))
        self.beta = nn.Parameter(torch.rand(1, 1, 1, self.feature_dim))
        
    def forward(self, x):
        x = (self.gamma * normalize(x) + self.beta).square()
        return x