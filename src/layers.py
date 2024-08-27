
import torch
import torch.nn as nn

import math

from .kernels import get_kernel

def softmax_attention(Q, K, V, mask):
    B, h, N, d = V.size()
    logits = Q @ K.transpose(-2, -1) / math.sqrt(d)
    logits = logits.masked_fill(mask[:, :, :N, :N] == 0, float('-inf'))
    scores = torch.softmax(logits, -1)
    out = scores @ V
    return out

def linear_attention(Q, K, V):
    kv = torch.einsum('bhtf, bhtg -> bhtfg', K, V).cumsum(dim=2)
    Z = 1 / (torch.einsum("bhtf, bhtf -> bht", Q, K.cumsum(2)) + 1e-6)
    out = torch.einsum('bhtf, bhtfg -> bhtg', Q, kv) * Z[:, :, :, None]
    return out

def ref_linear(Q, K, V):
    B, h, N, d = V.size()
    mask = torch.tril(torch.ones(N, N)).view(1, 1, N, N)
    logits = Q @ K.transpose(-2, -1) / math.sqrt(d)
    logits = logits.masked_fill(mask[:,:,:N,:N] == 0, 0)
    scores = logits / logits.sum(-1, keepdim=True)
    out = scores @ V
    return out

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_dim % config.attn_heads == 0
                
        self.wq = nn.Linear(config.hidden_dim, config.dhat)
        self.wk = nn.Linear(config.hidden_dim, config.dhat)
        self.wv = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.linear = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.separate_kernels = False
        if config.method in ['based', 'elu']:
            self.phi = get_kernel(config)
        elif config.method in ['rebased', 'learnable', 'squared']:
            self.phiq = get_kernel(config)
            self.phik = get_kernel(config)
            self.separate_kernels = True
        elif config.method != 'softmax':
            raise NotImplementedError(f'Method {config.method} is not implemented.')
        
        self.dhat = config.dhat
        self.hidden_dim = config.hidden_dim
        self.attn_heads = config.attn_heads
        self.method = config.method
    
    def forward(self, x, mask = None):
        B, N, d = x.size()
        
        Q, K, V = self.wq(x), self.wk(x), self.wv(x)

        # После этих преобразований матрицы Q и K будут иметь размер [B, h, N, d]
        Q = Q.view(B, N, self.attn_heads, self.dhat // self.attn_heads).transpose(1, 2)
        K = K.view(B, N, self.attn_heads, self.dhat // self.attn_heads).transpose(1, 2) 
        V = V.view(B, N, self.attn_heads, d // self.attn_heads).transpose(1, 2)
        
        if self.method in ['based', 'elu']:
            Q, K = self.phi(Q), self.phi(K)
        elif self.method in ['rebased', 'learnable', 'squared']:
            Q, K = self.phiq(Q), self.phik(K)
            
        if self.method == 'softmax':
            out = softmax_attention(Q, K, V, mask)
        else:
            out = linear_attention(Q, K, V)

        out = out.transpose(1, 2).contiguous().view(B, N, d)
        out = self.linear(out)
        
        return out

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_dim, config.mlp_hidden)
        self.gelu = config.activation()
        self.linear2 = nn.Linear(config.mlp_hidden, config.hidden_dim)
    
    def forward(self, x):
        x = self.linear2(self.gelu(self.linear1(x)))
        return x
    
class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = MLP(config)
        
    def forward(self, x, mask = None):
        attn = self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x + attn))
        return x