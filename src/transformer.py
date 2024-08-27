
import torch
import torch.nn as nn

from .layers import Layer
    
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.emb = nn.Linear(config.n_dims, self.config.hidden_dim)
        # We should try even-odd embedding
        self.pe = nn.Embedding(config.context, self.config.hidden_dim)
        self.layers = nn.Sequential(*[
            Layer(self.config) for i in range(config.num_layers)
        ])
        self.out = nn.Linear(self.config.hidden_dim, config.n_dims)
        
    def forward(self, x, b = 1):
        
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
            
        B, N, d = x.size()
        
        if self.config.method == 'softmax':
            mask = torch.tril(torch.ones(N, N)).view(1, 1, N, N).to(x.device)
        else:
            mask = None
        
        x = self.emb(x)
        x = x + self.pe(torch.arange(N, device=x.device))
        
        output = torch.zeros_like(x)
        pred_list = []
        for i in range(b):
            output = output + x # Input Injection
            for layer in self.layers:
                output = layer(output, mask)
            prediction = self.out(output)[:, ::2, 0]
            pred_list.append(prediction)
        return pred_list