
import torch
from torch.utils.data import Dataset

class LinregDataset(Dataset):
    def __init__(
        self,
        n_dims, n_points,
        xmean = 0, xstd = 1,
        wmean = 0, wstd = 1,
        total = 10000,
        device = 'cpu'
        ):
        
        self.n_dims = n_dims
        self.n_points = n_points
        
        self.length = total
        self.current = 0
        
        self.xmean, self.xstd = xmean, xstd
        self.wmean, self.wstd = wmean, wstd
        
        self.device = device
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        xs = torch.normal(self.xmean, self.xstd, (self.n_points, self.n_dims))
        w_b = torch.normal(self.wmean, self.wstd, (self.n_dims, 1))
        ys = (xs @ w_b).sum(-1)
        
        xs = xs.to(self.device)
        ys = ys.to(self.device)
        return self.combine(xs, ys), ys
    
    def combine(self, xs, ys):
        n, d = xs.shape
        device = ys.device

        ys_b_wide = torch.cat(
            (
                ys.view(n, 1),
                torch.zeros(n, d-1, device=device),
            ),
            axis = 1,
        )

        zs = torch.stack((xs, ys_b_wide), dim = 1)
        zs = zs.view(2 * n, d)

        return zs