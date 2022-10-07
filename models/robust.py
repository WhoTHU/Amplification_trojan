import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

def MMcenters(N=10, d=1):
    nodes = np.zeros((N, N-1))
    nodes[0, 0] = d / 2
    nodes[1, 0] = - d / 2
    for i in range(2, N):
        r = np.sqrt((nodes[0]**2).sum())
        c = np.sqrt(d**2 - r**2)
        a = (d**2 - 2 * r**2 )/c/2
        b = (d**2)/c/2
        nodes[:i, i-1] -= a
        nodes[i, i-1] += b
    nodes = torch.Tensor(nodes)
    return nodes


class MMDLoss(nn.Module):
    def __init__(self, N=10, d=1):
        super(MMDLoss, self).__init__()
        self.N = N
        self.d = d
        mmcenters = MMcenters(N, d)
        self.register_buffer('mmcenters', mmcenters)

    def forward(self, x, labels):
        x = x[:, :self.N-1]
        x = x.matmul(self.mmcenters.T)
        loss = F.cross_entropy(x, labels)
        return loss

    def final(self, x):
        x = x[:, :self.N-1]
        x = x.matmul(self.mmcenters.T)
        return x

class MMCLoss(nn.Module):
    def __init__(self, N=10, d=1):
        super(MMCLoss, self).__init__()
        self.N = N
        self.d = d
        mmcenters = MMcenters(N, d)
        self.register_buffer('mmcenters', mmcenters)

    def forward(self, x, labels):
        x = x[:, :self.N-1]
        center = self.mmcenters[labels]
        loss = ((x - center)**2).sum(1).mean()
        return loss

    def final(self, x):
        x = x[:, :self.N-1]
        x = x.matmul(self.mmcenters.T)
        return x




