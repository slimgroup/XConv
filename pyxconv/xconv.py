import torch
import torch.nn as nn
import torch.autograd.profiler as profiler

import matplotlib.pyplot as plt

from pyxconv import *

if __name__ == "__main__":
    c = Xconv3D(1, 1, (3, 3, 3), bias=False, ps=32, stride=1, padding=1)
    c2 = nn.Conv3d(1, 1, (3, 3, 3), bias=False, padding=1, stride=1)
 
    c2.weight = c.weight

    X = torch.randn(16, 1, 32, 32, 32)
    y2 = c2(X)
    loss2 = .5*torch.norm(y2)**2
    grad_t2 = torch.autograd.grad(loss2, c2.parameters())
    y = c(X)
    loss = .5*torch.norm(y)**2
    grad_t = torch.autograd.grad(loss, c.parameters())

    with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        y = c(X)
        loss = .5*torch.norm(y)**2
        grad_t = torch.autograd.grad(loss, c.parameters())
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        y2 = c2(X)
        loss2 = .5*torch.norm(y2)**2
        grad_t2 = torch.autograd.grad(loss2, c2.parameters())
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))


    plt.plot(grad_t[0].numpy().reshape(-1), label='ev')
    plt.plot(grad_t2[0].numpy().reshape(-1), label='torch')
    plt.legend()
    plt.show()