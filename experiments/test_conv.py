from pyxconv import *
from torchvision import datasets, transforms
from torch.autograd import gradcheck

from scipy import interpolate

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

b, ci, co, nx, ny = 2, 1, 1, 64, 64

dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())

def to_fine(x):
    r1 = np.linspace(0, 1, 28)
    r2 = np.linspace(0, 1, nx)
    r3 = np.linspace(0, 1, ny)
    return torch.tensor(interpolate.interp2d(r1, r1, x, kind='cubic')(r2, r3),
                        dtype=torch.float32)

xt = torch.cat([to_fine(dataset1[rd.randint(0, len(dataset1))][0]) for i in range(b*ci)]).reshape(b, ci, nx, ny)
xt.requires_grad_()

c1 = torch.nn.Conv2d(ci, co, (3, 3), stride=1, padding=1, bias=False)
c2 = Zconv2D(ci, co, (3, 3), ps=64, stride=1, padding=1, bias=False)
c3 = Xconv2D(ci, co, (3, 3), ps=32, stride=1, padding=1, bias=False)
c2.weight = torch.nn.Parameter(1+torch.randn(c2.weight.shape))
c1.weight = c2.weight
c3.weight = c2.weight

y1 = c1(xt)
y2 = c2(xt)
y3 = c3(xt)

def to_plot(x):
    return x.permute((0, 2, 1, 3)).reshape(b*nx, co*ny).detach().numpy()

plt.figure()
plt.subplot(131)
plt.imshow(to_plot(y2), aspect='auto', vmin=0, vmax=5, cmap='jet')
plt.subplot(132)
plt.imshow(to_plot(y2-y1), aspect='auto', vmin=-.5, vmax=.5, cmap='jet')
plt.subplot(133)
plt.imshow(to_plot(y1), aspect='auto', vmin=0, vmax=5, cmap='jet')

plt.figure()
plt.subplot(131)
plt.imshow(to_plot(torch.nn.ReLU()(y2)), aspect='auto', vmin=0, vmax=5, cmap='jet')
plt.subplot(132)
plt.imshow(to_plot(torch.nn.ReLU()(y2)- torch.nn.ReLU()(y1)), aspect='auto', vmin=-.5, vmax=.5, cmap='jet')
plt.subplot(133)
plt.imshow(to_plot(torch.nn.ReLU()(y1)), aspect='auto', vmin=0, vmax=5, cmap='jet')


out = torch.randn(b, co, nx, ny)

g1x, g1w, _ = y1.grad_fn(y1 - out)
g2x, g2w = y2.grad_fn.apply(y2 - out)[0:2]
g3x, g3w = y3.grad_fn.apply(y3 - out)[0:2]

plt.figure()
plt.plot(g1w.reshape(-1).detach().numpy(), label='true')
plt.plot(g2w.reshape(-1).detach().numpy(), label='rand')
plt.plot(g3w.reshape(-1).detach().numpy(), label='randx')
plt.legend()
plt.show()

h = [2**(-i) for i in range(2, 8)]

w = torch.clone(c2.weight)
dw = torch.nn.Parameter(1e-3 * torch.randn(c2.weight.shape))

J = torch.dot(dw.reshape(-1), g2w.reshape(-1))
f0 = .5*torch.norm(y2 - out)**2

err1 = torch.zeros(6)
err2 = torch.zeros(6)

for i, hh in enumerate(h):
    c2.weight = torch.nn.Parameter(w + hh * dw)
    y = c2(xt)

    err1[i] = torch.abs(.5*torch.norm(y - out)**2 - f0)
    err2[i] = torch.abs(.5*torch.norm(y - out)**2 - f0 - hh*J)
    print(hh, ", ", err1[i], ", ", hh*J, ", ", err2[i])


def convw(w):
    c2.weight = w
    return torch.norm(c2(xt))**2 

from IPython import embed; embed()