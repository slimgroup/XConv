from pyxconv import *
from pyxconv.modules import Xconv2D

import numpy as np

import copy
from torchvision import datasets, transforms
import cifarconvnet

import matplotlib.pyplot as plt

torch.manual_seed(123)

ci, co, b, k, ps = 3, 3, 128, 5, 512

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
dataset1 = datasets.CIFAR10('../data', train=True, download=True, transform=train_transform)
train_sampler = torch.utils.data.RandomSampler(dataset1, replacement=False)
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=b, sampler=train_sampler)

r1 = torch.nn.Conv2d(ci, co, k, bias=False, stride=1, padding=2)
r2 = Xconv2D(ci, co, k, ps=ps, mode='independent', bias=False, stride=1, padding=2)
r3 = Xconv2D(ci, co, k, ps=ps, mode='Gaussian', bias=False, stride=1, padding=2)
r4 = Xconv2D(ci, co, k, ps=ps, mode='Orthogonal', bias=False, stride=1, padding=2)
r2.weight = copy.deepcopy(r1.weight)
r3.weight = copy.deepcopy(r1.weight)
r4.weight = copy.deepcopy(r1.weight)

xc = next(iter(train_loader))[0]
xr = torch.randn(xc.shape)

def ni(inp):
    n = 1
    return (inp/n).detach().numpy().reshape(-1)

i=1
plt.figure(figsize=(12, 8))
for (inp, namein) in zip([xc, xr], ['C10', '\mathcal{N}(0, 1)']):
    for nameout in ['\mathcal{N}(0, 1)', 'C(x)', 'C10']:
        print(namein, nameout)
        if nameout == 'C(x)':
            y1 = r1(inp)
            g1 = y1.grad_fn(y1)
            y2 = r2(inp)
            g2 = y2.grad_fn.apply(y2)
            y3 = r3(inp)
            g3 = y3.grad_fn.apply(y3)
            y4 = r4(inp)
            g4 = y4.grad_fn.apply(y4)
        elif nameout == 'C10':
            y = next(iter(train_loader))[0]
            y1 = r1(inp)
            g1 = y1.grad_fn(y)
            y2 = r2(inp)
            g2 = y2.grad_fn.apply(y)
            y3 = r3(inp)
            g3 = y3.grad_fn.apply(y)
            y4 = r4(inp)
            g4 = y4.grad_fn.apply(y)
        else:
            y = torch.randn(xc.shape)
            y1 = r1(inp)
            g1 = y1.grad_fn(y)
            y2 = r2(inp)
            g2 = y2.grad_fn.apply(y)
            y3 = r3(inp)
            g3 = y3.grad_fn.apply(y)
            y4 = r4(inp)
            g4 = y4.grad_fn.apply(y)

        plt.subplot(2,3,i)
        plt.plot(ni(g1[1])[:200], label="true")
        plt.plot(ni(g2[1])[:200], label="per feature")
        plt.plot(ni(g3[1])[:200], label="full")
        plt.plot(ni(g4[1])[:200], label="ortho")
        plt.title(r"$x \in {}, y \in  {}$".format(namein, nameout))
        i += 1

plt.legend()
plt.tight_layout()
# plt.show()


net0 = cifarconvnet.CIFARConvNet()
xt, target = next(iter(train_loader))
g = {}

def get_gw(net, c, n=100):
    return ni(getattr(net, c).weight.grad.reshape(-1)[0:n:2])

for mode in [None, 'independent', 'gaussian', 'orthogonal']:
    net = copy.deepcopy(net0)
    if mode is not None:
        convert_net(net, 'net', ps=ps, mode='all', xmode=mode)

    y = net(xt)

    net.train()
    loss = F.nll_loss(y, target)
    loss.backward()
    for c in[f'conv{i}' for i in range(1, 5)]:
        g["%s%s"%(str(mode), c)] = get_gw(net, c, n=200)


fig = plt.figure(figsize=(16, 8))
for i, c in enumerate([f'conv{i}' for i in range(1, 5)]):
    plt.subplot(2, 2, i+1)
    plt.plot(g[f'None{c}'], "-*k", label="True", linewidth=2)
    plt.plot(g[f'independent{c}'], label="Indep.")
    plt.plot(g[f'gaussian{c}'], label="Multi")
    plt.plot(g[f'orthogonal{c}'], label="Multi-Ortho")
    plt.xticks([])
    plt.title(c)

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc='lower center', ncol=4)

plt.tight_layout()
plt.savefig(f"./figures/c4ifarfirst_{ps}.png", bbox_inches="tight")
plt.show()

from IPython import embed; embed()
