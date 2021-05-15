from pyxconv import *
from pyxconv.modules import Xconv2D

import numpy as np

import copy
from torchvision import datasets, transforms
import cifarconvnet

import matplotlib.pyplot as plt

# independent
# r x ci x batch
# all
# 2r x batch i.e ci/2 cheaper


ci, co, b, k, ps = 3, 3, 256, 5, 64

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
r2 = Xconv2D(ci, co, k, ps=ps, mode='features', bias=False, stride=1, padding=2)
r3 = Xconv2D(ci, co, k, ps=4*ps, mode='all', bias=False, stride=1, padding=2)
r2.weight = copy.deepcopy(r1.weight)
r3.weight = copy.deepcopy(r1.weight)

xc = iter(train_loader).next()[0]
xr = torch.randn(xc.shape)

def ni(inp):
    # print(torch.norm(inp, float('inf')))
    n = 1#torch.norm(inp, float('inf'))
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
        elif nameout == 'C10':
            y = iter(train_loader).next()[0]
            y1 = r1(inp)
            g1 = y1.grad_fn(y)
            y2 = r2(inp)
            g2 = y2.grad_fn.apply(y)
            y3 = r3(inp)
            g3 = y3.grad_fn.apply(y)
        else:
            y = torch.randn(xc.shape)
            y1 = r1(inp)
            g1 = y1.grad_fn(y)
            y2 = r2(inp)
            g2 = y2.grad_fn.apply(y)
            y3 = r3(inp)
            g3 = y3.grad_fn.apply(y)

        plt.subplot(2,3,i)
        plt.plot(ni(g1[1])[:200], label="true")
        plt.plot(ni(g2[1])[:200], label="per feature")
        plt.plot(ni(g3[1])[:200], label="full")
        plt.title(r"$x \in {}, y \in  {}$".format(namein, nameout))
        i += 1

plt.legend()
plt.tight_layout()
plt.show()


net = cifarconvnet.CIFARConvNet()
# torch.nn.init.constant_(net.fc5.weight, 1/10)

net2 = copy.deepcopy(net)
net3 = copy.deepcopy(net)
convert_net(net2, 'net', ps=ps, mode='all', xmode='features')
convert_net(net3, 'net', ps=4*ps, mode='all', xmode='all')

xt, target = iter(train_loader).next()

y1 = net(xt)
y2 = net2(xt)
y3 = net3(xt)


net.train()
loss = F.nll_loss(y1, target)
loss.backward()

net2.train()
loss2 = F.nll_loss(y2, target)
loss2.backward()

net3.train()
loss3 = F.nll_loss(y3, target)
loss3.backward()

print(y1 - y2)
print(y1 - y3)


def get_gw(net, c, n=200):
    return ni(getattr(net, c).weight.grad.reshape(-1)[:200])

plt.figure(figsize=(12, 8))
for i, c in enumerate([f'conv{i}' for i in range(1, 5)]):
    plt.subplot(2,2,i+1)
    plt.plot(get_gw(net, c, n=200), label="true")
    plt.plot(get_gw(net2, c, n=200), label="per feature")
    plt.plot(get_gw(net3, c, n=200), label="full")
    plt.title(c)
plt.legend()
plt.tight_layout()
plt.show()
