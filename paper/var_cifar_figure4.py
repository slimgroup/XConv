from pyxconv import *
import torch
from torchvision import datasets, transforms
from networks import CIFARConvNet

import copy
import matplotlib.pyplot as plt
import numpy as np

train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
dataset1 = datasets.CIFAR10('../data', train=True, download=True, transform=train_transform)

net = CIFARConvNet()

batches =  [64, 128, 256, 1024]
p_sizes = [0, 64, 256, 512]

grads1 = {'%s%s'%(b, ps): [] for b in batches for ps in p_sizes}
grads2 = {'%s%s'%(b, ps): [] for b in batches for ps in p_sizes}
grads3 = {'%s%s'%(b, ps): [] for b in batches for ps in p_sizes}
grads4 = {'%s%s'%(b, ps): [] for b in batches for ps in p_sizes}
grads = [grads1, grads2, grads3, grads4]

models = {}
device = torch.device("cuda")

for ps in p_sizes:
    model = copy.deepcopy(net)
    convert_net(model, 'net', mode='conv' if ps>0 else 'relu', ps=ps, xmode="gaussian")
    models[ps] = model.to(device)


std = torch.zeros(len(p_sizes), len(batches), 4)

def convs(model, ps):
    return [getattr(models[ps], 'conv%s'%i) for i in [1,2,3,4]]

for bb, b in enumerate(batches):
    loader = torch.utils.data.DataLoader(dataset1, shuffle=True, batch_size=b)

    for pp, ps in enumerate(p_sizes):
        if ps >= 0:
            for i, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)
                pred = models[ps](data)
                loss = F.nll_loss(pred, target)
                loss.backward()
            
                for g, c in zip(grads, convs(models, ps)):
                    g['%s%s'%(b, ps)].append(c.weight.grad/b)
                if i > 40:
                   break

            for i, g in enumerate(grads):
                stdl, meanl = torch.std_mean(torch.stack(g['%s%s'%(b, ps)]))
                print(f"{ps}, {b}, conv{i+1}: Standard deviation {stdl}, mean {meanl}")
                std[pp, bb, i] = stdl


np.save("stdis_40.npy", std.detach().numpy())

x = np.arange(len(batches))
width = 0.15  # the width of the bars

fig, ax = plt.subplots(2, 2, figsize=(15, 10))
for i, axi in enumerate(ax.flatten()):
    for pp, ps in enumerate(p_sizes):
        lab = "True" if ps == 0 else f'r={ps}'
        rect = axi.bar(x -(pp-2)*width, std[pp, :, i], label=lab, width=width)
    axi.set_ylabel('Standard deviation')
    axi.set_xlabel('Batch size')
    axi.set_title(f'conv{i+1}')
    axi.set_xticks(x)
    axi.set_xticklabels(batches)
    axi.legend()

fig.tight_layout()
fig.savefig("var_conv_40.pdf", bbox_inches="tight", dpi=150)
