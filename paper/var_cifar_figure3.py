from pyxconv import *
from pyxconv.modules import Xconv2D

import numpy as np

import copy
from torchvision import datasets, transforms
import networks

import matplotlib.pyplot as plt

torch.manual_seed(17)

class ComputeStats(object):
    """
    Inputs samples and on-the-fly compute weighted first- and second-order
    statistics
    """
    def __init__(self):
        super(ComputeStats, self).__init__()

        self.sum_of_weights = 0.0
        self.N = 0

        self.weighted_sum = 0.0
        self.weighted_sum_of_squares = 0.0

    def input_samples(self, samples, weights=None):
        """
        Input new samples and update the weighted sum and sum of squares
        """
        if weights == None:
            weights = np.ones(samples.shape[0])
        assert samples.shape[0] == weights.shape[0]

        sum_of_weights = np.sum(weights, axis=0)

        self.weighted_sum += np.average(samples,
                                        weights=weights,
                                        axis=0)*sum_of_weights

        self.weighted_sum_of_squares += np.average(samples**2,
                                                   weights=weights,
                                                   axis=0)*sum_of_weights

        self.sum_of_weights += sum_of_weights

        self.N += samples.shape[0]

    def compue_stats(self):
        """
        Return the weighted mean and standard deviation based on the
        intermediate weighted sum and sum of squares
        """

        sample_mean = self.weighted_sum/self.sum_of_weights

        sample_var = (self.weighted_sum_of_squares
                      - self.weighted_sum**2/self.sum_of_weights)
        sample_var *= self.N/((self.N-1) * self.sum_of_weights)

        return sample_mean, np.sqrt(sample_var)


ci, co, b, k, ps = 3, 3, 320, 5, 2048



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

xc = iter(train_loader).next()[0]
xr = torch.randn(xc.shape)

def ni(inp):
    # print(torch.norm(inp, float('inf')))
    n = 1#torch.norm(inp, float('inf'))
    return (inp/n).cpu().detach().numpy().reshape(-1)

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
            y = iter(train_loader).next()[0]
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


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

net0 = networks.CIFARConvNet().to(device)

xt, target = iter(train_loader).next()
xt, target = xt.to(device), target.to(device)
g = {}


mem_gain_fac = 2.5
train_loader_ = torch.utils.data.DataLoader(range(b), batch_size=int(b//mem_gain_fac), shuffle=True)

def get_gw(net, c, n=100):
    return ni(getattr(net, c).weight.grad.reshape(-1)[0:n:2])

for mode in [None, 'independent', 'gaussian', 'orthogonal']:
    net = copy.deepcopy(net0)
    if mode is not None:
        convert_net(net, 'net', ps=ps, mode='all', xmode=mode)

        y = net(xt)

        net.train()
        loss = F.nll_loss(y, target, reduction='mean')
        loss.backward()
        for c in[f'conv{i}' for i in range(1, 5)]:
            g["%s%s"%(str(mode), c)] = get_gw(net, c, n=200)

    else:

        for itr in range(100):

            idxs = iter(train_loader_).next()
            xt_, target_ = xt[idxs], target[idxs]

            y_ = net(xt_)

            net.train()
            loss = F.nll_loss(y_, target_, reduction='mean')
            loss.backward()
            for c in[f'conv{i}' for i in range(1, 5)]:
                g["%s%s%s"%(str(mode), c, itr)] = get_gw(net, c, n=200)

            net.zero_grad()


fig = plt.figure(figsize=(12, 8))
for i, c in enumerate([f'conv{i}' for i in range(1, 5)]):
    mean = 0.0
    stats = ComputeStats()
    plt.subplot(2, 2, i+1)

    for itr in range(100):
        stats.input_samples(g[f'None{c}{itr}'].reshape(1, -1))

    sample_mean, sample_std = stats.compue_stats()

    plt.fill_between(range(sample_mean.shape[0]),
                      sample_mean - 2.576*sample_std,
                      sample_mean + 2.576*sample_std,
                      color="#6e6e6e", alpha=0.4,
                      label=r"$99$% interval")

    plt.plot(sample_mean, "k", label="True", linewidth=2.0,
             alpha=1)

    plt.plot(g[f'independent{c}'], label="Indep.", alpha=.8, linewidth=2.0)
    plt.plot(g[f'gaussian{c}'], label="Multi", alpha=.8, linewidth=2.0)
    plt.plot(g[f'orthogonal{c}'], label="Multi-Orth", alpha=.8, linewidth=2.0)

    plt.xticks([])
    plt.title(c)

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels,loc='lower center', ncol=5, fontsize=18,
           bbox_to_anchor=(0.55, -0.035))

plt.tight_layout()
plt.savefig(f"./figures/c4ifarfirst_ps-{ps}_b-{b}.png",
            bbox_inches='tight', dpi=200,
                    pad_inches=.05)
