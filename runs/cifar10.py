from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

import sls
import resnet
import dla

from pyxconv import convert_net
import concurrent.futures

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def Net(ps=0):
    net = resnet.resnet20()
    if ps == 0:
        return net
    conv1 = net.conv1
    convert_net(net, 'net', mode='conv', ps=ps)
    net.conv1 = conv1
    return net

def nll_loss(model, x, y0, tracker, backward=True):
    y = model(x)
    loss = F.cross_entropy(y, y0)
    tracker['nfwd'] += 1
    loss.backward()
    return loss


def train(args, model, device, train_loader, optimizer, epoch, writer, tracker):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = nll_loss(model, data, target, tracker)
        optimizer.step()
        tracker['niter'] += 1
        writer.add_scalar('Loss/train', loss, tracker['niter'])
        if batch_idx % 25 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, ind, writer):
    model.eval()
    test_loss = 0
    correct = 0
    confusion_matrix = torch.zeros((10,10), dtype=torch.int)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            for i, l in enumerate(target):
                confusion_matrix[l.item(), pred[i].item()] += 1


    test_loss /= len(test_loader.dataset)
    writer.add_scalar('Loss/test', test_loss, ind)
    writer.add_scalar('Accuracy/test', correct / len(test_loader.dataset), ind)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    for i, r in enumerate(confusion_matrix):
        writer.add_scalar(f'Classes/{classes[i]}', r[i]/torch.sum(r)*100, ind)

    return test_loss, correct / len(test_loader.dataset)


def cifar_train(b: int, ps: int, args, cudaid: int) -> None:
    print(f"({b}, {ps}): Training mnist")
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print(f"({b}, {ps}): using cuda, {use_cuda}")
    device = torch.device(f"cuda:{cudaid}" if use_cuda else "cpu")
    print(f"({b}, {ps}): Training on device {device}")
    writer = SummaryWriter(log_dir=f"./cifar_bench_sgd/{b}_{ps}")
    
    args.lr = 2*args.lr if ps>0 else args.lr 
    
    train_kwargs = {'batch_size': b}
    test_kwargs = {'batch_size': b}
    print(f"({b}, {ps}): setup cuda kargs")
    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    print(f"({b}, {ps}): setup transforms")
    dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                       transform=train_transform)
    dataset2 = datasets.CIFAR10('../data', train=False,
                       transform=test_transform)
    print(f"({b}, {ps}): load datasets")
    train_sampler = torch.utils.data.RandomSampler(dataset1, replacement=False)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=b, sampler=train_sampler, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=b, shuffle=False, num_workers=2)
    print(f"({b}, {ps}): split test and train")
    model = Net(ps).to(device)
    
    #lr = args.lr if ps == 0 else args.lr * 3
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    print(f"({b}, {ps}): start training")
    tracker = {'nfwd': 0, 'ngrad': 0, 'niter': 0}
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer, tracker)
        tloss, tacc = test(model, device, test_loader, (epoch-1)*len(train_loader), writer)
        #scheduler.step()
        #torch.save(model, f"./cifar_bench_sgd/cifar_10_{b}_{ps}.pt")

def train_batch(b:int, args, cudaid: int, ps: int) -> None:
    cifar_train(b, ps, args, cudaid)

def bench(ndevice, args) -> None:
    for ps in[[0, 32]]:
        with concurrent.futures.ProcessPoolExecutor(max_workers=ndevice) as executor:
            futures = {executor.submit(train_batch, 128, args, i, p) for i, p in enumerate(ps)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result() 
                except Exception as exc:
                    print('generated an exception: %s' % (exc))

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    
    bench(2, args)
