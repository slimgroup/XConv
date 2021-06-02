from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import argparse

import networks

from pyxconv import convert_net


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def Net(ps=0, xmode='indepentent'):
    net = networks.CIFARConvNet()
    if ps == 0:
        return net
    convert_net(net, 'net', ps, mode='all', xmode=xmode)
    return net


def train(args, model, device, train_loader, optimizer, epoch, writer, tracker, ps):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        #if batch_idx % 25 == 0:
        #    update_mode(model, 'features')
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        pred = model(data)

        loss = F.nll_loss(pred, target)
        if torch.isnan(loss):
            continue
        loss.backward()
        optimizer.step()

        tracker['niter'] += 1
        pred = pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        writer.add_scalar('Loss/train', loss, tracker['niter'])
        if batch_idx % 25 == 0:
        #    update_mode(model, 'all')
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    writer.add_scalar(f'Accuracy/train', correct / len(train_loader.dataset), epoch)
    print('Train set Accuracy: {}/{} ({:.0f}%)'.format(correct, len(train_loader.dataset),
              100. * correct / len(train_loader.dataset)))

    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def test(model, device, test_loader, ind, writer):
    model.eval()
    test_loss = 0
    correct = 0
    confusion_matrix = torch.zeros((10,10), dtype=torch.int)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            for i, l in enumerate(target):
                confusion_matrix[l.item(), pred[i].item()] += 1


        test_loss /= len(test_loader.dataset)
        writer.add_scalar(f'Loss/test', test_loss, ind)
        writer.add_scalar(f'Accuracy/test', correct / len(test_loader.dataset), ind)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
              test_loss, correct, len(test_loader.dataset),
              100. * correct / len(test_loader.dataset)))

        for i, r in enumerate(confusion_matrix):
            writer.add_scalar(f'Classes/{classes[i]}', r[i]/torch.sum(r)*100, ind)


def cifar_train(b: int, ps: int, xmode, args, cudaid: int)-> None:
    print(f"({b}, {ps}): Training cifar10")
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print(f"({b}, {ps}): using cuda, {use_cuda}")
    device = torch.device(f"cuda:{cudaid}" if use_cuda else "cpu")
    print(f"({b}, {ps}): Training on device {device}")
    writer = SummaryWriter(log_dir=f"./cifar_bench_sgd/{b}_{ps}_{args.lr:.4f}")
    
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
    model = Net(ps, xmode).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=3e-5)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"({b}, {ps}): start training")
    
    tracker = {'n_fwd': 0, 'n_bck': 0, 'niter': 0}
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer, tracker, ps)
        test(model, device, test_loader, epoch, writer)
        scheduler.step()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    
    cifar_train(128, 0, None, args, 0)
    args.lr *= 1.5
    cifar_train(256, 32, 'indepentent', args, 0)
    cifar_train(256, 256, 'gaussian', args, 0)
    cifar_train(256, 256, 'orthogonal', args, 0)
