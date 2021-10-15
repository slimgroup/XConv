from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

# import sls
import networks

from pyxconv import convert_net
import concurrent.futures

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def Net(ps=0, xmode='gaussian'):
    net = networks.MnistNet()
    if ps == 0:
        return net
    convert_net(net, 'net', ps, mode='all', xmode=xmode)
    return net


def sls_closure(model, images, labels, backwards=False):
    logits = model(images)
    loss = F.nll_loss(logits, labels)

    if backwards and loss.requires_grad:
        loss.backward()

    return loss


def train(args, model, device, train_loader, optimizer, epoch, writer, tracker, ps):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        closure = lambda : sls_closure(model, data, target, backwards=False)
        loss = optimizer.step(closure)

        tracker['niter'] += 1
        writer.add_scalar('Loss/train', loss, tracker['niter'])
        if batch_idx % 25 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def test(model, device, test_loader, ind, writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        writer.add_scalar(f'Loss/test', test_loss, ind)
        writer.add_scalar(f'Accuracy/test', correct / len(test_loader.dataset), ind)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
              test_loss, correct, len(test_loader.dataset),
              100. * correct / len(test_loader.dataset)))


def mnist_train(b: int, ps: int, args, cudaid: int)-> None:
    print(f"({b}, {ps}): Training mnist")
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print(f"({b}, {ps}): using cuda, {use_cuda}")
    device = torch.device(f"cuda:{cudaid}" if use_cuda else "cpu")
    print(f"({b}, {ps}): Training on device {device}")
    writer = SummaryWriter(log_dir=f"./mnist_bench_sls/{b}_{ps}")
    
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    print(f"({b}, {ps}): setup transforms")
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                              transform=train_transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=test_transform)
    print(f"({b}, {ps}): load datasets")
    train_sampler = torch.utils.data.RandomSampler(dataset1, replacement=False)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=b, sampler=train_sampler, num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=b, shuffle=False, num_workers=2)

    print(f"({b}, {ps}): split test and train")
    model = Net(ps).to(device)
    
    # optimizer = sls.SlsEg(model.parameters(), init_step_size=1, n_batches_per_epoch=len(train_loader))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=3e-5)
    print(f"({b}, {ps}): start training")
    
    tracker = {'n_fwd': 0, 'n_bck': 0, 'niter': 0}
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer, tracker, ps)
        test(model, device, test_loader, epoch, writer)


def par_train(args) -> None:
    all_p = [0, 16, 32, 64, 128, 256]
    all_b = [64, 128, 256, 512, 1024, 2048]
    all_cases = torch.tensor([(b, ps) for b in all_b for ps in all_p])
    for c in torch.split(all_cases, args.ndevice):
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.ndevice) as executor:
            futures = {executor.submit(mnist_train, bi, pi, args, i%args.ndevice) for i, (bi, pi) in enumerate(c)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result() 
                except Exception as exc:
                    print('generated an exception: %s' % (exc))


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--ndevice', type=int, default=1,
                        help='Number of cuda devices')
    args = parser.parse_args()
    
    par_train(args)
