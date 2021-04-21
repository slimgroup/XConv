from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

import sls

from xconv import Xconv2D
import concurrent.futures

class Net(nn.Module):
    def __init__(self, ps):
        super(Net, self).__init__()
        if ps == 0:
            self.conv1 = nn.Conv2d(1, 32, 3, bias=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, bias=1, padding=1)
        else:
            self.conv1 = Xconv2D(1, 32, 3, ps, bias=1, padding=1)
            self.conv2 = Xconv2D(32, 64, 3, ps, bias=1, padding=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def nll_loss(model, x, y0, tracker, backward=True):
    y = model(x)
    loss = F.nll_loss(y, y0)
    tracker['nfwd'] += 1
    if backward and loss.requires_grad:
        loss.backward()

    return loss


def train(args, model, device, train_loader, optimizer, epoch, writer, tracker):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        closure = lambda : nll_loss(model, data, target, tracker, backward=False)
        loss = optimizer.step(closure)
        tracker['niter'] += 1
        writer.add_scalar('Loss/train', loss, tracker['niter'])
        writer.add_scalar('Count/forward', tracker['nfwd'], tracker['niter'])
        writer.add_scalar('Step/stepsize', optimizer.state['step_size'], tracker['niter'])
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


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
    writer.add_scalar('Loss/test', test_loss, ind)
    writer.add_scalar('Accuracy/test', correct / len(test_loader.dataset), ind)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, correct / len(test_loader.dataset)


def mnist_train(b: int, ps: int, args, cudaid: int) -> None:
    print(f"({b}, {ps}): Training mnist")
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print(f"({b}, {ps}): using cuda, {use_cuda}")
    device = torch.device(f"cuda:{cudaid}" if use_cuda else "cpu")
    print(f"({b}, {ps}): Training on device {device}")
    writer = SummaryWriter(log_dir=f"./mnist_bench/{b}_{ps}")
    
    args.lr = 2*args.lr if ps>0 else args.lr 
    
    train_kwargs = {'batch_size': b}
    test_kwargs = {'batch_size': b}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    print(f"({b}, {ps}): setup cuda kargs")
    nm = [transforms.Normalize((0.1307,), (0.3081,))]
    transformations = [transforms.ToTensor()]
    transformations.extend(nm if ps == 0 else [])
    transform=transforms.Compose(transformations)
    print(f"({b}, {ps}): setup transforms")
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    print(f"({b}, {ps}): load datasets")
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    print(f"({b}, {ps}): split test and train")
    model = Net(ps).to(device)
    
    optimizer = sls.SlsEg(model.parameters(), init_step_size=args.lr,
                          n_batches_per_epoch=len(train_loader))

    print(f"({b}, {ps}): start training")
    tracker = {'nfwd': 0, 'ngrad': 0, 'niter': 0}
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer, tracker)
        tloss, tacc = test(model, device, test_loader, (epoch-1)*len(train_loader), writer)

def train_batch(b:int, args, cudaid: int) -> None:
    for ps in [0]+[2**i for i in range(4, 9)]:
        mnist_train(b, ps, args, cudaid)

def bench(ndevice, args) -> None:
    b_sizes = [2**i for i in range(6, 12)]
    for chunk in [b_sizes[i:i + ndevice] for i in range(0, len(b_sizes), ndevice)]:
        with concurrent.futures.ProcessPoolExecutor(max_workers=ndevice) as executor:
            futures = {executor.submit(train_batch, b, args, i%ndevice) for i, b in enumerate(chunk)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result() 
                except Exception as exc:
                    print('generated an exception: %s' % (exc))

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=2.0, metavar='LR',
                        help='learning rate (default: 2.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    bench(4, args)
