from __future__ import print_function

import argparse
import os
import multiprocessing
import torch
import torchvision
from torch import optim, nn
from torch.backends import cudnn
from torchvision import transforms

from models import DLA
from train import train, test

# Training settings

parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--optimizers', type=str, default='Adam', metavar='OP',
                    help='optimizers for training (default: Adam)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr_decay', type=float, default=0.0, metavar='LRDECAY',
                    help='lr_decay (default: 0.0)')
parser.add_argument('--weight_decay', type=float, default=0.0, metavar='WEIGHTDECAY',
                    help='weight_decay (default: 0.0)')
parser.add_argument('--dampening', type=int, default=0, metavar='DAMPENING',
                    help='dampening (default: 0)')
parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                    help='momentum (default: 0.0)')
parser.add_argument('--alpha', type=float, default=0.99, metavar='ALPHA',
                    help='alpha(default: 0.99)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--resume', '-r', action='store_true', default=False,
                    help='resume from checkpoint')
args = parser.parse_args()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_processors = multiprocessing.cpu_count()
torch.set_num_threads(num_processors)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Set device
def prepare_device():
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def prepare_data():
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    return trainloader, testloader


def build_model(_device, _net):
    # net
    print('==> Building net..')
    _net = _net.to(_device)
    if _device == 'cuda':
        _net = torch.nn.DataParallel(_net)
    cudnn.benchmark = True
    return _net


criterion = nn.CrossEntropyLoss()


def prepare_optimizer(_net):
    print('==> Preparing optimizers...')
    optimizers_dict = {
        'Adam': optim.Adam(_net.parameters(), lr=args.lr, weight_decay=args.weight_decay),
        'SGD': optim.SGD(_net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                         dampening=args.dampening),
        'Adagrad': optim.Adagrad(_net.parameters(), lr=args.lr, lr_decay=args.lr_decay,
                                 weight_decay=args.weight_decay),
        'RMSprop': optim.RMSprop(_net.parameters(), lr=args.lr, alpha=args.alpha, momentum=args.momentum,
                                 weight_decay=args.weight_decay),
        'Rprop': optim.Rprop(_net.parameters(), lr=args.lr),
        'Adadelta': optim.Adadelta(_net.parameters(), lr=args.lr, weight_decay=args.weight_decay),
    }
    _optimizer = optimizers_dict.get(args.optimizers, optim.Adam(_net.parameters(), lr=args.lr))
    print('==> Preparing scheduler...')
    _scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer, T_max=200)
    return _optimizer, _scheduler


def run(train_loader, test_loader, _device, _criterion, _optimizer, _scheduler):
    for k in args.__dict__:
        print(str(k) + '------' + str(args.__dict__[k]))
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train(epoch, _optimizer, net, train_loader, _device, _criterion)
        test(epoch, net, test_loader, _device, _criterion)
        _scheduler.step()


if __name__ == '__main__':
    device = prepare_device()
    net = build_model(device, DLA())
    optimizer, scheduler = prepare_optimizer(net)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    trainloader, testloader = prepare_data()
    run(trainloader, testloader, device, criterion, optimizer, scheduler)
