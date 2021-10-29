from __future__ import print_function

import argparse
import logging
import multiprocessing
import os
import queue
from logging import handlers
from logging.handlers import QueueHandler, QueueListener

import torch
import torchvision
from torch import optim, nn
from torch.backends import cudnn
from torchvision import transforms

from models import DLA
from train import train, test, init_best_acc

que = queue.Queue(-1)
queue_handler = QueueHandler(que) # no limit on size

logger = logging.getLogger(__name__)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('--env', type=str, default='dev', metavar='DIR',
                    help='input data dir (default ./data)')
parser.add_argument('--data_dir', type=str, default='./data', metavar='DIR',
                    help='input data dir (default ./data)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--optimizers', type=str, default='Adam', metavar='OP',
                    help='optimizers for training (default: Adam)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--predict-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--end_epoch', type=int, default=128, metavar='ENDEPOCHS',
                    help='number of end-epochs to train (default: 10)')
parser.add_argument('--test', action='store_true', default=False,
                    help='quickly test')
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
torch.set_num_threads(num_processors * 2)
IMAGE_SIZE = 32


def prepare_log():
    # logging
    rh = logging.handlers.RotatingFileHandler(filename='./logs/cifar.log', mode='a', maxBytes=204800, backupCount=7)
    logging.basicConfig(
        level=logging.getLevelName('INFO'),
        format='%(asctime)s.%(msecs)03d [%(levelname)s] %(process)d %(name)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[rh],
    )
    _listener = QueueListener(que, rh)
    return _listener


# Set device
def prepare_device():
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def prepare_data(data_dir, _device_nums):
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
        root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test)
    _testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=_device_nums * 6 if _device_nums > 0 else 4,
        pin_memory=(args.env != 'dev'))
    _trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=_device_nums * 6 if _device_nums > 0 else 4,
        pin_memory=(args.env != 'dev'))
    return _trainloader, _testloader


def build_model(_device, _net):
    # net
    print('==> Building net..')
    _net = _net.to(_device)
    _device_nums = torch.cuda.device_count()
    if _device == 'cuda':
        if _device_nums > 1:
            _net = torch.nn.DataParallel(_net)
    cudnn.benchmark = True
    return _net, _device_nums


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
        'AdamW': optim.AdamW(_net.parameters(), lr=args.lr, weight_decay=args.weight_decay),
    }
    _optimizer = optimizers_dict.get(args.optimizers, optim.Adam(_net.parameters(), lr=args.lr))
    print('==> Preparing scheduler...')
    _scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer, T_max=200)
    return _optimizer, _scheduler


def run(_net, train_loader, test_loader, _device, _criterion, _optimizer, _scheduler, _best_acc, _start_epoch):
    for k in args.__dict__:
        print(str(k) + '------' + str(args.__dict__[k]))
    init_best_acc(_best_acc)
    if not args.test:
        for epoch in range(_start_epoch, args.end_epoch):
            train(_net, epoch, _optimizer, train_loader, _device, _criterion)
            test(_net, epoch, test_loader, _device, _criterion)
            _scheduler.step()
    else:
        test(_net, start_epoch, test_loader, _device, _criterion)


if __name__ == '__main__':
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir('outputs'):
        os.mkdir('outputs')
    listener = prepare_log()
    listener.start()
    device = prepare_device()
    net, device_nums = build_model(device, DLA())
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth', map_location=torch.device(device))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    optimizer, scheduler = prepare_optimizer(net)
    trainloader, testloader = prepare_data(args.data_dir, device_nums)
    run(net, trainloader, testloader, device, criterion, optimizer, scheduler, best_acc, start_epoch)
    listener.stop()
