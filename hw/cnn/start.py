from __future__ import print_function

import argparse
import logging
import multiprocessing
import os

import torch
import torchvision
from azureml.core.run import Run
from concurrent_log_handler import ConcurrentRotatingFileHandler
from torch import optim, nn
from torch.backends import cudnn
from torchvision import transforms

from models import DPN92
from train import train, test, init_best_acc

run_context = Run.get_context()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('--env',
                    type=str,
                    default='dev',
                    metavar='DIR',
                    help='input data dir (default ./data)')
parser.add_argument('--data_dir',
                    type=str,
                    default='../data/cifar10/',
                    metavar='DIR',
                    help='input data dir (default ./data)')
parser.add_argument('--batch-size',
                    type=int,
                    default=128,
                    metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--optimizers',
                    type=str,
                    default='Adam',
                    metavar='OP',
                    help='optimizers for training (default: Adam)')
parser.add_argument('--test-batch-size',
                    type=int,
                    default=1000,
                    metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--predict-batch-size',
                    type=int,
                    default=128,
                    metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--end_epochs',
                    type=int,
                    default=128,
                    metavar='ENDEPOCHS',
                    help='number of end-epochs to train (default: 10)')
parser.add_argument('--test',
                    action='store_true',
                    default=False,
                    help='quickly test')
parser.add_argument('--lr',
                    type=float,
                    default=0.01,
                    metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr_decay',
                    type=float,
                    default=0.0,
                    metavar='LRDECAY',
                    help='lr_decay (default: 0.0)')
parser.add_argument('--weight_decay',
                    type=float,
                    default=0.0,
                    metavar='WEIGHTDECAY',
                    help='weight_decay (default: 0.0)')
parser.add_argument('--dampening',
                    type=int,
                    default=0,
                    metavar='DAMPENING',
                    help='dampening (default: 0)')
parser.add_argument('--momentum',
                    type=float,
                    default=0.0,
                    metavar='M',
                    help='momentum (default: 0.0)')
parser.add_argument('--alpha',
                    type=float,
                    default=0.99,
                    metavar='ALPHA',
                    help='alpha(default: 0.99)')
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    metavar='S',
                    help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes',
                    type=int,
                    default=4,
                    metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--cuda',
                    action='store_true',
                    default=True,
                    help='enables CUDA training')
parser.add_argument('--dry-run',
                    action='store_true',
                    default=False,
                    help='quickly check a single pass')
parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    default=False,
                    help='resume from checkpoints')
args = parser.parse_args()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoints epoch
num_processors = multiprocessing.cpu_count()
torch.set_num_threads(num_processors * 2)
IMAGE_SIZE = 32


def prepare_log():
    # send all messages, for demo; no other level or filter logic applied.
    file_handler = ConcurrentRotatingFileHandler(
        filename='./logs/cifar.log',
        mode='a',
        maxBytes=1024 * 1024 * 500,
        backupCount=7,
        encoding='utf-8',
    )
    logging.basicConfig(
        level=logging.getLevelName('INFO'),
        format=
        '%(asctime)s.%(msecs)03d [%(levelname)s] %(process)d %(name)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[file_handler],
    )


# Set device
def prepare_device():
    use_cuda = args.cuda and torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")


def prepare_data(data_dir, _device_nums):
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=data_dir,
                                            train=True,
                                            download=False,
                                            transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir,
                                           train=False,
                                           download=False,
                                           transform=transform_test)
    _testloader = torch.utils.data.DataLoader(testset,
                                              batch_size=100,
                                              shuffle=False,
                                              num_workers=_device_nums *
                                              6 if _device_nums > 0 else 4,
                                              pin_memory=(args.env != 'dev'))
    _trainloader = torch.utils.data.DataLoader(trainset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=_device_nums *
                                               6 if _device_nums > 0 else 4,
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
        'Adam':
        optim.Adam(_net.parameters(),
                   lr=args.lr,
                   weight_decay=args.weight_decay),
        'SGD':
        optim.SGD(_net.parameters(),
                  lr=args.lr,
                  momentum=args.momentum,
                  weight_decay=args.weight_decay,
                  dampening=args.dampening),
        'Adagrad':
        optim.Adagrad(_net.parameters(),
                      lr=args.lr,
                      lr_decay=args.lr_decay,
                      weight_decay=args.weight_decay),
        'RMSprop':
        optim.RMSprop(_net.parameters(),
                      lr=args.lr,
                      alpha=args.alpha,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay),
        'Rprop':
        optim.Rprop(_net.parameters(), lr=args.lr),
        'Adadelta':
        optim.Adadelta(_net.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay),
        'AdamW':
        optim.AdamW(_net.parameters(),
                    lr=args.lr,
                    weight_decay=args.weight_decay),
    }
    _optimizer = optimizers_dict.get(args.optimizers,
                                     optim.Adam(_net.parameters(), lr=args.lr))
    print('==> Preparing scheduler...')
    _scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer,
                                                            T_max=200)
    return _optimizer, _scheduler


def run(_net, train_loader, test_loader, _device, _criterion, _optimizer,
        _scheduler, _best_acc, _start_epoch):
    for k in args.__dict__:
        run_config = str(k) + '--' + str(args.__dict__[k])
        print(run_config)
    run_context.log_table('Runing Configuration', args.__dict__)
    init_best_acc(_best_acc)
    if not args.test:
        for epoch in range(_start_epoch, args.end_epochs):
            train(_net, epoch, _optimizer, train_loader, _device, _criterion)
            test(_net, epoch, test_loader, _device, _criterion)
            _scheduler.step()
    else:
        test(_net, start_epoch, test_loader, _device, _criterion)


if __name__ == '__main__':
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.isdir('outputs'):
        os.mkdir('outputs')
    prepare_log()
    device = prepare_device()
    net, device_nums = build_model(device, DPN92())
    if args.resume:
        # Load checkpoints.
        print('==> Resuming from checkpoints..')
        assert os.path.isdir(
            'checkpoints'), 'Error: no checkpoints directory found!'
        checkpoints = torch.load('./checkpoints/ckpt.pth',
                                 map_location=torch.device(device))
        net.load_state_dict(checkpoints['net'])
        best_acc = checkpoints['acc']
        start_epoch = checkpoints['epoch']
    optimizer, scheduler = prepare_optimizer(net)
    trainloader, testloader = prepare_data(args.data_dir, device_nums)
    run(net, trainloader, testloader, device, criterion, optimizer, scheduler,
        best_acc, start_epoch)
