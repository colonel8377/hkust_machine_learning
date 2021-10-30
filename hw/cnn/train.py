# Training
import logging
import os

import torch

from utils import progress_bar
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)

best_acc = 0
writer = SummaryWriter('./run')


def init_best_acc(_best_acc):
    global best_acc
    best_acc = _best_acc


def train(model, epoch, op, trainloader, device, criterion):
    print('\nEpoch: %d' % epoch)
    logger.info('Epoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        op.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        op.step()
        train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        writer.add_scalar("Loss/train", loss, total, epoch)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        if total % 100 == 0:
            logger.info('Train: Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    writer.flush()


def test(model, epoch, testloader, device, criterion):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    if not os.path.isdir('log'):
        os.mkdir('log')
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            writer.add_scalar("Loss/test", loss, total, epoch)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            if total % 100 == 0:
                logger.info('Test: Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        writer.flush()
    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, './checkpoint/ckpt.pth')
        torch.save(state, './outputs/ckpt.pth')
        best_acc = acc
