# _*_ coding : UTF-8 _*_
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

import os
import argparse
import time
import sys


CIFAR_PATH = 'dataset'
LOG_DIR = 'runs'
MILESTONES = [60, 120, 160]
# time of we run the script
DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    # Experiment setting
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', type=int, default=200,
                        help="total epochs for training")
    parser.add_argument('-bs', type=int, default=128, help='batch size for training dataloader')
    parser.add_argument('--workers', type=int, default=4,
                        help="number of worker to load data")
    args = parser.parse_args()

    return args


def train(epoch):
    start = time.time()
    net.train()

    correct = 0
    total = 0
    for batch_index, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        n_iter = (epoch - 1) * len(trainloader) + batch_index + 1
        # ???
        '''
        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
        '''
        if batch_index % 100 == 0:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.bs + len(images),
                total_samples=len(trainloader.dataset)
            ))

        # update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)
        writer.add_scalar('Train/accuracy', 100.0*correct/total, n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
    finish = time.time()

    print('Epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


def validate(epoch, tb=True):
    # switch to evaluate mode
    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

    finish = time.time()
    '''
    if torch.cuda.is_available():
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    '''
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(testloader.dataset),
        correct / len(testloader.dataset),
        finish - start
    ))

    # add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(testloader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct / len(testloader.dataset), epoch)

    return correct / len(testloader.dataset)


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_network(args):
    # return given network
    if args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return net

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    args = parse_args()
    set_seed()

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # 数据增强
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    cifar100_training = torchvision.datasets.CIFAR100(root=CIFAR_PATH, train=True, download=True,
                                                      transform=transform_train)
    trainloader = torch.utils.data.DataLoader(cifar100_training, batch_size=args.bs, shuffle=True,
                                              num_workers=args.workers)

    cifar100_testing = torchvision.datasets.CIFAR100(root=CIFAR_PATH, train=False, download=True,
                                                     transform=transform_test)
    testloader = torch.utils.data.DataLoader(cifar100_testing, batch_size=100, shuffle=False, num_workers=args.workers)

    # Model
    print('==> Building model..')
    net = get_network(args)
    net = net.to(device)

    # sys.exit()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES,
                                                     gamma=0.2)

    # use tensorboard
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    writer = SummaryWriter(log_dir=os.path.join(
        LOG_DIR, args.net, TIME_NOW))
    # network architechture
    input_tensor = torch.Tensor(1, 3, 32, 32)
    input_tensor = input_tensor.to(device) # remember to set x = x.to()
    writer.add_graph(net, input_tensor)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        acc = validate(epoch)
        scheduler.step()
        # Save checkpoint
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/cifar100-'+str(args.net)+'.pth',_use_new_zipfile_serialization=False)
            best_acc = acc
            print('Best acc is %.4f' % best_acc)
            print()

    writer.close()
