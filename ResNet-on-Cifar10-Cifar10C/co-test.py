# _*_ coding : UTF-8 _*_
import argparse
import sys

import torch
import torchvision
import torchvision.transforms as transforms
from torch.backends import cudnn
from torch.utils.data import DataLoader

import time

CIFAR_PATH = './dataset'
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Co-Testing')
    # Experiment setting
    parser.add_argument('-net1', type=str, required=True, help='net1 type')
    parser.add_argument('-net2', type=str, required=True, help='net2 type')
    args = parser.parse_args()

    return args


def get_network(args):
    # return given network
    if args == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return net


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = parse_args()

    print('==> Preparing data..')
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    cifar10_testing = torchvision.datasets.CIFAR10(root=CIFAR_PATH, train=False, download=True,
                                                     transform=transform_test)
    testloader = torch.utils.data.DataLoader(cifar10_testing, batch_size=100, shuffle=False, num_workers=4)

    print('==> Building model..')
    net1 = get_network(args.net1)
    net1 = net1.to(device)
    net2 = get_network(args.net2)
    net2 = net2.to(device)

    if device == 'cuda':
        net1 = torch.nn.DataParallel(net1)
        net2 = torch.nn.DataParallel(net2)
        cudnn.benchmark = True

    WEIGHT_PATH1 = './checkpoint/cifar10-' + str(args.net1) + '.pth'
    WEIGHT_PATH2 = './checkpoint/cifar10-' + str(args.net2) + '.pth'
    # print(WEIGHT_PATH)
    # path='./checkpoint/cifar100-resnet18.pth'
    checkpoint1 = torch.load(WEIGHT_PATH1)
    net1.load_state_dict(checkpoint1['net'])
    net1.eval()
    checkpoint2 = torch.load(WEIGHT_PATH2)
    net2.load_state_dict(checkpoint2['net'])
    net2.eval()

    correct = 0.0
    total = 0

    start_time = time.time()
    with torch.no_grad():
        for n_iter, (image, label) in enumerate(testloader):
            # print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(testloader)))

            image, label = image.to(device), label.to(device)

            output1 = net1(image)
            output2 = net2(image)
            output = (output1 + output2) / 2

            _, pred = output.max(1)

            correct += pred.eq(label).sum().item()


    end_time = time.time()
    print("Time cost:", end_time - start_time)

    acc = correct / len(testloader.dataset)
    print("Accuracy: {:.4f}".format(acc))
# print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
