# _*_ coding : UTF-8 _*_
import argparse
import sys

import torch
import torchvision
import torchvision.transforms as transforms
from torch.backends import cudnn
from torch.utils.data import DataLoader

import time

from models.resnet import resnet18

CIFAR_PATH = './dataset'
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
    # Experiment setting
    parser.add_argument('-net', type=str, required=True, help='net type')
    args = parser.parse_args()

    return args

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

    args = parse_args()

    print('==> Preparing data..')
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    cifar10_testing = torchvision.datasets.CIFAR10(root=CIFAR_PATH, train=False, download=True,
                                                     transform=transform_test)
    testloader = torch.utils.data.DataLoader(cifar10_testing, batch_size=100, shuffle=False, num_workers=4)

    print('==> Building model..')
    net = get_network(args)
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    WEIGHT_PATH = './checkpoint/cifar10-' + str(args.net)+'.pth'
    # print(WEIGHT_PATH)
    # path='./checkpoint/cifar100-resnet18.pth'
    checkpoint = torch.load(WEIGHT_PATH)
    net.load_state_dict(checkpoint['net'])
    net.eval()

    correct = 0.0
    total = 0

    start_time=time.time()
    with torch.no_grad():
        for n_iter, (image, label) in enumerate(testloader):
            # print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(testloader)))

            image,label=image.to(device),label.to(device)

            output = net(image)
            _, pred = output.max(1)

            correct += pred.eq(label).sum().item()


    end_time=time.time()
    print("Time cost:",end_time-start_time)

    acc=correct / len(testloader.dataset)
    print("Accuracy: {:.4f}".format(acc))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))

