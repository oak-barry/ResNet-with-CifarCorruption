# _*_ coding : UTF-8 _*_
import argparse
import pickle
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]


class GetLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    def __len__(self):
        return len(self.data)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100C Co-Testing')
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


def show_performance(distortion_name):
    PATH = "./dataset/CIFAR-100-C/" + distortion_name + ".npy"

    source_data = np.load(PATH).transpose((0, 3, 1, 2))
    source_data = torch.from_numpy(source_data).to(dtype=torch.float32) / 255
    # normalize (using broadcast)
    for tmp in source_data:
        # print(tmp[0].size())
        tmp[0] = (tmp[0] - mean[0]) / std[0]
        tmp[1] = (tmp[1] - mean[1]) / std[1]
        tmp[2] = (tmp[2] - mean[2]) / std[2]

    # print(source_data.size())

    source_label = np.load("./dataset/CIFAR-100-C/labels.npy")
    source_label = torch.from_numpy(source_label)
    # print(source_label.type())

    torchdata = GetLoader(source_data, source_label)
    testset = torch.utils.data.DataLoader(torchdata, batch_size=100, shuffle=False, num_workers=4)
    # print(len(testset))

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for batch_index, (image, label) in enumerate(testset):
            image, label = image.to(device), label.to(device)
            # print(inputs.size())
            # print(targets.size())
            output1 = net1(image)
            output2 = net2(image)
            outputs = (output1 + output2) / 2
            # print(outputs)

            _, pred = outputs.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            # print(label)
            correct = pred.eq(label.long()).float()

            # compute top 5
            correct_5 += correct[:, :5].sum()

            # compute top1
            correct_1 += correct[:, :1].sum()

    err1 = 1 - correct_1 / len(testset.dataset)
    err5 = 1 - correct_5 / len(testset.dataset)

    return err1.to('cpu').numpy(), err5.to('cpu').numpy()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = parse_args()

    # load model
    net1 = get_network(args.net1)
    net1 = net1.to(device)
    net2 = get_network(args.net2)
    net2 = net2.to(device)

    if device == 'cuda':
        net1 = torch.nn.DataParallel(net1)
        net2 = torch.nn.DataParallel(net2)
        cudnn.benchmark = True

    WEIGHT_PATH1 = './checkpoint/cifar100-' + str(args.net1) + '.pth'
    WEIGHT_PATH2 = './checkpoint/cifar100-' + str(args.net2) + '.pth'
    checkpoint1 = torch.load(WEIGHT_PATH1)
    net1.load_state_dict(checkpoint1['net'])
    net1.eval()  # very critical!
    checkpoint2 = torch.load(WEIGHT_PATH2)
    net2.load_state_dict(checkpoint2['net'])
    net2.eval()  # very critical!

    distortions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
    ]

    time_start = time.time()
    error_rates1 = []
    error_rates2 = []
    for distortion_name in distortions:
        rate1, rate2 = show_performance(distortion_name)
        error_rates1.append(rate1)
        error_rates2.append(rate2)
        print('Distortion: {:15s}  | CE1[Top 1 err] (%): {:.2f}'.format(distortion_name, 100 * rate1))
        print('            {:15s}  | CE2[Top 5 err] (%): {:.2f}'.format(' ', 100 * rate2))
        print()

    time_end = time.time()
    print('mCE1 (%): {:.2f}'.format(100 * np.mean(error_rates1)))
    print('mCE2 (%): {:.2f}'.format(100 * np.mean(error_rates2)))
    print('Time cost:', time_end - time_start, "s")
