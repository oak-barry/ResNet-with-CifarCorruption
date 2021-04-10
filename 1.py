# _*_ coding : UTF-8 _*_
# 开发人员 ：BarryCao
# 开发时间 ：2021/3/24 15:05
# 文件名称 ：1.PY
# 开发工具 ：PyCharm
import torch
import numpy as np

'''
net=resnet18().to('cuda')
net = torch.nn.DataParallel(net)
net.load_state_dict(torch.load("./checkpoint/cifar100-resnet18.pth")['net'])
acc=torch.load("./checkpoint/cifar100-resnet18.pth")['acc']
epoch=torch.load("./checkpoint/cifar100-resnet18.pth")['epoch']
state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        }
torch.save(state, './checkpoint/cifar100-resnet18.pth',_use_new_zipfile_serialization=False)
'''

'''
err1=torch.tensor([1]).to('cuda')
err5=torch.tensor([5]).to('cuda')
print(err1)
print(err5)
err1 = err1.to('cpu').numpy()
err5 = err5.to('cpu').numpy()
print(err1)
print(err5)

error_rates1 = []
error_rates2 = []
error_rates1.append(err1)
error_rates2.append(err5)
print(np.mean(error_rates1))
print(np.mean(error_rates2))
'''
correct=4.3
acc=correct / 5
print("Accuracy: {:.4f}".format(acc))