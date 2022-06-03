# ResNet with CifarCorruption
This repo is a combination of the original two repos 'ResNet-on-Cifar10-Cifar10C' and 'ResNet-on-Cifar100-Cifar100C'. And this project is part of my undergraduate graduation design. As an undergraduate student new to deep learning and pytorch, the code looks like some poor. And there are a lot of similarity between them. Therefore it is possible to bring them together. Maybe I will do this later.

# ResNet-on-Cifar10-Cifar10C
We train resnet on cifar10, test on it to get the accuracy and test on cifar10c which is a dataset with corruption to test the robustness of the model. Besides, we try to ensemble two different resnet and combine their outputs to test on cifar10 and cifar10c in order to konw if network ensemble is capable of improving the accuracy and robustness of the original model.

# ResNet-on-Cifar100-Cifar100C
The only difference between the above one is just that this uses the cifar100 and cifar100c dataset. The network and procedure are the same.

# Reference
1. The resnet.py follows 'https://github.com/kuangliu/pytorch-cifar';
2. The cifar10 and cifar100 datasets are downloaded from 'Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.';
3. The cifar10c and cifar100c datasets are provided by 'Hendrycks, Dan and Thomas G. Dietterich. “Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.” ArXiv abs/1903.12261 (2019): n. pag.'.

If you feel this repo is helpful for you, please star it. ^-^