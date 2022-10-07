import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

eps = 1e-10
noise = 0.3


class MLP(nn.Module):
    def __init__(self, layer_sizes=[784, 1000, 500, 250, 250, 250, 10]):
        super(MLP, self).__init__()
        self.layer_sizes = layer_sizes
        self.nlayer = len(self.layer_sizes)-1
        self.fcs = nn.ModuleList([nn.Linear(self.layer_sizes[l], self.layer_sizes[l+1], bias=False) for l in range(self.nlayer)])
        self.bn = nn.ModuleList([nn.BatchNorm1d(self.layer_sizes[l+1], momentum=0.1, affine=False) for l in range(self.nlayer)])
    def forward(self, x, rt_layers=False):
        layers = []
        indices = []
        x = x.view(-1, self.layer_sizes[0])
        layers.append(x)
        for i in range(self.nlayer):
            x = self.fcs[i](x)
            layers.append(x)
            if i < self.nlayer-1:
                # x = self.bn[i](x)
                x = F.relu(x)
            # else:
                # x = self.bn[i](x)

        if rt_layers:
            return layers, indices
        else:
            return x


class CNN_MNIST(nn.Module):
    def __init__(self, bn_affine=False):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32, affine=bn_affine)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32, affine=bn_affine)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64, affine=bn_affine)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64, affine=bn_affine)

        self.fc1 = nn.Linear(64 * 7 * 7, 200)
        # self.drop1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(200, 10)

    def forward(self, x, rt_layers=False):
        layers = []
        indices = []
        x = x.view(-1, 1, 28, 28)
        layers.append(x)
        x = self.bn1(self.conv1(x))
        layers.append(x)
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        layers.append(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)

        x = self.bn3(self.conv3(x))
        layers.append(x)
        x = F.relu(x)

        x = self.bn4(self.conv4(x))
        layers.append(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)

        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        layers.append(x)
        x = F.relu(x)

        # x = self.drop1(x)
        x = self.fc2(x)
        layers.append(x)
        if rt_layers:
            return layers, indices
        else:
            return x


class CNN_CIFAR10(nn.Module):
    def __init__(self, bn_affine=False):
        super(CNN_CIFAR10, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64, affine=bn_affine)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64, affine=bn_affine)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128, affine=bn_affine)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128, affine=bn_affine)


        self.fc1 = nn.Linear(128 * 8 * 8, 200)
        self.drop1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(200, 10)

    def forward(self, x, rt_layers=False):
        layers = []
        indices = []
        x = x.view(-1, 3, 32, 32)
        layers.append(x)
        x = self.bn1(self.conv1(x))
        layers.append(x)
        x = F.relu(x)

        x = self.bn2(self.conv2(x))
        layers.append(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)

        x = self.bn3(self.conv3(x))
        layers.append(x)
        x = F.relu(x)

        x = self.bn4(self.conv4(x))
        layers.append(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)

        x = x.view(-1, 128 * 8 * 8)
        x = self.fc1(x)
        layers.append(x)
        x = F.relu(x)

        # x = self.drop1(x)
        x = self.fc2(x)
        layers.append(x)
        if rt_layers:
            return layers, indices
        else:
            return x


class TailClassifier(nn.Module):
    def __init__(self, sizes=(128*8*8,)):
        super(TailClassifier, self).__init__()
        self.sizes = sizes
        self.fc1 = nn.Linear(sizes[0], 200)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, self.sizes[0])
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = self.fc2(x)
        return x

    def layerwise(self, x):
        layers = []
        x = x.view(-1, self.sizes[0])
        x = F.relu(self.fc1(x))
        layers.append(x)
        x = self.drop1(x)
        x = self.fc2(x)
        layers.append(x)
        return layers


def FeaNorm(x, alpha):
    m = x.mean(dim=1, keepdim=True)
    v = (x - m).pow(2).mean(dim=1, keepdim=True)
    x = (x - m) / v.sqrt() - alpha
    return x


class CNNFea(nn.Module):
    def __init__(self, alpha=None):
        super(CNNFea, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 8 * 8, 200)
        self.drop1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(200, 10)
        self.alpha = alpha

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)

        x = self.conv1(x)
        # x = self.bn1(x)
        x = FeaNorm(x, alpha=self.alpha)
        x = F.relu(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = FeaNorm(x, alpha=self.alpha)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        # x = self.bn3(x)
        x = FeaNorm(x, alpha=self.alpha)
        x = F.relu(x)

        x = self.conv4(x)
        # x = self.bn4(x)
        x = FeaNorm(x, alpha=self.alpha)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 128 * 8 * 8)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x

    def layerwise(self, x):
        layers = []

        x = x.view(-1, 3, 32, 32)
        x = self.conv1(x)
        # x = self.bn1(x)
        x = FeaNorm(x, alpha=self.alpha)
        x = F.relu(x)
        layers.append(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = FeaNorm(x, alpha=self.alpha)
        x = F.relu(x)
        layers.append(x)

        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        # x = self.bn3(x)
        x = FeaNorm(x, alpha=self.alpha)
        x = F.relu(x)
        layers.append(x)

        x = self.conv4(x)
        # x = self.bn4(x)
        x = FeaNorm(x, alpha=self.alpha)
        x = F.relu(x)
        layers.append(x)

        x = F.max_pool2d(x, 2)

        x = x.view(-1, 128 * 8 * 8)
        x = self.fc1(x)
        x = F.relu(x)
        layers.append(x)
        x = self.drop1(x)
        x = self.fc2(x)
        layers.append(x)
        return layers


class CNNFeaAll(nn.Module):
    def __init__(self, alpha=None):
        super(CNNFeaAll, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 8 * 8, 200)
        self.drop1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(200, 10)
        self.alpha = alpha

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)

        x = self.conv1(x)
        # x = self.bn1(x)
        x = FeaNorm(x, alpha=self.alpha)
        x = F.relu(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = FeaNorm(x, alpha=self.alpha)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        # x = self.bn3(x)
        x = FeaNorm(x, alpha=self.alpha)
        x = F.relu(x)

        x = self.conv4(x)
        # x = self.bn4(x)
        x = FeaNorm(x, alpha=self.alpha)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 128 * 8 * 8)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x

    def layerwise(self, x):
        layers = []

        x = x.view(-1, 3, 32, 32)
        x = self.conv1(x)
        # x = self.bn1(x)
        x = FeaNorm(x, alpha=self.alpha)
        x = F.relu(x)
        layers.append(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = FeaNorm(x, alpha=self.alpha)
        x = F.relu(x)
        layers.append(x)

        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        # x = self.bn3(x)
        x = FeaNorm(x, alpha=self.alpha)
        x = F.relu(x)
        layers.append(x)

        x = self.conv4(x)
        # x = self.bn4(x)
        x = FeaNorm(x, alpha=self.alpha)
        x = F.relu(x)
        layers.append(x)

        x = F.max_pool2d(x, 2)

        x = x.view(-1, 128 * 8 * 8)
        x = self.fc1(x)
        x = FeaNorm(x, alpha=self.alpha)
        x = F.relu(x)
        layers.append(x)
        x = self.drop1(x)
        x = self.fc2(x)
        layers.append(x)
        return layers



