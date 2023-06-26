# 在该文件NeuralNetwork类中定义你的模型 
# 在自己电脑上训练好模型，保存参数，在这里读取模型参数（不要使用JIT读取），在main中返回读取了模型参数的模型

import os

os.system("sudo pip3 install torch")
os.system("sudo pip3 install torchvision")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.data import DataLoader
    

class NeuralNetwork(nn.Module):
        def __init__(self,input_channels):
        super().__init__()
        # 第1个卷积层
        self.conv1 = nn.Conv2d(input_channels, 96, kernel_size=11, stride=4)
        # 第1个池化层
        self.pooling1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 第2个卷积层
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        # 第2个池化层
        self.pooling2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 第3个卷积层
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        # 第4个卷积层
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        # 第5个卷积层
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        # 第3个池化层
        self.pooling3 = nn.MaxPool2d(kernel_size=3, stride=2)

        ##最后的三个FC
        self.Flatten = nn.Flatten(start_dim=1,end_dim=-1)
        # 计算得出的当前的前面处理过后的shape，当然也可print出来以后再确定
        self.Linear1 = nn.Linear(6400, 4096)
        self.drop1 = nn.Dropout(p = 0.5)
        self.Linear2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout(p = 0.5)
        self.Linear3 = nn.Linear(4096, 10)

    def forward(self,X):
        X = self.pooling1(F.relu(self.conv1(X)))
        X = self.pooling2(F.relu(self.conv2(X)))
        X = F.relu(self.conv3(X))
        X = F.relu(self.conv4(X))
        X = F.relu(self.conv5(X))
        X = self.pooling3(X)
        X = X.view(X.size()[0], -1)
        X = self.drop1(F.relu(self.Linear1(X)))
        X = self.drop2(F.relu(self.Linear2(X)))
        X = F.relu(self.Linear3(X))

        return X

def read_data():
    # 这里可自行修改数据预处理，batch大小也可自行调整
    # 保持本地训练的数据读取和这里一致
    dataset_train = torchvision.datasets.CIFAR10(root='../data/exp03', train=True, download=True, transform=torchvision.transforms.ToTensor())
    dataset_val = torchvision.datasets.CIFAR10(root='../data/exp03', train=False, download=False, transform=torchvision.transforms.ToTensor())
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=256, shuffle=False)
    return dataset_train, dataset_val, data_loader_train, data_loader_val

def main():
    model = NeuralNetwork() # 若有参数则传入参数
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model.load_state_dict(torch.load(parent_dir + '/pth/model.pth'))
    return model
    
