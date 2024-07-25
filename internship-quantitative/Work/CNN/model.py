from torch.utils.data import DataLoader
from dataset import dataset
from torchvision.transforms import ToTensor,Resize,Normalize,Compose
from torchvision.io import read_image
from early_stop import earlystop
from importlib import reload
from torch.optim.adamw import AdamW
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import logging
reload(logging)
import re
import pandas as pd
import numpy as np
from torch.nn.functional import relu,softmax
import time


class model(nn.Module):#VGG模型
    def __init__(self):
        super(model,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=64,padding=1,kernel_size=3)
        self.norm1=nn.BatchNorm2d(64)
        self.conv2=nn.Conv2d(in_channels=64,out_channels=64,padding=1,kernel_size=3)
        self.norm2=nn.BatchNorm2d(64)
        self.pool1=nn.MaxPool2d(kernel_size=2)
        self.norm3=nn.BatchNorm2d(64)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,padding=1,kernel_size=3)
        self.norm4=nn.BatchNorm2d(128)
        self.conv4=nn.Conv2d(in_channels=128,out_channels=128,padding=1,kernel_size=3)
        self.norm5=nn.BatchNorm2d(128)
        self.pool2=nn.MaxPool2d(kernel_size=2)
        self.norm6=nn.BatchNorm2d(128)
        self.conv5=nn.Conv2d(in_channels=128,out_channels=256,padding=1,kernel_size=3)
        self.norm7=nn.BatchNorm2d(256)
        self.conv6=nn.Conv2d(in_channels=256,out_channels=256,padding=1,kernel_size=3)
        self.norm8=nn.BatchNorm2d(256)
        self.conv7=nn.Conv2d(in_channels=256,out_channels=256,padding=1,kernel_size=3)
        self.norm9=nn.BatchNorm2d(256)
        self.pool3=nn.MaxPool2d(kernel_size=2)
        self.norm10=nn.BatchNorm2d(256)
        self.conv8=nn.Conv2d(in_channels=256,out_channels=512,padding=1,kernel_size=3)
        self.norm11=nn.BatchNorm2d(512)
        self.conv9=nn.Conv2d(in_channels=512,out_channels=512,padding=1,kernel_size=3)
        self.norm12=nn.BatchNorm2d(512)
        self.conv10=nn.Conv2d(in_channels=512,out_channels=512,padding=1,kernel_size=3)
        self.norm13=nn.BatchNorm2d(512)
        self.pool4=nn.MaxPool2d(kernel_size=2)
        self.norm14=nn.BatchNorm2d(512)
        self.conv11=nn.Conv2d(in_channels=512,out_channels=512,padding=1,kernel_size=3)
        self.norm15=nn.BatchNorm2d(512)
        self.conv12=nn.Conv2d(in_channels=512,out_channels=512,padding=1,kernel_size=3)
        self.norm16=nn.BatchNorm2d(512)
        self.pool5=nn.MaxPool2d(kernel_size=2)
        self.fc1=nn.Linear(512*10*10,512)
        self.norm17=nn.BatchNorm1d(512)
        self.out1=nn.Dropout()
        self.fc2=nn.Linear(512,512)
        self.norm18=nn.BatchNorm1d(512)
        self.out2=nn.Dropout()
        self.fc3=nn.Linear(512,3)
        
    def forward(self,x):
        x=self.norm1(relu(self.conv1(x)))
        x=self.norm2(relu(self.conv2(x)))
        x=self.norm3(self.pool1(x))
        x=self.norm4(relu(self.conv3(x)))
        x=self.norm5(relu(self.conv4(x)))
        x=self.norm6(self.pool2(x))
        x=self.norm7(relu(self.conv5(x)))
        x=self.norm8(relu(self.conv6(x)))
        x=self.norm9(relu(self.conv7(x)))
        x=self.norm10(self.pool3(x))
        x=self.norm11(relu(self.conv8(x)))
        x=self.norm12(relu(self.conv9(x)))
        x=self.norm13(relu(self.conv10(x)))
        x=self.norm14(self.pool4(x))
        x=self.norm15(relu(self.conv11(x)))
        x=self.norm16(relu(self.conv12(x)))
        x=self.pool5(x)
        x=x.view(-1,512*10*10)
        x=self.out1(self.norm17(self.fc1(x)))
        x=self.out2(self.norm18(self.fc2(x)))
        x=self.fc3(x)
        x=x.view(-1,3)
        x=softmax(x,dim=1)
        
        return x
