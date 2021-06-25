from aug_mission import *

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision.transforms as transforms
from torchvision import datasets, models
import matplotlib.pyplot as plt

import os
import argparse
import random
import numpy as np
import warnings

from PIL import Image
plt.ion()  # interactive mode
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'    # opt.cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # "cpu"   

data_size = (1024, 1024)
data_root = './Dataset'   # '../Dataset/Kaggle265'
target_size = (256, 256)
target_root = './Targetset'   # '../Dataset/Kaggle265'

# Initiate dataset and dataset transform
data_pre_transforms = transforms.Compose([
    transforms.Resize(data_size),
    transforms.RandomHorizontalFlip(),
])
data_post_transforms = transforms.Compose([
    transforms.ToTensor(),
])

patch_dim = 256
jitter = 8
batch_size = 16
num_workers = 0
num_epoch = 1

loader = aug_loader(patch_dim, data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)

# 警告：伪代码施工现场！

# model_all = models.resnet18(pretrained=True)
# model_ft = nn.Sequential(*(list(model_all.children())[:-1])).to(device)
# print(model_ft)
model_ft = MyNet().to(device)
layer_soft = SpatialSoftmax(1024, 1024, device=device)
# model 替换为 swin - Transformer

optimizer = optim.SGD([
    {'params': model_ft.parameters(), 'lr': 1e-3, 'momentum': 0.99, 'weight_decay': 1e-3},
])
criterion = torch.nn.MSELoss()

for epoch in range(num_epoch):
    for _, (target, data, label_x, label_y) in enumerate(tqdm(loader)):
        # imshow(target[0,0])
        # imshow(data[0,0])
        target = target.to(device)
        data = data.to(device)
        label_x = label_x.to(device)
        label_y = label_y.to(device)
        
        trans_T = model_ft(target)
        trans_D = model_ft(data)
        print(trans_T.shape)
        print(trans_D.shape)

        outputs = F.conv2d(input=trans_D, weight=trans_T)
        pred_x, pred_y = layer_soft(outputs)

        loss = criterion(pred_x, label_x) + criterion(pred_y, label_y)
        optimizer.zero_grad()
        loss.backward()
