from .aug_mission import *

import torch
import torch.nn as nn
import torch.nn.parallel
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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'    # opt.cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_size = (224, 224)
data_root = '/Dataset'   # '../Dataset/Kaggle265'
target_size = (96, 96)
target_root = '/Targetset'   # '../Dataset/Kaggle265'

# Initiate dataset and dataset transform
data_pre_transforms = {
    'train': transforms.Compose([
        transforms.Resize(data_size),
        transforms.RandomHorizontalFlip(),
    ]),
    'test': transforms.Compose([
        transforms.Resize(data_size),
    ]),
}
data_post_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6086, 0.4920, 0.4619], std=[0.2577, 0.2381, 0.2408])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6086, 0.4920, 0.4619], std=[0.2577, 0.2381, 0.2408])
    ]),
}

patch_dim = 96
jitter = 8
batch_size = 256
num_workers = 4
num_epoch = 100

loader = aug_loader(patch_dim, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)

# 警告：伪代码施工现场！

model_all = models.resnet18(pretrained=True)
model_ft = nn.Sequential(*(list(model_all.children())[:-1]))
# model 替换为 swin - Transformer

optimizer = optim.SGD()
criterion = torch.nn.MSELoss()

for epoch in range(num_epoch):
    for _, (target, data, label_x, label_y) in enumerate(tqdm(loader)):
        target = target.to(device)
        data = data.to(device)
        label_x = label_x.to(device)
        label_y = label_y.to(device)
        
        trans_T = model_ft(target)
        trans_D = model_ft(data)

        outputs = torch.nn.functional.conv2d(trans_D, trans_T)
        pred_x = torch.softmax(outputs) * [0, 1, 2, 3, ...]
        pred_y = torch.softmax(outputs) * [0, 1, 2, 3, ...]

        loss = criterion(pred_x, label_x) + criterion(pred_y, label_y)
        optimizer.zero_grad()
        loss.backward()
