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


data_size = (224, 224)
data_root = '/Dataset'   # '../Dataset/Kaggle265'
target_size = (224, 224)
target_root = '/Targetset'   # '../Dataset/Kaggle265'

# Initiate dataset and dataset transform
data_pre_transforms = {
    'train': transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
    ]),
    'test': transforms.Compose([
        transforms.Resize(image_size),
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

loader = DJloader(patch_dim, gap, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)


# 警告：伪代码施工现场！

model_all = models.resnet18(pretrained=True)
model_ft = nn.Sequential(*(list(model_all.children())[:-1]))
# model 替换为 swin - Transformer

trans_D = model_ft(data)
trans_T = model_ft(target)

outputs = torch.nn.functional.conv2d(trans_D, trans_T)
pred_ = torch.softmax(outputs) * [0, 1, 2, 3, ...]
label = ()

loss = torch.nn.MSELoss(pred_, label)