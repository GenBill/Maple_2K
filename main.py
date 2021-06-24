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


# 警告：伪代码施工现场！

model_ft = models.resnet18(pretrained=True)
# model 替换为 swin - Transformer

trans_D = model_ft(data)
trans_T = model_ft(target)

outputs = torch.nn.functional.conv2d(trans_D, trans_T)
pred_ = torch.nn.functional.argsoftmax(outputs)
label = ()

loss = torch.nn.MSELoss(pred_, label)