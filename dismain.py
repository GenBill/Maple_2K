from aim_mission import *

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
from tensorboardX import SummaryWriter, writer

import os
import argparse
import random
import numpy as np
import warnings
from PIL import Image
plt.ion()  # interactive mode
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'    # opt.cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     # "cpu" #

datawriter = SummaryWriter()
data_root = './Dataset'   # '../Dataset/Kaggle265'
target_root = './Targetset'   # '../Dataset/Kaggle265'

num_workers = 0
step = 8
halfstep = int(step*3/2)

loader = aim_loader(data_root, target_root, num_workers)
# model_all = models.resnet50(pretrained=True)
model_all = models.resnet152(pretrained=True)
model_ft = nn.Sequential(
    *(list(model_all.children())[:-1]),
    nn.Flatten()
).to(device)

criterion = torch.nn.MSELoss()
# criterion = torch.nn.L1Loss()

model_ft.eval()
with torch.no_grad():
    for num_iter, (data, target, data_size, target_size) in enumerate(loader):
        target = target.to(device)
        data = data.to(device)
        data_size = data_size.item()
        target_size = target_size.item()

        min_loss = 1e4
        min_i, min_j = -1, -1

        for i in tqdm(range(0, data_size-target_size, step)):
            for j in range(0, data_size-target_size, step):
                trans_T = model_ft(target)
                trans_D = model_ft(data[:,:,i:i+target_size,j:j+target_size])
                loss = criterion(trans_T, trans_D).item()
                if min_loss>loss:
                    min_i, min_j = i, j
                    min_loss = loss
        
        if step>1:

            head_i = max(0, min_i-halfstep)
            head_j = max(0, min_j-halfstep)
            tail_i = min(min_i+halfstep, data_size-target_size)
            tail_j = min(min_j+halfstep, data_size-target_size)
        
            for i in range(head_i, tail_i):
                for j in range(head_j, tail_j):
                    trans_T = model_ft(target)
                    trans_D = model_ft(data[:,:,i:i+target_size,j:j+target_size])
                    loss = criterion(trans_T, trans_D).item()
                    # loss = torch.mean(torch.square(criterion(trans_T, trans_D))).item()
                    if min_loss>loss:
                        min_i, min_j = i, j
                        min_loss = loss

        data[0,:,min_i:min_i+target_size,min_j:min_j+target_size] = target[0,:,:,:]
        x, y = get_position(min_i, min_j, data_size, target_size)
        
        datawriter.add_image('new_img', data[0,:,:,:], num_iter)
        datawriter.add_scalar('img_loss', min_loss, num_iter)
        datawriter.add_scalars('position', {'x': x, 'y': y}, num_iter)
        
        print('Iter : {}'.format(num_iter))
        print('Pos = ({}, {})'.format(x, y))
        print('Loss = {}'.format(min_loss))
datawriter.close()