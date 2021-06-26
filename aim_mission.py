from tqdm import tqdm
import numpy as np
import math
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision import datasets, models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import matplotlib.pyplot as plt
import cv2

import os
import time
import copy
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def imshow(img):
    npimg = img.detach().numpy()
    plt.imshow(npimg)
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_position(min_i, min_j, data_size, target_size):
    x = min_i + (target_size - data_size)//2
    y = min_j + (target_size - data_size)//2
    return x, y

class Aimset(Dataset):

    def __init__(self, data_dir, target_dir):
        self.data_dir = data_dir
        self.target_dir = target_dir

        self.dataset = os.listdir(self.data_dir)
        self.targetset = os.listdir(self.target_dir)
        # self.dataset = datasets.ImageFolder(self.image_paths, self.preTransform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # print(index)
        data_index = '/RefImg_' + str(index) + '.bmp'
        target_index = '/RealImg_' + str(index) + '.bmp'
        data_path = os.path.join(self.data_dir+data_index)          # 获取图像的路径或目录
        target_path = os.path.join(self.target_dir+target_index)    # 获取图像的路径或目录
        data = Image.open(data_path).convert('RGB')                 # 读取图像
        target = Image.open(target_path).convert('RGB')             # 读取图像
        
        data_size = min(data.size)
        target_size = min(target.size)
        
        # temp = cv2.medianBlur(np.array(data), 3)
        # data = Image.fromarray(temp)
        # temp = cv2.medianBlur(np.array(target), 3)
        # # temp = cv2.fastNlMeansDenoisingColored(temp, None, 10,10,7,21)
        # target = Image.fromarray(temp)

        data = transforms.CenterCrop(data_size)(data)
        data = transforms.ToTensor()(data)
        target = transforms.CenterCrop(target_size)(target)
        target = transforms.ToTensor()(target)

        return data, target, data_size, target_size

def aim_loader(data_root, target_root, num_workers):
    image_datasets = Aimset(data_root, target_root) 
    assert image_datasets
    dataloaders = torch.utils.data.DataLoader(
            image_datasets, batch_size=1,
            pin_memory=True, shuffle=False, num_workers=num_workers
        )
    return dataloaders
