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

class Selfset(Dataset):

    def __init__(self, path_dir, patch_dim, preTransform=None, postTransform=None):
        self.path_dir = path_dir
        self.patch_dim = patch_dim

        self.preTransform = preTransform
        self.postTransform = postTransform
        self.dataset = os.listdir(self.path_dir)
        # self.dataset = datasets.ImageFolder(self.image_paths, self.preTransform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image_index = self.dataset[index]
        image_path = os.path.join(self.path_dir, image_index)       # 获取图像的路径或目录
        image = Image.open(image_path)         # .convert('RGB')    # 读取图像
        image = self.preTransform(image)

        uniform_patch_x_coord = int(math.floor((image.size[0] - self.patch_dim) * random.random()))
        uniform_patch_y_coord = int(math.floor((image.size[1] - self.patch_dim) * random.random()))
                    
        # 必要模块：数据增强
        # self.prep_patch(image)

        if self.preTransform:
            uniform_patch = self.postTransform(image)[0,
                uniform_patch_x_coord : uniform_patch_x_coord + self.patch_dim, 
                uniform_patch_y_coord : uniform_patch_y_coord + self.patch_dim
            ]
            origin_patch = self.postTransform(image)
        else:
            uniform_patch = transforms.ToTensor(image)[0,
                uniform_patch_x_coord : uniform_patch_x_coord + self.patch_dim, 
                uniform_patch_y_coord : uniform_patch_y_coord + self.patch_dim
            ]
            origin_patch = transforms.ToTensor(image)

        return uniform_patch.unsqueeze(0), origin_patch, uniform_patch_x_coord + self.patch_dim//2, uniform_patch_y_coord + self.patch_dim//2

# General Code for supervised train
def patchtrain(model, fc_layer, dataloaders, criterion, optimizer, scheduler, 
    device, checkpoint_path, file, saveinterval=1, last_epochs=0, num_epochs=20):

    since = time.time()
    best_acc = 0.0

    for epoch in range(last_epochs, last_epochs+num_epochs):
        print('\nEpoch {}/{} \n'.format(epoch, last_epochs+num_epochs - 1))
        file.write('\nEpoch {}/{} \n'.format(epoch, last_epochs+num_epochs - 1))
        file.write('-' * 10)
        file.write('\n')
        file.flush()

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
                fc_layer.train()
            else:
                model.eval()  # Set model to evaluate mode
                fc_layer.eval()

            running_loss = 0.0
            running_corrects = 0
            n_samples = 0

            # Iterate over data.
            for _, (input_0, input_1, labels) in enumerate(tqdm(dataloaders[phase])):
                input_0 = input_0.to(device)
                input_1 = input_1.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                batchSize = labels.size(0)
                n_samples += batchSize

                # forward
                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):
                    fea_output_0 = model(input_0)
                    fea_output_1 = model(input_1)
                    outputs = fc_layer(torch.cat((fea_output_0, fea_output_1), dim=1))
                    
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                # statistics
                running_loss += loss.item() * labels.size(0)
                pred_top_1 = torch.topk(outputs, k=1, dim=1)[1]
                running_corrects += pred_top_1.eq(labels.view_as(pred_top_1)).int().sum().item()

            # Metrics
            top_1_acc = running_corrects / n_samples
            epoch_loss = running_loss / n_samples
            print('{} Loss: {:.6f} Top 1 Acc: {:.6f} \n'.format(phase, epoch_loss, top_1_acc))

            file.write('{} Loss: {:.6f} Top 1 Acc: {:.6f} \n'.format(phase, epoch_loss, top_1_acc))
            file.flush()

            # deep copy the model
            if phase == 'test' and top_1_acc > best_acc:
                best_acc = top_1_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_fc_wts = copy.deepcopy(fc_layer.state_dict())
        if (epoch+1) % saveinterval == 0:
            torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (checkpoint_path, epoch))
            torch.save(fc_layer.state_dict(), '%s/fc_epoch_%d.pth' % (checkpoint_path, epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s \n'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f} \n'.format(best_acc))
    file.write('Training complete in {:.0f}m {:.0f}s \n'.format(time_elapsed // 60, time_elapsed % 60))
    file.write('Best test Acc: {:4f} \n'.format(best_acc))
    file.flush()

    # load best model weights
    model.load_state_dict(best_model_wts)
    fc_layer.load_state_dict(best_fc_wts)
    return model, fc_layer

def aug_loader(patch_dim, data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers):
    image_datasets = Selfset(data_root, patch_dim, 
        preTransform = data_pre_transforms, postTransform=data_post_transforms)
    assert image_datasets
    dataloaders = torch.utils.data.DataLoader(
            image_datasets, batch_size=batch_size,
            pin_memory=True, shuffle=True, num_workers=num_workers
        )
    return dataloaders

class MyNet(nn.Module):
    def __init__(self):
        #使用super()方法调用基类的构造器，即nn.Module.__init__(self)
        super(MyNet,self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 5, padding=2)
        self.conv3 = nn.Conv2d(16, 16, 5, padding=2)
        self.conv4 = nn.Conv2d(16, 1, 5, padding=2)

    def forward(self,x):
        # print(x.shape)
        # x = F.max_pool2d(F.relu(self.conv1(x)),2)
        # x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        return x

class SpatialSoftmax(nn.Module):
    def __init__(self, height, width, device, temperature=None):
        super(SpatialSoftmax, self).__init__()
        self.height = height
        self.width = width
        self.device = device
        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
            np.linspace(0.0, self.height),
            np.linspace(0.0, self.width)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # Output:
        softmax_attention = F.softmax(feature, dim=-1)

        self.pos_x = self.pos_x.to(self.device)
        self.pos_y = self.pos_y.to(self.device)
        softmax_attention = softmax_attention.to(self.device)
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)
        return expected_x, expected_y
        # expected_xy = torch.cat([expected_x, expected_y], 1)
        # feature_keypoints = expected_xy.view(-1, self.channel * 2)

        # return feature_keypoints

def soft_argmax(voxels):
	"""
	Arguments: voxel patch in shape (batch_size, channel, H, W)
	Return: 2D coordinates in shape (batch_size, channel, 2)
	"""
	assert voxels.dim()==5
	# alpha is here to make the largest element really big, so it
	# would become very close to 1 after softmax
	alpha = 1000.0 
	N,C,H,W = voxels.shape
	soft_max = nn.functional.softmax(voxels.view(N,C,-1)*alpha,dim=2)
	soft_max = soft_max.view(voxels.shape)
	indices_kernel = torch.arange(start=0,end=H*W).unsqueeze(0)
	indices_kernel = indices_kernel.view((H,W))
	conv = soft_max*indices_kernel
	indices = conv.sum(2).sum(2).sum(2)
	y = indices%W
	x = (indices/W).floor()%H
	coords = torch.stack([x,y],dim=2)
	return coords