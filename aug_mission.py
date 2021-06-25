from tqdm import tqdm
import numpy as np
import math
import random

import torch
import torch.nn.parallel
from torchvision import datasets, models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import time
import copy
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class Selfset(Dataset):

    def __init__(self, split, root_paths, patch_dim, jitter, preTransform=None, postTransform=None):
        self.root_paths = root_paths
        self.image_paths = root_paths + '/' + split

        self.patch_dim = patch_dim
        self.jitter = jitter
        self.margin = math.ceil(self.patch_dim/2.0) + self.jitter

        self.preTransform = preTransform
        self.postTransform = postTransform
        self.dataset = datasets.ImageFolder(self.image_paths, self.preTransform)

    def __len__(self):
        return len(self.dataset)
    
    def prep_patch(self, image):

        # for some patches, randomly downsample to as little as 100 total pixels
        # 说是要下采样，结果放大后又缩小？迷惑行为
        # 可能可以添加抖动
        if(random.random() < .33):
            pil_patch = Image.fromarray(image)
            original_size = pil_patch.size
            randpix = int(math.sqrt(random.random() * (95 * 95 - 10 * 10) + 10 * 10))
            pil_patch = pil_patch.resize((randpix, randpix)) 
            pil_patch = pil_patch.resize(original_size) 
            np.copyto(image, np.array(pil_patch))

    def __getitem__(self, index):
        img_PIL, _ = self.dataset[index]
        image = np.array(img_PIL)
        # If image is too small, try another image
        if image.shape[1] <= self.min_width or image.shape[0] <= self.min_width:
            return self.__getitem__(index)

        uniform_patch_x_coord = int(math.floor((image.shape[0] - self.margin*2) * random.random())) + self.margin - int(round(self.patch_dim/2.0))
        uniform_patch_y_coord = int(math.floor((image.shape[1] - self.margin*2) * random.random())) + self.margin - int(round(self.patch_dim/2.0))
        
        uniform_patch = image[
            uniform_patch_x_coord : uniform_patch_x_coord + self.patch_dim, 
            uniform_patch_y_coord : uniform_patch_y_coord + self.patch_dim
        ]                
        # 必要模块：随机图片抖动
        self.prep_patch(uniform_patch)

        if self.preTransform:
            uniform_patch = self.postTransform(uniform_patch)
            origin_patch = self.postTransform(image)
        else:
            uniform_patch = transforms.ToTensor(uniform_patch)
            origin_patch = transforms.ToTensor(image)

        return uniform_patch, origin_patch, uniform_patch_x_coord, uniform_patch_y_coord

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

def aug_loader(patch_dim, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers):

    image_datasets = {
        x: Selfset(x, data_root, patch_dim, jitter, 
        preTransform = data_pre_transforms[x], postTransform=data_post_transforms[x])
        for x in ['train', 'test']
    }
    assert image_datasets
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size,
            pin_memory=True, shuffle=True, num_workers=num_workers
        ) for x in ['train', 'test']}
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    return dataloaders

