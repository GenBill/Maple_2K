import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt

from vit_pytorch import ViT

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

# print(v)
model_ft = nn.Sequential(*(list(v.children())[:-1]))
# print(model_ft)

img = torch.randn(1, 3, 256, 256)

preds = model_ft(img) # (1, 1000)

print(img.shape)
img = img[0].permute(1,2,0)
plt.imshow(img)

print(preds.shape)
preds = preds[0].permute(1,2,0)
plt.imshow(preds)