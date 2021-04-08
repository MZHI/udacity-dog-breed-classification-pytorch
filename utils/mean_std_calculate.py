#!/usr/bin/env python3
# -*- coding utf-8 -*-

from PIL import ImageFile
import torch
import torchvision.transforms as transforms
from torchvision import datasets
ImageFile.LOAD_TRUNCATED_IMAGES = True

preprocess = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(224),
   transforms.ToTensor()
])

dataset_train = datasets.ImageFolder("../data/dogImages/train/", transform=preprocess)

loader = torch.utils.data.DataLoader(dataset_train,
                         batch_size=10,
                         num_workers=0,
                         shuffle=False)

print("Dataset: {}".format(loader.dataset))
print("length of sampler: {}".format(len(loader.sampler)))

mean = 0.0
for images, _ in loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
mean = mean / len(loader.dataset)
print(f"Calculated mean: {mean}")


var = 0.0
for images, _ in loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    var += ((images - mean.unsqueeze(1))**2).sum([0,2])
std = torch.sqrt(var / (len(loader.dataset)*224*224))
print(f"Calculated std: {std}")