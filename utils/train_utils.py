import numpy as np
from datetime import datetime
import torch
from torchvision import datasets
from torchvision.transforms import transforms
from pathlib import Path


def create_loaders(data_path, mean, std, batch_size, num_workers):
    # TODO add augmentations
    normalize = transforms.Normalize(
        mean=mean,
        std=std
    )
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    train_data = datasets.ImageFolder(Path(data_path) / "train", trainsform=preprocess)
    valid_data = datasets.ImageFolder(Path(data_path) / "valid", trainsform=preprocess)
    test_data = datasets.ImageFolder(Path(data_path) / "test", trainsform=preprocess)

    loaders = {'train': torch.utils.DataLoader(train_data, batch_size=batch_size,
                                               num_workers=num_workers, shuffle=True),
               'valid': torch.utils.DataLoader(valid_data, batch_size=batch_size,
                                               num_workers=num_workers, shuffle=True),
               'test': torch.utils.DataLoader(test_data, batch_size=batch_size,
                                              num_workers=num_workers, shuffle=True)}
    return loaders


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda,
          resume_train=False, result_dict=None):
    """
        :param n_epochs:   number of epochs
        :param loaders:    dictionary with train and val loaders
        :param model:      network model
        :param optimizer:  optimizer
        :param criterion:  criterion
        :param use_cuda:   if use or not cuda
    """
    # TODO

    return model
