#!/usr/bin/env python3
# -*- coding utf-8 -*-

from pathlib import Path
from glob import glob
from models.model_utils import models, load_checkoint, init_model
from utils.train_utils import create_loaders
from utils.test_utils import evaluate
import torch.nn as nn


NCLASSES = 133
experiments_path = "checkpoints"
exp = "exp_17"
checkpoint_type = "best" # best/last
data_path = "data/dogImages"
mean = [0.4864, 0.4560, 0.3918]
std = [0.2602, 0.2536, 0.2562]
batch_size = 32
num_workers = 2
use_cuda = True


experiments = glob(experiments_path + "/*")
experiment_path = [v for v in experiments if exp in v]
assert len(experiment_path) > 0
experiment_path = experiment_path[0]

checkpoints = glob(experiment_path + "/*")
checkpoint_path = [v for v in checkpoints if checkpoint_type in v]
assert len(checkpoint_path) > 0
checkpoint_path = checkpoint_path[0]

print(f"Found checkpoint: {checkpoint_path}")

# get model type from checkpoint name
model_type = None
for m in models:
    if m in checkpoint_path:
        model_type = m
        break
assert model_type
print(f"Get model type from checkpoint name: {model_type}")

# TODO init model and then load state dict
model = init_model(model_type, NCLASSES, pretrained=False)
assert model
model, result_dict = load_checkoint(checkpoint_path, model)

# TODO init optimizer and load state dict
# TODO init sheduler and load state dict

# load test dataset and create dataloader
loaders = create_loaders(data_path, mean, std, batch_size, num_workers, False, None, ["test"])


# evaluate model on test dataset
criterion = nn.CrossEntropyLoss()
accuracy, test_loss = evaluate(loaders["test"], model, criterion, use_cuda)