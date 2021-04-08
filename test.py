#!/usr/bin/env python3
# -*- coding utf-8 -*-

from pathlib import Path
from glob import glob
import argparse
import torch
import torch.nn as nn
from models.model_utils import models, load_checkpoint, init_model
from utils.general_utils import warn_cuda_not_available, check_cuda_device_id
from utils.train_utils import create_loaders
from utils.test_utils import evaluate


def get_args():
    parser = argparse.ArgumentParser('''Train classification model''')
    parser.add_argument('--data-path', type=str, required=True,
                        help="Path to folder where train/val/test directories exist")
    parser.add_argument('--checkpoints-dir', type=str, required=True,
                        help="Path to root checkpoints folder")
    parser.add_argument('--exp', type=str, required=True,
                        help="pattern in experiment's name")
    parser.add_argument('--checkpoint-type', type=str, default="best",
                        help="Which type of checkpoint load: [best/last]")
    parser.add_argument('--device', type=int, default=0,
                        help="ID of CUDA device, to specify device")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('--num-workers', type=int, required=False, default=2,
                        help="Number of workers to parallel data loading on CPU")
    parser.add_argument('--num-classes', type=int, required=False,
                        help="""Number of classes, equal to last layer output size.
                                If not set, will be calculated base on train dataset""")
    parser.add_argument('--mean', nargs='+', default=[0.485, 0.456, 0.406], type=float,
                        help='RGB mean for image normalization')
    parser.add_argument('--std', nargs='+', default=[0.229, 0.224, 0.225], type=float,
                        help='RGB std for image normalization')
    parser.add_argument('--model-type', type=str, required=False, default=None,
                        help="Type of network model. Select from: [Base, Base_1, AlexNet, vgg16]")

    args = parser.parse_args()
    return args


def main(args):

    use_cuda = torch.cuda.is_available()
    # ask for resuming if no CUDA available
    if not use_cuda:
        warn_cuda_not_available()

    # set local input params from args
    data_path = args.data_path
    checkpoints_dir = args.checkpoints_dir
    device = args.device
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_classes = args.num_classes
    mean, std = args.mean, args.std
    model_type = args.model_type
    exp = args.exp
    checkpoint_type = args.checkpoint_type

    print(f"EVALUATE for experiment: {exp}")

    if use_cuda:
        device = check_cuda_device_id(device)
        torch.cuda.set_device(device)

    experiments = glob(checkpoints_dir + "/*")
    experiment_path = [v for v in experiments if exp in v]
    assert len(experiment_path) > 0
    experiment_path = experiment_path[0]

    checkpoints = glob(experiment_path + "/*")
    checkpoint_path = [v for v in checkpoints if checkpoint_type in v]
    assert len(checkpoint_path) > 0
    checkpoint_path = checkpoint_path[0]

    print(f"Found checkpoint: {checkpoint_path}")

    # get model type from checkpoint name
    if model_type is None:
        for m in models:
            if m in checkpoint_path:
                model_type = m
                break
        print(f"Get model type from checkpoint name: {model_type}")
    assert model_type

    # init model and then load state dict
    model = init_model(model_type, num_classes, pretrained=False)
    assert model
    model, result_dict = load_checkpoint(checkpoint_path, model)

    for k, v in result_dict.items():
        print(f"{k}: {v}")

    # TODO init optimizer and load state dict
    # TODO init sheduler and load state dict

    # load test dataset and create dataloader
    loaders = create_loaders(data_path, mean, std, batch_size, num_workers, False, None, ["test"])

    # evaluate model on test dataset
    criterion = nn.CrossEntropyLoss()
    accuracy, test_loss = evaluate(loaders["test"], model, criterion, use_cuda)

    print("\n\n")


if __name__ == "__main__":
    args = get_args()
    main(args)
