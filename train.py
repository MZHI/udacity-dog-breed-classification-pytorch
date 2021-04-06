#!/usr/bin/env python3
# -*- coding utf-8 -*-

import argparse
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
import torch.nn as nn
from utils.general_utils import check_checkpoint_exist, create_checkpoint_name
from utils.general_utils import ask_for_delete, warn_cuda_not_available, check_cuda_device_id
from utils.general_utils import init_optimizer
from utils.train_utils import create_loaders
from models.model_utils import init_model
from utils.train_utils import train


def get_args():
    parser = argparse.ArgumentParser('''Train classification model''')
    parser.add_argument('--data-path', type=str, required=True,
                        help="Path to folder where train/val/test directories exist")
    parser.add_argument('--checkpoints-dir', type=str, required=True,
                        help="Path to root checkpoints folder")
    parser.add_argument('--device', type=int, default=0,
                        help="ID of CUDA device, to specify device")
    parser.add_argument('--log-path', type=str, default="",
                        help="Path to tensorboard directory")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('--num-epochs', type=int, required=True,
                        help="Number of train epochs before stopping")
    parser.add_argument('--early-stopping', type=int, default=10,
                        help="Number of epochs before stopping if valid loss not descreases")
    parser.add_argument('--num-workers', type=int, required=False, default=2,
                        help="Number of workers to parallel data loading on CPU")
    parser.add_argument('--num-classes', type=int, required=False,
                        help="""Number of classes, equal to last layer output size.
                                If not set, will be calculated base on train dataset""")
    parser.add_argument('--mean', nargs='+', default=[0.485, 0.456, 0.406], type=float,
                        help='RGB mean for image normalization')
    parser.add_argument('--std', nargs='+', default=[0.229, 0.224, 0.225], type=float,
                        help='RGB std for image normalization')
    parser.add_argument('--lr', type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument('--optim', type=str, required=False, default='SGD',
                        help="Type of optimizer. Select from: [SGD, Adam]")
    parser.add_argument('--momentum', type=float, required=False, default=0.9,
                        help="Momentum. Using only for SGD optimizer. Default 0.9")
    parser.add_argument('--dropout', type=float, default=0.5,
                        help="Value for dropout. Default 0.5")
    parser.add_argument('--model-type', type=str, required=True,
                        help="Type of network model. Select from: [Base, AlexNet]")
    parser.add_argument('--prefix', type=str, required=False,
                        help="Prefix for checkpoint and logs naming")
    parser.add_argument("--scheduler-patience", type=int, default=None,  # recommendation: 3 or 5
                        help="Number of epochs with loss no improvement to reduce optimizer step."
                             "If not used (None as default) then scheduler will not be used")
    parser.add_argument('--scheduler-factor', type=float, default=0.5,  # recommendation: 0.5 or 0.1
                        help='Scheduler factor to reduce the learning rate')
    parser.add_argument('--scheduler-cooldown', type=int, default=2,  # recommendation: 0 or 1, or 2, ...
                        help='Scheduler cooldown after reducing the LR')
    parser.add_argument('--save-last', type=int, default=1,
                        help="Whether to save last checkpoint besides the best")
    parser.add_argument('--resume-train', type=int, default=False,
                        help="Is resume specific experiment or train from scratch")

    args = parser.parse_args()
    return args


def main(args):

    use_cuda = torch.cuda.is_available()
    # TODO ask for resuming if no CUDA available
    if not use_cuda:
        warn_cuda_not_available()

    # set local input params from args
    data_path = args.data_path
    checkpoints_dir = args.checkpoints_dir
    device = args.device
    log_path = args.log_path
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    early_stopping = args.early_stopping
    num_workers = args.num_workers
    num_classes = args.num_classes
    mean, std = args.mean, args.std
    lr, momentum = args.lr, args.momentum
    dropout = args.dropout
    optimizer_name = args.optim
    model_type = args.model_type
    prefix = args.prefix
    scheduler_patience = args.scheduler_patience
    scheduler_factor = args.scheduler_factor
    scheduler_cooldown = args.scheduler_cooldown
    save_last = args.save_last
    resume_train = args.resume_train

    checkpoint_name = create_checkpoint_name(prefix,
                                             model_type,
                                             optimizer_name,
                                             batch_size,
                                             lr, dropout)
    checkpoint_dir = Path(checkpoints_dir) / checkpoint_name
    tensorboard_dir = Path(log_path) / checkpoint_name

    if use_cuda:
        device = check_cuda_device_id(device)
        torch.cuda.set_device(device)

    if resume_train:
        # check existing of checkpoint if resume_train==1
        raise NotImplemented("This functionality not implemented yet")
        if not check_checkpoint_exist(str(checkpoint_dir)):
            raise Exception("For --resume-train==1 checkpoint {} must exist".format(checkpoint_name))
    else:
        # check if directory for this checkpoint_name doesn't exist.
        # and ask for delete if exists
        if check_checkpoint_exist(str(checkpoint_dir)):
            ask_for_delete(str(checkpoint_dir))
        if check_checkpoint_exist(str(tensorboard_dir)):
            ask_for_delete(str(tensorboard_dir))

    # create directories for checkpoint and tensorboard logs
    checkpoint_dir.parent.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    tensorboard_dir.parent.mkdir(exist_ok=True)
    tensorboard_dir.mkdir(exist_ok=True)

    # create tensorboard writer
    writer = SummaryWriter(tensorboard_dir)

    # load datasets and set dataloaders
    loaders = create_loaders(data_path, mean, std, batch_size, num_workers)

    if num_classes is None:
        num_classes = len(loaders['train'].dataset.class_to_idx)

    model = None
    optimizer = None
    if resume_train:
        # TODO load model and optimizer
        pass
    else:
        # create model and optimizer
        model = init_model(model_type, num_classes)
        if use_cuda:
            model.cuda()
        optimizer = init_optimizer(optimizer_name, model, lr, momentum)

    assert model
    assert optimizer

    scheduler = None
    if scheduler_factor is not None and scheduler_patience is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=scheduler_patience, verbose=True, factor=scheduler_factor, cooldown=scheduler_cooldown
        )

    criterion = nn.CrossEntropyLoss()

    # training process
    train(str(checkpoint_dir),
          num_epochs,
          loaders,
          model,
          optimizer,
          criterion,
          use_cuda,
          writer,
          scheduler,
          resume_train,
          result_dict=None,
          early_stopping=early_stopping)

    print("Train process finished")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    args = get_args()
    main(args)
