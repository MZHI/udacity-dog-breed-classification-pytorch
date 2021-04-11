# -*- coding utf-8 -*-

import numpy as np
from datetime import datetime
import torch
from torchvision import datasets
from torchvision.transforms import transforms
from pathlib import Path
from PIL import ImageFile
from tqdm import tqdm
from utils.general_utils import save_checkpoint

ImageFile.LOAD_TRUNCATED_IMAGES = True


def create_loaders(data_path, mean, std, batch_size, num_workers,
                   use_augm=True, color_jit_params=None, splits=None, debug=False):
    if splits is None:
        splits = ["train", "valid", "test"]

    # if color_jit_params is None:
    #     color_jit_params = [0.0, 0.0, 0.0, 0.0]

    if debug:
        print(f"splits to load: {splits}")
        print(f"Color Jitter parameters: {color_jit_params}")

    normalize = transforms.Normalize(
        mean=mean,
        std=std
    )

    torch.manual_seed(184)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    aug_transforms_list = []
    if color_jit_params is not None:
        aug_transforms_list.append(transforms.ColorJitter(brightness=color_jit_params[0],
                                                          contrast=color_jit_params[1],
                                                          saturation=color_jit_params[2],
                                                          hue=color_jit_params[3]))
    aug_transforms_list.append(transforms.RandomResizedCrop(224))
    aug_transforms_list.append(transforms.RandomHorizontalFlip())
    aug_transforms_list.append(transforms.ToTensor())
    aug_transforms_list.append(normalize)

    preprocess_augm = transforms.Compose(aug_transforms_list)

    data = {}
    for split in splits:
        transform_cur = preprocess
        if split == "train" and use_augm:
            transform_cur = preprocess_augm
            if debug:
                print(f"For [{split}] dataset use augmentation: {use_augm}")
        data[split] = datasets.ImageFolder(Path(data_path) / split, transform=transform_cur)

    if debug:
        print(f"Batch size: {batch_size}, num workers: {num_workers}")

    loaders = {}
    for split in splits:
        loaders[split] = torch.utils.data.DataLoader(data[split], batch_size=batch_size,
                                                     num_workers=num_workers, shuffle=True)

    return loaders


def train(checkpoint_path: str, n_epochs, loaders, model, optimizer, criterion, use_cuda,
          writer=None, scheduler=None, resume_train=False, result_dict=None, early_stopping=10):
    """
        :param checkpoint_path  path to checkpoint's directory. [checkpoints_dir]/[checkpoint_name]
        :param n_epochs:        number of epochs
        :param loaders:         dictionary with train and val loaders
        :param model:           network model
        :param optimizer:       optimizer
        :param criterion:       criterion
        :param use_cuda:        if use or not cuda
        :param writer:          tensorboard writer
        :param scheduler        scheduler for learning rate
        :param resume_train     flag to resume train model
        :param result_dict      state dictionary, using if resume_train=True
        :param early_stopping   early stopping
    """

    valid_loss_min = np.Inf
    total_train_time_seconds = 0.0
    epoch_begin = 1
    best_epoch = epoch_begin
    num_classes = len(loaders['train'].dataset.class_to_idx)

    if resume_train:
        # TODO
        pass

    start_training_time = datetime.now()

    for epoch in range(epoch_begin, epoch_begin + n_epochs):
        train_loss = 0.0
        valid_loss = 0.0

        lr = optimizer.param_groups[0]['lr']
        if writer is not None:
            writer.add_scalar('Learning Rate', lr, epoch)

        #######################
        ### train the model ###
        #######################
        model.train()
        with tqdm(loaders['train'], unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Train. Epoch {epoch}")
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)
                tepoch.set_postfix(loss=train_loss)

        ##########################
        ### validate the model ###
        ##########################
        model.eval()
        with tqdm(loaders['valid'], unit="batch") as vepoch:
            for data, target in vepoch:
                vepoch.set_description(f"Valid. Epoch {epoch}")
                if use_cuda:
                    data, target = data.cuda(), target.cuda()

                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item() * data.size(0)
                vepoch.set_postfix(loss=valid_loss)

        # calculate average batch train and validation loss
        train_loss /= len(loaders['train'].sampler)
        valid_loss /= len(loaders['valid'].sampler)

        if writer is not None:
            writer.add_scalar('train/TrainLoss', train_loss, epoch)
            writer.add_scalar('val/ValidLoss', valid_loss, epoch)

        if scheduler is not None:
            scheduler.step(valid_loss)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
        ))

        ##########################
        ### evaluation on test ###
        ##########################
        test_loss = 0.0
        correct = 0.0
        total = 0.0
        model.eval()
        with tqdm(loaders['test'], unit="batch") as ttepoch:
            for data, target in ttepoch:
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)
                #         test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))

                # convert output probabilities to predicted class
                pred = output.data.max(1, keepdim=True)[1]
                # compare predictions to true label
                correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred)).cpu().numpy()))
                total += data.size(0)
                ttepoch.set_postfix(loss=test_loss, accuracy=f"{correct * 100.0 / total:.2f}%")

        test_loss /= len(loaders['test'].sampler)

        if writer is not None:
            writer.add_scalar('test/TestLoss', test_loss, epoch)
            writer.add_scalar('test/TestAccuracy', 100.0 * correct / total, epoch)

        print('Test Loss: {:.6f}\n'.format(test_loss))
        print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
            100. * correct / total, correct, total))

        # save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            print(
                "Epoch {}: Upgrate of model quality: ({:.5f})->({:.5f})".format(epoch, valid_loss_min, valid_loss))

            delta_time = datetime.now() - start_training_time
            checkpoint = {
                'datetime_saved': datetime.now(),
                'epoch': epoch,
                'loss_train': train_loss,
                'loss_val': valid_loss,
                'lr': lr,
                'num_classes': num_classes,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else " ",
                'total_train_time_seconds': total_train_time_seconds +
                                            delta_time.seconds +
                                            delta_time.days * 24 * 3600,
                # 'classes_dict': classes_dict,
                'checkpoint_name': Path(checkpoint_path).name
            }
            save_checkpoint(checkpoint=checkpoint, save_path=checkpoint_path,
                            checkpoint_type='best')
            valid_loss_min = valid_loss
            best_epoch = epoch
        elif epoch - best_epoch > early_stopping:
            # ---------- check for stop training process ---------- #
            delta_time = datetime.now() - start_training_time
            print('STOPPED at {} epoch'.format(epoch))
            checkpoint = {
                'datetime_saved': datetime.now(),
                'epoch': epoch,
                'loss_train': train_loss,
                'loss_val': valid_loss,
                'lr': lr,
                'num_classes': num_classes,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'total_train_time_seconds': total_train_time_seconds +
                                            delta_time.seconds +
                                            delta_time.days * 24 * 3600,
                # 'classes_dict': classes_dict,
                'checkpoint_path': Path(checkpoint_path).name
            }
            save_checkpoint(checkpoint=checkpoint, save_path=checkpoint_path,
                            checkpoint_type='last')
            break

        # save last checkpoint:
        delta_time = datetime.now() - start_training_time
        checkpoint = {
            'datetime_saved': datetime.now(),
            'epoch': epoch,
            'loss_train': train_loss,
            'loss_val': valid_loss,
            'lr': lr,
            'num_classes': num_classes,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_train_time_seconds': total_train_time_seconds +
                                        delta_time.seconds +
                                        delta_time.days * 24 * 3600,
            # 'classes_dict': classes_dict,
            'checkpoint_path': Path(checkpoint_path).name
        }
        save_checkpoint(checkpoint=checkpoint, save_path=checkpoint_path,
                        checkpoint_type='last')

    return model, best_epoch
