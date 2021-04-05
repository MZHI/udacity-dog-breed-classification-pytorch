import numpy as np
from datetime import datetime
import torch
from torchvision import datasets
from torchvision.transforms import transforms
from pathlib import Path
from PIL import ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
          writer=None, resume_train=False, result_dict=None):
    """
        :param n_epochs:    number of epochs
        :param loaders:     dictionary with train and val loaders
        :param model:       network model
        :param optimizer:   optimizer
        :param criterion:   criterion
        :param use_cuda:    if use or not cuda
        :param writer:      tensorboard writer
        :param resume_train flag to resume train model
        :param result_dict  state dictionary, using if resume_train=True
    """

    valid_loss_min = np.Inf
    total_train_time_seconds = 0.0
    epoch_begin = 1
    best_epoch = epoch_begin

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
                loss.bachward()
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

        ## calculate average batch train and validation loss
        train_loss /= len(loaders['train'].sampler)
        valid_loss /= len(loaders['valid'].sampler)

        if writer is not None:
            writer.add_scalar('{}/TrainLoss'.format("train"), train_loss, epoch)
            writer.add_scalar('{}/ValidLoss'.format("val"), valid_loss, epoch)

        # print training/validation statistics
        # print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        #     epoch,
        #     train_loss,
        #     valid_loss
        # ))

        # save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            print(
                "Epoch {}: Upgrate of model quality: ({:.5f})->({:.5f})".format(epoch, valid_loss_min, valid_loss))

            delta_time = datetime.now() - start_training_time
            # checkpoint = {
            #     'datetime_saved': datetime.now(),
            #     'epoch': epoch,
            #     'loss_train': train_loss,
            #     'loss_val': valid_loss,
            #     'lr': lr,
            #     'num_classes': model.fc3.out_features,
            #     'model_state_dict': model_scratch.state_dict(),
            #     'optimizer_state_dict': optimizer_scratch.state_dict(),
            #     'end_training_time': delta_time,
            #     'total_train_time_seconds': total_train_time_seconds +
            #                                 delta_time.seconds +
            #                                 delta_time.days * 24 * 3600,
            #     'classes_dict': classes_dict,
            #     'tensorboard_dir': tensorboard_dir
            # }
            # save_checkpoint(checkpoint=checkpoint, save_path=checkpoints_path,
            #                 checkpoint_type='best')
            #             torch.save(model_scratch.state_dict(), 'model_scratch.pt')
            valid_loss_min = valid_loss
            best_epoch = epoch

    return model
