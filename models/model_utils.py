# -*- coding utf-8 -*-

from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as torch_models
from models.nn_alex_net import BasicCNN, BasicCNN_v1


models = ['Base', 'Base_1', 'AlexNet', 'vgg16']
freeze_fc_dict = {
    'AlexNet': [6, 4, 1],
    'vgg16': [6, 3, 0]
}


def check_model_type(model_type):
    return model_type in models


def check_num_fc_freeze(num_freeze, model_type):
    if model_type == 'Base':
        raise Exception('For Base models there aren\'t pretrained weights')
    elif model_type == 'AlexNet':
        assert num_freeze <= 3
    elif model_type == 'vgg16':
        assert num_freeze <= 3
    else:
        raise NotImplementedError('Other model architectures not implemented yet')


def init_model(model_type, n_classes, pretrained=False, num_fc_train=1):
    assert check_model_type(model_type)
    model = None
    if model_type == "Base":
        model = BasicCNN(n_classes=n_classes)
    elif model_type == "Base_1":
        model = BasicCNN_v1(n_classes)
    elif model_type == "AlexNet":
        if pretrained:
            model = torch_models.alexnet(pretrained=True)
            # freeze all layers except num_fc_train FC layers

            for param in model.features.parameters():
                param.requires_grad = False
            model.avgpool.requires_grad = False

            for i in list(range(num_fc_train, 3)):
                model.classifier[freeze_fc_dict['AlexNet'][i]].requires_grad = False
        else:
            model = torch_models.alexnet()
        last_layer = nn.Linear(model.classifier[6].in_features, n_classes)
        model.classifier[6] = last_layer
    elif model_type == 'vgg16':
        if pretrained:
            model = torch_models.vgg16(pretrained=True)
            # freeze all layers except num_fc_train FC layers

            for param in model.features.parameters():
                param.requires_grad = False

            for i in list(range(num_fc_train, 3)):
                model.classifier[freeze_fc_dict['vgg16'][i]].requires_grad = False
        else:
            model = torch_models.vgg16()
        last_layer = nn.Linear(model.classifier[6].in_features, n_classes)
        model.classifier[6] = last_layer
    else:
        pass
    return model


def load_checkpoint(checkpoint_path, model):
    assert Path(checkpoint_path).exists()
    checkpoint_dict = torch.load(checkpoint_path)
    print(f"Successful loaded checkpoint from {checkpoint_path}")
    print(f"Available checkpoint's dictionary keys: {checkpoint_dict.keys()}")

    result_dict = dict()
    if 'datetime_saved' in checkpoint_dict.keys():
        result_dict['datetime_saved'] = checkpoint_dict['datetime_saved']
    else:
        result_dict['datetime_saved'] = None

    if 'epoch' in checkpoint_dict.keys():
        result_dict['epoch'] = checkpoint_dict['epoch']
    else:
        result_dict['epoch'] = None

    if 'loss_train' in checkpoint_dict.keys():
        result_dict['loss_train'] = checkpoint_dict['loss_train']
    else:
        result_dict['loss_train'] = None

    if 'loss_val' in checkpoint_dict.keys():
        result_dict['loss_val'] = checkpoint_dict['loss_val']
    else:
        result_dict['loss_val'] = None

    if 'lr' in checkpoint_dict.keys():
        result_dict['lr'] = checkpoint_dict['lr']
    else:
        result_dict['lr'] = None

    if 'num_classes' in checkpoint_dict.keys():
        result_dict['num_classes'] = checkpoint_dict['num_classes']
    else:
        result_dict['num_classes'] = None

    if 'model_state_dict' in checkpoint_dict.keys():
        _model_state_dict = checkpoint_dict['model_state_dict']
    else:
        _model_state_dict = None

    if 'optimizer_state_dict' in checkpoint_dict.keys():
        _optimizer_state_dict = checkpoint_dict['optimizer_state_dict']
    else:
        _optimizer_state_dict = None

    if 'sheduler_state_dict' in checkpoint_dict.keys():
        _sheduler_state_dict = checkpoint_dict['sheduler_state_dict']
    else:
        _sheduler_state_dict = None

    if 'total_train_time_seconds' in checkpoint_dict.keys():
        result_dict['total_train_time_seconds'] = checkpoint_dict['total_train_time_seconds']
    else:
        result_dict['total_train_time_seconds'] = None

    if 'checkpoint_name' in checkpoint_dict.keys():
        result_dict['checkpoint_name'] = checkpoint_dict['checkpoint_name']
    else:
        result_dict['checkpoint_name'] = None

    assert _model_state_dict is not None, 'Not found model_state_dict in checkpoint dict'
    model.load_state_dict(_model_state_dict)

    # TODO for resume training
    # optimizer.load_state_dict(_optimizer_state_dict)
    # sheduler.load_state_dict()

    return model, result_dict