# -*- coding utf-8 -*-

from models.nn_alex_net import BasicCNN, BasicCNN_v1
import torch.nn as nn
import torchvision.models as torch_models

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
    elif model_type == "Base_1":
        model = BasicCNN_v1(n_classes)
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
