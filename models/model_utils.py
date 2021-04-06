# -*- coding utf-8 -*-

from models.nn_alex_net import BasicCNN
import torch.nn as nn
import torchvision.models as torch_models

models = ['Base', 'AlexNet']


def check_model_type(model_type):
    return model_type in models


def init_model(model_type, n_classes, pretrained=False):
    assert check_model_type(model_type)
    model = None
    if model_type == "Base":
        model = BasicCNN(n_classes=n_classes)
    elif model_type == "AlexNet":
        if pretrained:
            model = torch_models.alexnet(pretrained=True)
            # freeze all alyers except last
            for param in model.features.parameters():
                param.requires_grad = False
            model.avgpool.requires_grad = False
            model.classifier[0].requires_grad = False
            model.classifier[1].requires_grad = False
            model.classifier[3].requires_grad = False
            model.classifier[4].requires_grad = False
        else:
            model = torch_models.alexnet()
        last_layer = nn.Linear(model.classifier[6].in_features, n_classes)
        model.classifier[6] = last_layer
    else:
        pass
    return model
