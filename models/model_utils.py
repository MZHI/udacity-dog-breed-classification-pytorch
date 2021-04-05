# -*- coding utf-8 -*-

from models.nn_alex_net import BasicCNN


models = ['Base', 'AlexNet']


def check_model_type(model_type):
    return model_type in models


def init_model(model_type, n_classes):
    assert check_model_type(model_type)
    model = None
    if model_type == "Base":
        model = BasicCNN(n_classes=n_classes)
    else:
        pass
    return model
