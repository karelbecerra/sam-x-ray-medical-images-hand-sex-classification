import torch
import torch.nn as nn
import torchvision

def get_base_model(type):
    model = None
    match type:
        case 'efficientnet_b0':
          model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
        case 'efficientnet_b1':
          model = torchvision.models.efficientnet_b1(weights=torchvision.models.EfficientNet_B1_Weights.DEFAULT)
        case 'efficientnet_b2':
          model = torchvision.models.efficientnet_b2(weights=torchvision.models.EfficientNet_B2_Weights.DEFAULT)
        case _: # Default model
          model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
    return model


def get_activation(type):
    activation = None
    match type:
        case 'relu':
          activation = nn.ReLU()
        case 'elu':
          activation = nn.ELU()
        case 'softmax':
          activation = nn.Softmax(dim=1)
        case _: # Default model
          activation = f
    return activation

def get_avgpool(type, avgpool):
    match type:
        case 'AdaptiveMaxPool2d':
          avgpool = nn.AdaptiveMaxPool2d((1, 1))
        case _: # Default model
          avgpool = avgpool
    return avgpool

def f(x):
    return x

class F(torch.nn.Module):
    def __init__(self):
        super(F, self).__init__()

    def forward(self, x):
        return x