import torch

from src.utils import torch_setup
from src.models.EfficientNet import EfficientNet
from src.models.CustomHead import CustomHead

def build_model(parameters):
    device = torch_setup.setup(parameters)
    model = EfficientNet( parameters=parameters )
    basemodel = model.basemodel()
    basemodel.classifier = CustomHead( num_features(basemodel), parameters )

    model.to(device) # move the model to GPU
    return model, device


# Get final CNN size from features. Should match with new added classifier
def num_features(model):
    return model.features( torch.randn(1, 3, 400, 300) ).size(1)