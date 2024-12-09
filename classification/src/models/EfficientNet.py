import torch.nn as nn
from src.models. utils import get_base_model

class EfficientNet(nn.Module):
    def __init__( self, parameters ):
        super(EfficientNet, self).__init__()

        model = get_base_model(parameters.model_type)
        self.model = model

    def basemodel(self):
        return self.model
  
    def forward(self, x):
        return self.model(x)
