import torch.nn as nn
from src.models.utils import get_activation, f

# TODO why this init?
import torch.nn.init as init

# Define the custom head
class CustomHead(nn.Module):
    def __init__(self, num_features, parameters):
        super(CustomHead, self).__init__()

        # Block 1
        #self.linear1 = nn.Linear( in_features=num_features, out_features=parameters.classifier.block1.output )
        #self.activation1 = get_activation( parameters.classifier.block1.activation )

        # Block 2
        #self.bn2 = nn.BatchNorm1d( num_features=parameters.classifier.block1.output )
        #self.linear2 = nn.Linear( in_features=parameters.classifier.block1.output, out_features=parameters.classifier.block2.output )
        self.bn2 = nn.BatchNorm1d( num_features=num_features )
        self.linear2 = nn.Linear( in_features=num_features, out_features=parameters.classifier.block2.output )


    def forward(self, x):
        # Block 1
        #x = self.linear1(x)
        #x = self.activation1(x)
   
        # Block 2
        x = self.bn2(x)
        x = self.linear2(x)
        return x