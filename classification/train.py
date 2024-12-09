from torchsummary import  summary

from src.utils import tools
from src.trainer import train_model
from src.loaddataset import load_datasets
from src.buildmodel import build_model
from src.utils import config

tools.print_now("START TRAINING")

# Load parameters
parameters = config.load_parameters()

# Load dataset
train_loader, valid_loader, train_dataset_size = load_datasets(parameters=parameters)

# Build model
model, device = build_model(parameters)
summary(model,input_size=(3,400,200))

# Train model
train_model(parameters=parameters, model=model, device=device, train_loader=train_loader, valid_loader=valid_loader, train_dataset_size=train_dataset_size)

tools.print_now("END TRAINING -- ")