import os
import torch
from torchvision.transforms import v2

from src.dataloader.BoneAgeDataLoader import BoneAgeDataLoader
from torch.utils.data import DataLoader

def load_datasets(parameters):
    # Load Data
    train_set = BoneAgeDataLoader(data_file=parameters.dataset.train_file, parameters=parameters, transform=transform_train)
    valid_set = BoneAgeDataLoader(data_file=parameters.dataset.validation_file, parameters=parameters, transform=no_transformation)

    train_loader = DataLoader(dataset=train_set, drop_last=True, num_workers=1, pin_memory=True, shuffle=True, batch_size=parameters.batch )  
    valid_loader = DataLoader(dataset=valid_set, drop_last=True, num_workers=1, pin_memory=True, shuffle=True, batch_size=parameters.batch)
    return train_loader, valid_loader, len(train_set)

def load_dataset_for_inference(data_file, batch_size):
    set = BoneAgeDataLoader(data_file=data_file, transform=no_transformation)
    loader = DataLoader(dataset=set, drop_last=False, num_workers=1, pin_memory=True, shuffle=False, batch_size=batch_size )  
    return loader

def transform_train(data, parameters):
    trans = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomAffine(degrees=parameters.augmentation.rotation_range, 
                        translate=(parameters.augmentation.width_shift_range,parameters.augmentation.height_shift_range)),
    ])
    return trans(data)

def no_transformation(data, parameters):
    transformer = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
    ])
    return transformer(data)
