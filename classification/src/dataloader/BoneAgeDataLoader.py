import numpy as nn
import torch
from torch.utils.data import Dataset

from src.dataloader.reader import read_dataset, one_hot_shot, gray_image_to_3channel_numpy_array

class BoneAgeDataLoader(Dataset):
    def __init__(self, data_file, parameters=None, transform=None):
        self.parameters = parameters
        self.transform = transform

        data, metadata = read_dataset(data_file=data_file)

        # get sex column (column 2 of the metadata array)
        metadata = metadata[:, 2]   

        # num_classes (2) : male and female
        num_classes = 2

        #metadata = nn.array(metadata, dtype=nn.int16)
        t_to_tensor = torch.tensor(metadata.astype(nn.float16)) #to_tensor(metadata)

        self.metadata = one_hot_shot(t_to_tensor, num_classes=num_classes)

        # Convert to 3 channel image (RGB or 3 gray identical images) if it's only 1 channel
        self.data = data if data.shape[3] == 3 else gray_image_to_3channel_numpy_array(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image = self.data[item]
        if self.transform:
            image = self.transform(image, self.parameters)
        return image, self.metadata[item]
