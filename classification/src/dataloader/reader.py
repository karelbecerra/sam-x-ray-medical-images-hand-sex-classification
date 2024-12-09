import numpy as np
import torch

def read_dataset(data_file):
    data, metadata = read_file_data(data_file)
    if data.ndim == 3:
        data = np.expand_dims(data, axis=3)
    #print('Data checking...', data.shape, metadata.shape)    
    return data, metadata


def read_file_data(data_file):
    data = np.load(data_file)
    imag_data = data['data']
    meta_data = data['md']
    return imag_data, meta_data


def filter_by_gender(x, md, gender):
    idx_to_filter = md[:, 2] == gender
    x = x[idx_to_filter]
    md = md[idx_to_filter]    
    return x, md


def gray_image_to_3channel_numpy_array(x):
    x = np.squeeze(x)
    x = np.stack((x,)*3, axis=-1)
    return x


def norm_by_max(data, max=None):
    if max is None:
        max = np.amax(data)    
    norm_data = data/(1.0*max)
    return norm_data, max


def one_hot_shot(metadata, num_classes=2):
    metadata = torch.nn.functional.one_hot(metadata.to(torch.int64), num_classes=num_classes)    
    return metadata.float()