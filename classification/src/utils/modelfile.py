import os
import glob
import torch
import numpy as np

def init_metrics(parameters):
    if not os.path.exists(parameters.dataset.checkpoint_dir):
        os.makedirs(parameters.dataset.checkpoint_dir)
    start_epoch = 0
    min_loss = np.inf
    metrics_train, metrics_valid = {}, {}
    metrics_train['loss'] = []
    metrics_valid['loss'] = []
    metrics_train['accuracy'] = []
    metrics_valid['accuracy'] = []
    return start_epoch, min_loss, metrics_train, metrics_valid


def load_model(parameters, model, model_file='last_model.pth'):
    model_file = os.path.join(parameters.dataset.checkpoint_dir,model_file)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    print("Loading model ... Best epoch ({}), Min loss ({}) ".format(checkpoint['epoch'], checkpoint['min_loss']) )
    model.load_state_dict(checkpoint['state_dict'])
    return model


def save_model(parameters, epoch, min_loss, metrics_train, metrics_valid, model):
    is_best = metrics_valid['loss'][-1] < min_loss
    if is_best:      
        print('       #### NEW BEST MODEL ####')
        min_loss = metrics_valid['loss'][-1]

        checkpoint = {  'epoch': epoch,   'state_dict': model.state_dict(),
                      'min_loss': min_loss, 'metrics_train': metrics_train, 'metrics_valid': metrics_valid
                      }
        name = parameters.config + '_' + parameters.run + '.pth'
        file_name = os.path.join(parameters.dataset.checkpoint_dir, name)
        torch.save( checkpoint, file_name)
    return min_loss
