import os
import yaml
from munch import munchify
import argparse

def read_args():
    # PARAMETERS SAMPLE
    # python train.py --config sex_class --batch 16 --epochs 10
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Config file path: sex_class', type=str)
    parser.add_argument('--batch', default=-1, help='Batch size. If set it will overwrite config value', type=int)
    parser.add_argument('--epochs', default=-1, help='Epochs. If set it will overwrite config value', type=int)
    parser.add_argument('--run', default='run1', help='Run# to keep track of multiple runs (Example: run1, run2 ... runN)', type=str)
    args = parser.parse_args()
  
    config = args.config
    batch = args.batch
    epochs = args.epochs
    run = args.run
    return config, batch, epochs, run


def load_parameters(config=None, batch=-1, epochs=-1, run='run1'):
    if(config == None):
        config, batch, epochs, run = read_args()
    
    with open(os.path.join('configs', config + '.yaml')) as f:
        parameters = munchify(yaml.load(f, Loader=yaml.SafeLoader))
        parameters.config = config
        parameters.epochs = epochs if epochs > 0 else parameters.epochs
        parameters.batch = batch if batch > 0 else parameters.batch
        parameters.run = run
    return parameters
