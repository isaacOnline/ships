# Script for performing the final preprocessing steps for a model, and saving the dataset to disk
# Useful if you want to perform all the preprocessing at once - which may be necessary as some of the
# datasets have large memory requirements, so shouldn't be processed in unison, which can happen
# if the datasets are calculated on the fly. The downside is that many of the datasets are huge, so
# it's difficult to store them on the same disk at once, if you don't have multiple free TB for storage

import sys
import os
import time
from config import config
import atexit
import json
import gc

os.environ['PYTHONHASHSEED'] = '0'

import numpy as np
import pandas as pd


from utils import ProcessorManager, TestArgParser
from utils.utils import total_system_ram

# These need to come before tensorflow is imported so that if we're using CPU we can unregister the GPUs before tf
# imports them.
parser = TestArgParser()
args = parser.parse()
manager = ProcessorManager(debug=True)
manager.open()

import mlflow

if args.debug:
    mlflow.set_experiment(experiment_name='Ships Debugging')


from loading.data_loader import DataLoader
import utils



if __name__ == '__main__':
    # Parse command line arguments

    start_ts = time.time()

    utils.set_seed(args.seed)

    loader = DataLoader(config, args, conserve_memory=True)
    a = loader.load_set('train', 'train', 'y')
    if type(a) == list:
        nbytes = 0
        for ds in a:
            nbytes += ds.nbytes
        print(nbytes)
        print(nbytes / total_system_ram())
    else:
        print(a.nbytes)
        print(a.nbytes / total_system_ram())
    del a
    gc.collect()

    a = loader.load_set('train', 'train', 'x')
    if type(a) == list:
        nbytes = 0
        for ds in a:
            nbytes += ds.nbytes
        print(nbytes)
        print(nbytes / total_system_ram())
    else:
        print(a.nbytes)
        print(a.nbytes / total_system_ram())
    del a

    gc.collect()

    loader.load_set('valid', 'train', 'y')
    gc.collect()
    loader.load_set('valid', 'train', 'x')
    gc.collect()

    loader.load_set('valid', 'test', 'y')
    gc.collect()
    loader.load_set('valid', 'test', 'x')
    gc.collect()

    loader.load_set('test', 'test', 'y')
    gc.collect()
    loader.load_set('test', 'test', 'x')
    gc.collect()
