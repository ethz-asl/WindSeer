#!/usr/bin/env python

from __future__ import print_function

import argparse
import nn_wind_prediction.data as data
import nn_wind_prediction.models as nn_models
import nn_wind_prediction.nn as nn_custom
import nn_wind_prediction.utils as utils
import numpy as np
import torch
from torch.utils.data import DataLoader

# ----  Default Params --------------------------------------------------------------
compressed = False
dataset = 'data/test.tar'
savename = 'prediction.hdf5'
# -----------------------------------------------------------------------------------

# define here the models
models = []
models.append({'name': 'model_1',
               'version': 'latest',
               'prediction_level': 10})
models.append({'name': 'model_2',
               'version': 'latest',
               'prediction_level': 10})
models.append({'name': 'model_3',
               'version': 'latest',
               'prediction_level': 10})

parser = argparse.ArgumentParser(description='Script to plot a prediction of the network')
parser.add_argument('-c', dest='compressed', action='store_true', help='Input tar file compressed')
parser.add_argument('-d', dest='dataset', default=dataset, help='The test dataset')
parser.add_argument('-save_name', dest='savename', default=savename, help='The name of the prediction database')

args = parser.parse_args()
args.compressed = args.compressed or compressed

# check if the dataset arguments are consistent
dataset_kwargs = None
for item in models:
    params = utils.EDNNParameters('trained_models/' + item['name'] + '/params.yaml')

    if dataset_kwargs == None:
        dataset_kwargs = params.MyDataset_kwargs()

    else:
        if dataset_kwargs != params.MyDataset_kwargs():
            print('ERROR: Dataset arguments are not consistent')
            exit()

# check if gpu available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load dataset
testset = data.MyDataset(args.dataset, compressed = args.compressed,
                         augmentation = False, **dataset_kwargs)

for item in models:
    params = utils.EDNNParameters('trained_models/' + item['name'] + '/params.yaml')
    NetType = getattr(nn_models, params.model['model_type'])
    net = NetType(**params.model_kwargs())
    net.load_state_dict(torch.load('trained_models/' + item['name'] + '/' + item['version'] + '.model', map_location=lambda storage, loc: storage))
    net.to(device)
    try:
        net.set_prediction_level(item['prediction_level'])
    except AttributeError:
        pass
    item['net'] = net
    item['params'] = params

nn_custom.save_prediction_to_database(models, device, params, savename, testset)
