#!/usr/bin/env python

from __future__ import print_function

import windseer.data as data
import windseer.nn as nn_custom
import windseer.utils as utils

import argparse
import numpy as np
import os
import torch

parser = argparse.ArgumentParser(
    description='Script to plot a prediction of the network'
    )
parser.add_argument('-d', dest='dataset', required=True, help='The test dataset')
parser.add_argument(
    '-save_name',
    dest='savename',
    default='prediction_dataset.hdf5',
    help='The name of the prediction database'
    )
parser.add_argument(
    '-m',
    dest='models',
    required=True,
    nargs='+',
    help='Models used to predict (directory, can be multiple models'
    )
parser.add_argument(
    '-v',
    dest='versions',
    required=True,
    nargs='+',
    help='Model version (requires the same number of entries as models'
    )

args = parser.parse_args()

models = []
for m, v in zip(args.models, args.versions):
    models.append({'name': m, 'version': v, 'prediction_level': 10})

# check if the dataset arguments are consistent
dataset_kwargs = None
for item in models:
    params = utils.WindseerParams(os.path.join(item['name'], 'params.yaml'))

    if dataset_kwargs == None:
        dataset_kwargs = params.Dataset_kwargs()

    else:
        if dataset_kwargs != params.Dataset_kwargs():
            print('ERROR: Dataset arguments are not consistent')
            exit()

# check if gpu available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load dataset
dataset_kwargs['return_grid_size'] = True
testset = data.HDF5Dataset(args.dataset, augmentation=False, **dataset_kwargs)

for item in models:
    net, params = utils.load_model(
        item['name'], item['version'], args.dataset, device, eval=True
        )
    item['net'] = net
    item['params'] = params

nn_custom.save_prediction_to_database(models, device, params, args.savename, testset)
