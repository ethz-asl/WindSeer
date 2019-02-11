#!/usr/bin/env python

from __future__ import print_function

import argparse
import nn_wind_prediction.data as data
import nn_wind_prediction.models as models
import nn_wind_prediction.nn as nn_custom
import nn_wind_prediction.utils as utils
import numpy as np
import torch
from torch.utils.data import DataLoader

# ----  Default Params --------------------------------------------------------------
compressed = False
dataset = 'data/test.tar'
index = 0 # plot the prediction for the following sample in the set, 1434
model_name = 'model_1'
model_version = 'latest'
compute_prediction_error = False
use_terrain_mask = True # should not be changed to false normally
plot_worst_prediction = False
plot_prediction = True
prediction_level = 10
num_worker = 0
# -----------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Script to plot a prediction of the network')
parser.add_argument('-c', dest='compressed', action='store_true', help='Input tar file compressed')
parser.add_argument('-ds', dest='dataset', default=dataset, help='The test dataset')
parser.add_argument('-i', dest='index', type=int, default=index, help='The index of the sample in the dataset')
parser.add_argument('-model_name', dest='model_name', default=model_name, help='The model name')
parser.add_argument('-model_version', dest='model_version', default=model_version, help='The model version')
parser.add_argument('-cpe', dest='compute_prediction_error', action='store_true', help='If set the velocity prediction errors over the full dataset is computed')
parser.add_argument('-pwp', dest='plot_worst_prediction', action='store_true', help='If set the worst prediction of the input dataset is shown. Needs compute_prediction_error to be true.')
parser.add_argument('-plot', dest='plot_prediction', action='store_true', help='If set the prediction is plotted')

args = parser.parse_args()
args.compressed = args.compressed or compressed
args.compute_prediction_error = args.compute_prediction_error or compute_prediction_error
args.plot_worst_prediction = args.plot_worst_prediction or plot_worst_prediction
args.plot_prediction = args.plot_prediction or plot_prediction

# check if gpu available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the model parameter
params = utils.EDNNParameters('trained_models/' + args.model_name + '/params.yaml')

# load dataset
testset = data.MyDataset(args.dataset, compressed = args.compressed,
                         augmentation = False, subsample = False, return_grid_size = True, **params.MyDataset_kwargs())
testloader = torch.utils.data.DataLoader(testset, batch_size=1, # needs to be one
                                             shuffle=False, num_workers=num_worker)
# load the model and its learnt parameters
NetworkType = getattr(models, params.model['model_type'])
net = NetworkType(**params.model_kwargs())

net.load_state_dict(torch.load('trained_models/' + args.model_name + '/' + args.model_version + '.model', map_location=lambda storage, loc: storage))
net.to(device)

try:
    net.set_prediction_level(prediction_level)
except AttributeError:
    pass

# define loss function
loss_fn = torch.nn.MSELoss()

# compute the errors on the dataset
if args.compute_prediction_error:
    prediction_errors, losses, worst_index, maxloss = nn_custom.dataset_prediction_error(net, device, params, loss_fn, testloader)
    np.savez('prediction_errors_' + model_name + '.npz', prediction_errors=prediction_errors, losses=losses)

    if args.plot_worst_prediction:
        args.index = worst_index

# predict the wind, compute the loss and plot if requested
input = testset[args.index][0]
label = testset[args.index][1]
scale = 1.0
if params.data['autoscale']:
    scale = testset[args.index][2].item()

nn_custom.predict_wind_and_turbulence(input, label, scale, device, net, params, args.plot_prediction, loss_fn)
