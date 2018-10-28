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
model_name = 'test_model_naKd4sF8mK'
model_version = 'latest'
compute_prediction_error = False
use_terrain_mask = True # should not be changed to false normally
plot_worst_prediction = False
plot_prediction = True
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
testset = data.MyDataset(device, args.dataset, compressed = args.compressed,
                         augmentation = False, subsample = False, **params.MyDataset_kwargs())
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=0)
# load the model and its learnt parameters
if params.model['d3']:
    if params.model['predict_uncertainty']:
        net = models.ModelEDNN3D_Twin(**params.model3d_kwargs())
    else:
        net = models.ModelEDNN3D(**params.model3d_kwargs())
else:
    net = models.ModelEDNN2D(params.model['n_input_layers'], params.model['interpolation_mode'], params.model['align_corners'], params.model['skipping'], params.data['use_turbulence'])

net.load_state_dict(torch.load('trained_models/' + args.model_name + '/' + args.model_version + '.model', map_location=lambda storage, loc: storage))
net.to(device)

# define loss function
loss_fn = torch.nn.MSELoss()

# compute the errors on the dataset
if args.compute_prediction_error:
    velocity_errors, worst_index, maxloss = nn_custom.dataset_prediction_error(net, device, params, loss_fn, testloader)
    np.save('velocity_prediction_errors.npy', velocity_errors)

    if args.plot_worst_prediction:
        args.index = worst_index

# predict the wind, compute the loss and plot if requested
input, label = testset[args.index]
nn_custom.predict_wind_and_turbulence(input, label, device, net, params, args.plot_prediction, loss_fn)
