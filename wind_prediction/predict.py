#!/usr/bin/env python

from __future__ import print_function

import argparse
import nn_wind_prediction.data as nn_data
import nn_wind_prediction.models as models
import nn_wind_prediction.nn as nn_custom
import nn_wind_prediction.utils as utils
import numpy as np
import torch
import os
from torch.utils.data import DataLoader

# ----  Default Params --------------------------------------------------------------
compressed = False
dataset = 'data/test.hdf5'
index = 0 # plot the prediction for the following sample in the set, 1434
model_name = 'test_model'
model_version = 'latest'
print_loss = False
compute_prediction_error = False
use_terrain_mask = True # should not be changed to false normally
plot_worst_prediction = False
plot_prediction = False
prediction_level = 10
num_worker = 0
add_all = False
# -----------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Script to plot a prediction of the network')
parser.add_argument('-c', dest='compressed', action='store_true', help='Input tar file compressed')
parser.add_argument('-ds', dest='dataset', default=dataset, help='The test dataset')
parser.add_argument('-i', dest='index', type=int, default=index, help='The index of the sample in the dataset')
parser.add_argument('-model_name', dest='model_name', default=model_name, help='The model name')
parser.add_argument('-model_version', dest='model_version', default=model_version, help='The model version')
parser.add_argument('-pl', dest='print_loss', action='store_true', help='If the loss used for training should be computed for the sample and then printed')
parser.add_argument('-cpe', dest='compute_prediction_error', action='store_true', help='If set the velocity prediction errors over the full dataset is computed')
parser.add_argument('-pwp', dest='plot_worst_prediction', action='store_true', help='If set the worst prediction of the input dataset is shown. Needs compute_prediction_error to be true.')
parser.add_argument('-plot', dest='plot_prediction', action='store_true', help='If set the prediction is plotted')
parser.add_argument('-save', dest='save_prediction', action='store_true', help='If set the prediction is saved')

args = parser.parse_args()
args.compressed = args.compressed or compressed
args.print_loss = args.print_loss or print_loss
args.compute_prediction_error = args.compute_prediction_error or compute_prediction_error
args.plot_worst_prediction = args.plot_worst_prediction or plot_worst_prediction
args.plot_prediction = args.plot_prediction or plot_prediction

# check if gpu available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the model parameter
params = utils.EDNNParameters('trained_models/' + args.model_name + '/params.yaml')

# load dataset
testset = nn_data.HDF5Dataset(args.dataset, compressed = args.compressed,
                             augmentation = False, return_grid_size = True, **params.Dataset_kwargs())
testloader = torch.utils.data.DataLoader(testset, batch_size=1, # needs to be one
                                             shuffle=False, num_workers=num_worker)

# get grid size of test dataset if potential flow is used
if params.model_kwargs()['potential_flow']:
    grid_size = nn_data.get_grid_size(args.dataset)
    params.model_kwargs()['grid_size'] = grid_size

# load the model and its learnt parameters
NetworkType = getattr(models, params.model['model_type'])
net = NetworkType(**params.model_kwargs())

# load state dict
state_dict = torch.load('trained_models/' + args.model_name + '/' + args.model_version + '.model',
                                                                            map_location=lambda storage, loc: storage)

# load params
net.load_state_dict(state_dict)
net.to(device)

try:
    net.set_prediction_level(prediction_level)
except AttributeError:
    pass

# define combined loss
loss_fn = nn_custom.CombinedLoss(**params.loss)

# if homoscedastic loss factors were learned during training, recover them
if loss_fn.learn_scaling:
    loss_state_dict = torch.load('trained_models/' + args.model_name + '/' + args.model_version + '.loss',
                                                                            map_location=lambda storage, loc: storage)
    loss_fn.load_state_dict(loss_state_dict)

# print the loss if requested
if args.print_loss:
    with torch.no_grad():
        i = 0
        for data in testloader:
            if i == args.index:
                input = data[0]
                label = data[1]
                input, label = input.to(device), label.to(device)
                output = net(input)
                print('\n------------------------------------------------------------')
                print('\tEvaluation w/ loss(es) used in training\n')
                for k in range(len(loss_fn.loss_components)):
                    component_loss = loss_fn.loss_components[k](output, label, input)
                    print(loss_fn.loss_component_names[k], ':', component_loss.item(), end='')

                    if len(loss_fn.loss_components) >1:
                        factor = loss_fn.loss_factors[k]
                        if loss_fn.learn_scaling:
                            print(', homoscedastic factor :', factor.item())
                        else:
                            print(', const factor :', factor.item())

                if len(loss_fn.loss_components) > 1:
                    print('Combined :', loss_fn(output, label, input).item())
                print('\n------------------------------------------------------------')
            i+=1

# prediction criterion
criterion = torch.nn.MSELoss()
print('\tPrediction w/ criterion: ', criterion.__class__.__name__,'\n')

# compute the errors on the dataset
if args.compute_prediction_error and all(elem in params.data['label_channels'] for elem in ['ux', 'uy', 'uz']):
    prediction_errors, losses, worst_index, maxloss = nn_custom.dataset_prediction_error(net, device, params, criterion, testloader)
    np.savez('prediction_errors_' + args.model_name + '.npz', prediction_errors=prediction_errors, losses=losses)

    if args.plot_worst_prediction:
        args.index = worst_index
elif args.compute_prediction_error and not all(elem in params.data['label_channels'] for elem in ['ux', 'uy', 'uz']):
    print('Warning: cannot compute prediction error database, not all velocity components were provided in label channels')

# predict the wind, compute the loss and plot if requested
data = testset[args.index]
input = data[0]
label = data[1]
scale = 1.0
if params.data['autoscale']:
    scale = data[2].item()

print('Test index name: {0}'.format(testset.get_name(args.index)))
if args.save_prediction:
    savename = 'data/'+os.path.splitext(testset.get_name(args.index))[0]
else:
    savename = None

if args.plot_prediction:
    channels_to_plot = 'all'
else:
    channels_to_plot = None

nn_custom.predict_channels(input, label, scale, device, net, params, channels_to_plot, args.dataset,
                           plot_divergence =False, loss_fn=criterion, savename=savename)