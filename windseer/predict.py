#!/usr/bin/env python

from __future__ import print_function

import windseer.data as nn_data
import windseer.nn as nn
import windseer.utils as utils

import argparse
import numpy as np
import os
import torch

parser = argparse.ArgumentParser(
    description='Script to plot a prediction of the network'
    )
parser.add_argument(
    '-ds', dest='dataset', default='/tmp/test.hdf5', help='Dataset filename'
    )
parser.add_argument(
    '-i',
    dest='index',
    type=int,
    default=0,
    help='The index of the sample in the dataset'
    )
parser.add_argument(
    '-model', dest='model_dir', default='test_model', help='Model directory path'
    )
parser.add_argument(
    '-model_version', dest='model_version', default='latest', help='The model version'
    )
parser.add_argument(
    '-pl',
    dest='print_loss',
    action='store_true',
    help=
    'If the loss used for training should be computed for the sample and then printed'
    )
parser.add_argument(
    '-cpe',
    dest='compute_prediction_error',
    action='store_true',
    help='If set the velocity prediction errors over the full dataset is computed'
    )
parser.add_argument(
    '-cse',
    dest='compute_single_error',
    action='store_true',
    help='If set the velocity prediction errors over a single sample is computed'
    )
parser.add_argument(
    '-n',
    dest='n_iter',
    type=int,
    default=100,
    help='The number of forward passes for the single sample evaluation'
    )
parser.add_argument(
    '-save',
    dest='save_prediction',
    action='store_true',
    help='If set the prediction is saved'
    )
parser.add_argument(
    '-s',
    dest='seed',
    type=int,
    default=0,
    help='If larger than 0 this sets the seed of the random number generator'
    )
parser.add_argument(
    '--plottools',
    dest='plottools',
    action='store_true',
    help='If set the prediction is plotted'
    )
parser.add_argument(
    '--mayavi', action='store_true', help='Generate some extra plots using mayavi'
    )
parser.add_argument(
    '--azimuth', type=float, help='Set the azimuth angle of the mayavi view'
    )
parser.add_argument(
    '--elevation', type=float, help='Set the elevation angle of the mayavi view'
    )
parser.add_argument(
    '--distance', type=float, help='Set the distance of the mayavi view'
    )
parser.add_argument(
    '--focalpoint', type=float, nargs=3, help='Set the focalpoint of the mayavi view'
    )
parser.add_argument(
    '--animate_mayavi',
    type=int,
    default=-1,
    help=
    'Animate a mayavi figure (0: prediction plot, 1: error plot, 2: measurement plot, 3 uncertainty plot'
    )
parser.add_argument(
    '--save_animation', action='store_true', help='Save snapshots of the animation'
    )
args = parser.parse_args()

if args.seed > 0:
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

if args.save_prediction:
    savename = os.path.splitext(testset.get_name(args.index))[0]
else:
    savename = None

mayavi_configs = {'view_settings': {}}
if not args.azimuth is None:
    mayavi_configs['view_settings']['azimuth'] = args.azimuth
if not args.elevation is None:
    mayavi_configs['view_settings']['elevation'] = args.elevation
if not args.distance is None:
    mayavi_configs['view_settings']['distance'] = args.distance
if not args.focalpoint is None:
    mayavi_configs['view_settings']['focalpoint'] = args.focalpoint
if len(mayavi_configs['view_settings']) == 0:
    mayavi_configs['view_settings'] = None

mayavi_configs['animate'] = args.animate_mayavi
mayavi_configs['save_animation'] = args.save_animation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net, params = utils.load_model(
    args.model_dir, 'latest', args.dataset, device, eval=True
    )

testset = nn_data.HDF5Dataset(
    args.dataset, augmentation=False, return_grid_size=True, **params.Dataset_kwargs()
    )

loss_fn = nn.CombinedLoss(**params.loss)
if loss_fn.learn_scaling:
    loss_state_dict = torch.load(
        os.path.join(args.model_dir, args.model_version + '.loss'),
        map_location=lambda storage, loc: storage
        )
    loss_fn.load_state_dict(loss_state_dict)

if args.print_loss:
    with torch.no_grad():
        data = testset[args.index]
        input = data[0]
        label = data[1]
        scale = 1.0
        if params.data['autoscale']:
            scale = data[3].item()

        prediction, inputs, labels = nn.get_prediction(
            input, label, scale, device, net, params, True
            )

        print('\n------------------------------------------------------------')
        print('\tEvaluation w/ loss(es) used in training\n')
        for k in range(len(loss_fn.loss_components)):
            component_loss = loss_fn.loss_components[k](prediction, labels, inputs)
            print(loss_fn.loss_component_names[k], ':', component_loss.item(), end='')

            if len(loss_fn.loss_components) > 1:
                factor = loss_fn.loss_factors[k]
                if loss_fn.learn_scaling:
                    print(', homoscedastic factor :', factor.item())
                else:
                    print(', const factor :', factor.item())

        if len(loss_fn.loss_components) > 1:
            print('Combined :', loss_fn(prediction, labels, inputs).item())
        print('\n------------------------------------------------------------')

model_name = os.path.basename(os.path.normpath(args.model_dir))
if args.compute_single_error:
    prediction_errors, losses, metrics, worst_index, maxloss = nn.compute_prediction_error(
        net,
        device,
        params,
        loss_fn,
        testset,
        single_sample=True,
        num_predictions=args.n_iter,
        print_output=True
        )

    np.savez(
        'prediction_errors_' + model_name + '_sample_' + str(args.index) + '.npz',
        prediction_errors=prediction_errors,
        losses=losses
        )

if args.compute_prediction_error:
    prediction_errors, losses, metrics, worst_index, maxloss = nn.compute_prediction_error(
        net, device, params, loss_fn, testset, single_sample=False, print_output=True
        )

    np.savez(
        'prediction_errors_' + model_name + '.npz',
        prediction_errors=prediction_errors,
        losses=losses
        )

nn.predict_and_visualize(
    testset,
    args.index,
    device,
    net,
    params,
    'all',
    plot_divergence=False,
    loss_fn=torch.nn.MSELoss(),
    savename=savename,
    plottools=args.plottools,
    mayavi=args.mayavi,
    blocking=True,
    mayavi_configs=mayavi_configs
    )
