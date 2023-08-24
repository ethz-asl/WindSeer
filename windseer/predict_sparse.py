import argparse
import numpy as np
import random
import torch

import windseer.data as data
import windseer.nn as nn
import windseer.plotting as plotting
import windseer.utils as utils

parser = argparse.ArgumentParser(
    description='Predict the flow based on the sparse measurements'
    )
parser.add_argument('config_yaml', help='Input yaml config')
parser.add_argument(
    '-model_dir', dest='model_dir', required=True, help='The directory of the model'
    )
parser.add_argument(
    '-model_version', dest='model_version', default='latest', help='The model version'
    )
parser.add_argument(
    '--mayavi', action='store_true', help='Generate some extra plots using mayavi'
    )
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the NN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net, params = utils.load_model(args.model_dir, args.model_version, None, device, True)

if params.data['input_mode'] < 3:
    raise ValueError('Models with an input mode other than 5 are not supported')

config = utils.BasicParameters(args.config_yaml)
config = utils.update_sparse_config(config, params)

# load the data
measurement, terrain, ground_truth, mask, scale, _, _ = data.load_measurements(
    config.params['measurements'], config.params['model']
    )

config.params['model']['input_channels'] += ['mask']

input = data.compose_model_input(
    measurement, mask, terrain, config.params['model'], device
    )

with torch.no_grad():
    prediction, _, _ = nn.get_prediction(
        input,
        ground_truth,
        scale,
        device,
        net,
        config.params['model']['config'],
        scale_input=True
        )

if args.mayavi:
    ui = []
    plotting.mlab_plot_measurements(
        measurement[:, :3],
        mask,
        terrain,
        terrain_mode='blocks',
        terrain_uniform_color=True,
        blocking=False
        )

    ui.append(
        plotting.mlab_plot_prediction(
            prediction['pred'],
            terrain,
            terrain_mode='blocks',
            terrain_uniform_color=True,
            prediction_channels=config.params['model']['label_channels'],
            blocking=False
            )
        )

if not ground_truth is None:
    ground_truth = ground_truth.cpu().squeeze()

plotting.plot_prediction(
    params.data['label_channels'],
    prediction=prediction['pred'][0].cpu().detach(),
    provided_input_channels=params.data['input_channels'],
    input=input[0].cpu().detach(),
    terrain=terrain.cpu().squeeze(),
    label=ground_truth,
    measurements=measurement[0].cpu().detach(),
    measurements_mask=mask.squeeze().cpu()
    )
