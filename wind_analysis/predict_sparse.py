import argparse
import numpy as np
import random
import torch

import nn_wind_prediction.models as models
import nn_wind_prediction.utils as nn_utils
from analysis_utils import utils

parser = argparse.ArgumentParser(description='Predict the flow based on the sparse measurements')
parser.add_argument('config_yaml', help='Input yaml config')
parser.add_argument('-model_dir', dest='model_dir', required=True, help='The directory of the model')
parser.add_argument('-model_version', dest='model_version', default='latest', help='The model version')
parser.add_argument('--mayavi', action='store_true', help='Generate some extra plots using mayavi')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = nn_utils.BasicParameters(args.config_yaml)

# load the NN
nn_params = nn_utils.EDNNParameters(args.model_dir + '/params.yaml')
NetworkType = getattr(models, nn_params.model['model_type'])
net = NetworkType(**nn_params.model_kwargs())
state_dict = torch.load(args.model_dir + '/' + args.model_version + '.model',
                        map_location=lambda storage, loc: storage)
net.load_state_dict(state_dict)
net.to(device)
net.eval()

if nn_params.data['input_mode'] < 3:
    raise ValueError('Models with an input mode other than 5 are not supported')

config.params['model'] = {}
config.params['model']['input_channels'] = nn_params.data['input_channels']
config.params['model']['label_channels'] = nn_params.data['label_channels']
config.params['model']['autoscale'] = nn_params.data['autoscale']
for key in nn_params.data.keys():
    if 'scaling' in key:
        config.params['model'][key] = nn_params.data[key]

# load the data
measurement, terrain, ground_truth, mask, scale, wind_data = utils.load_measurements(config.params['measurements'], config.params['model'])

config.params['model']['input_channels'] += ['mask']

measurement = measurement.to(device)
terrain = terrain.to(device)
if ground_truth is not None:
    ground_truth = ground_truth[0].cpu()
mask = mask.to(device)

input_idx = []
if 'ux'  in config.params['model']['input_channels']:
    input_idx.append(0)
if 'uy'  in config.params['model']['input_channels']:
    input_idx.append(1)
if 'uz'  in config.params['model']['input_channels']:
    input_idx.append(2)

input = torch.cat([terrain, measurement[:, input_idx], mask.unsqueeze(0)], dim = 1)

with torch.no_grad():
    prediction = utils.predict(net, input, scale, config.params['model'])

if args.mayavi:
    ui = []
    nn_utils.mlab_plot_measurements(measurement, mask, terrain, terrain_mode='blocks', terrain_uniform_color=True, blocking=False)

    ui.append(
        nn_utils.mlab_plot_prediction(prediction['pred'], terrain, terrain_mode='blocks', terrain_uniform_color=True,
                                      prediction_channels=config.params['model']['label_channels'], blocking=False))

nn_utils.plot_prediction(nn_params.data['label_channels'],
                         prediction = prediction['pred'][0].cpu().detach(),
                         provided_input_channels = nn_params.data['input_channels'],
                         input = input[0].cpu().detach(),
                         terrain = terrain.cpu().squeeze(),
                         label = ground_truth,
                         measurements = measurement[0].cpu().detach(),
                         measurements_mask = mask.squeeze().cpu())
