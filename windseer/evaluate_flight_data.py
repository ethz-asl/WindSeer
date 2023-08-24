import argparse
import numpy as np
import random
import torch

import windseer.evaluation as eval
import windseer.plotting as plotting
import windseer.utils as utils

parser = argparse.ArgumentParser(
    description='Evaluate the model performance on real flight data'
    )
parser.add_argument('config_yaml', help='Input yaml config')
parser.add_argument(
    '-model_dir', dest='model_dir', required=True, help='The directory of the model'
    )
parser.add_argument(
    '-model_version', dest='model_version', default='latest', help='The model version'
    )
parser.add_argument(
    '--paths', action='store_true', help='Plot the flight paths with mayavi'
    )
parser.add_argument(
    '--loiter',
    action='store_true',
    help='Evaluate the data using the loiter tools instead of along the flight path'
    )
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = utils.BasicParameters(args.config_yaml)

if config.params['measurements']['type'] != 'log':
    raise ValueError('Only using log data is supported by this script')

# load the NN
if config.params['evaluation']['compute_baseline']:
    net = None
    params = None

else:
    net, params = utils.load_model(
        args.model_dir, args.model_version, None, device, True
        )

    if params.data['input_mode'] < 3:
        raise ValueError('Models with an input mode other than 5 are not supported')

config = utils.update_sparse_config(config, params)

# load the data
data = eval.load_wind_data(config, args.loiter)

if args.paths:
    x_res = (data['grid_dimensions']['x_max'] -
             data['grid_dimensions']['x_min']) / data['grid_dimensions']['n_cells']
    y_res = (data['grid_dimensions']['y_max'] -
             data['grid_dimensions']['y_min']) / data['grid_dimensions']['n_cells']
    z_res = (data['grid_dimensions']['z_max'] -
             data['grid_dimensions']['z_min']) / data['grid_dimensions']['n_cells']
    trajectories = []
    flight_data = [data['wind_data']]
    if 'wind_data_validation' in data.keys():
        flight_data += data['wind_data_validation']

    for fd in flight_data:
        traj = {
            'x': (fd['x'] - data['grid_dimensions']['x_min']) / x_res,
            'y': (fd['y'] - data['grid_dimensions']['y_min']) / y_res,
            'z': (fd['alt'] - data['grid_dimensions']['z_min']) / z_res
            }
        trajectories.append(traj)

    plotting.mlab_plot_trajectories(
        trajectories,
        data['terrain'],
        terrain_mode='blocks',
        terrain_uniform_color=False,
        blocking=False
        )

if args.loiter:
    # add the mask to the input channel for the plotting scripts
    config.params['model']['input_channels'] += ['mask']
    eval.loiter_evaluation(data, net, config, device, True)
else:
    eval.evaluate_flight_log(
        data['wind_data'], data['scale'], data['terrain'], data['grid_dimensions'], net,
        config, device, data['wind_data_validation'], False, True
        )
