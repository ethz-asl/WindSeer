import argparse
import numpy as np
import random
import torch

import nn_wind_prediction.models as models
import nn_wind_prediction.utils as nn_utils
from analysis_utils import utils
from analysis_utils.ulog_utils import extract_wind_data, filter_wind_data
from analysis_utils.sparse_evaluation import evaluate_flight_log

parser = argparse.ArgumentParser(description='Evaluate the model performance on real flight data using a sliding window')
parser.add_argument('config_yaml', help='Input yaml config')
parser.add_argument('-model_dir', dest='model_dir', required=True, help='The directory of the model')
parser.add_argument('-model_version', dest='model_version', default='latest', help='The model version')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = nn_utils.BasicParameters(args.config_yaml)

if config.params['measurements']['type'] != 'log':
    raise ValueError('Only using log data is supported by this script')

# load the NN
if config.params['evaluation']['compute_baseline']:
    net = None

    config.params['model'] = {}
    config.params['model']['input_channels'] = ['ux', 'uy', 'uz']
    config.params['model']['label_channels'] = ['ux', 'uy', 'uz']
    config.params['model']['autoscale'] = False

else:
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
    config.params['model']['grid_size'] = nn_params.model['model_args']['grid_size']

    if 'input_smoothing' in nn_params.data.keys():
        config.params['model']['input_smoothing'] = nn_params.data['input_smoothing']
        config.params['model']['input_smoothing_interpolation'] = nn_params.data['input_smoothing_interpolation']
        config.params['model']['input_smoothing_interpolation_linear'] = nn_params.data['input_smoothing_interpolation_linear']
    else:
        config.params['model']['input_smoothing'] = False
    
    for key in nn_params.data.keys():
        if 'scaling' in key:
            config.params['model'][key] = nn_params.data[key]

# load the data
_, terrain, _, _, scale, wind_data, grid_dimensions = utils.load_measurements(config.params['measurements'], config.params['model'])

wind_data_validation = []
if not config.params['evaluation']['validation_file'] is None:
    for val_file in config.params['evaluation']['validation_file']:
        wd_val = extract_wind_data(val_file, False)
        if config.params['measurements']['log']['filter_window_size'] > 0:
                wd_val = filter_wind_data(wd_val, config.params['measurements']['log']['filter_window_size'])
        wind_data_validation.append(wd_val)

config.params['model']['input_channels'] += ['mask']

evaluate_flight_log(wind_data, scale, terrain, grid_dimensions, net, config, device, wind_data_validation)