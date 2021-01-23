import numpy as np
import os
import torch

import nn_wind_prediction.data as nn_data
from .bin_log_data import bin_log_data
from .extract_cosmo_data import get_cosmo_cell
from .get_mapgeo_terrain import get_terrain
from .input_fn_definitions import *
from .ulog_utils import extract_wind_data

class SimpleStepOptimizer:
    def __init__(self, variables, lr=1e-4, lr_decay=0.0):
        self.var = variables
        self.learning_rate = lr
        self.lr_decay = lr_decay
        self.iterations = 0

    def __str__(self):
        return 'SimpleStepOptimizer, lr={0:0.2e}, decay={0:0.2e}'.format(self.learning_rate, self.lr_decay)

    def zero_grad(self):
        pass

    def step(self):
        # Basic gradient step
        with torch.no_grad():
            for v in self.var:
                v -= self.learning_rate * v.grad
                v.grad.zero_()
        self.iterations += 1
        self.learning_rate *= (1-self.lr_decay)

def get_optimizer(params, config):
    if config['name'] == 'SimpleStepOptimizer':
        return SimpleStepOptimizer(params, **config['kwargs'])
    elif config['name'] == 'Adadelta':
        return torch.optim.Adadelta(params, **config['kwargs'])
    elif config['name'] == 'Adagrad':
        return torch.optim.Adagrad(params, **config['kwargs'])
    elif config['name'] == 'Adam':
        return torch.optim.Adam(params, **config['kwargs'])
    elif config['name'] == 'Adamax':
        return torch.optim.Adamax(params, **config['kwargs'])
    elif config['name'] == 'ASGD':
        return torch.optim.ASGD(params, **config['kwargs'])
    elif config['name'] == 'SGD':
        return torch.optim.SGD(params, **config['kwargs'])
    else:
        raise ValueError('Invalid optimizer name: ' + config['name'])

def get_loss_fn(config):
    if config['name'] == 'L1':
        return torch.nn.L1Loss()
    elif config['name'] == 'MSE':
        return torch.nn.MSELoss()
    else:
        raise ValueError('Invalid loss name: ' + config['name'])

def get_input_fn(config, measurement, mask):
    # get some characteristics of the flow to initialize the params
    if mask is not None:
        scale = measurement.norm(dim=1).sum() / mask.sum()
    else:
        scale = 1.0

    idx = mask.squeeze().nonzero(as_tuple=True)
    
    angle = torch.atan2(measurement[:,1, idx[0], idx[1], idx[2]], measurement[:, 0, idx[0], idx[1], idx[2]]).mean().cpu().item()
    
    if config['type'] == 'bp_corners_1':
        initial_params = torch.Tensor([scale, angle]).to(measurement.device).requires_grad_()
        # boundary layer profile on each corner with a shared scale and rotation angle
        return lambda p, t, i: bp_corners_1(p, t, i, config['kwargs']), initial_params

    elif config['type'] == 'bp_corners_4':
        initial_params = torch.Tensor([scale, scale, scale, scale, angle, angle, angle, angle]).to(measurement.device).requires_grad_()
        # boundary layer profile on each corner with a separate scale and rotation angle
        return lambda p, t, i: bp_corners_4(p, t, i, config['kwargs']), initial_params

    elif config['type'] == 'bp_corners_1_roughness':
        roughness = mask.squeeze().shape[0]
        initial_params = torch.Tensor([scale, angle, roughness]).to(measurement.device).requires_grad_()
        # boundary layer profile on each corner with a shared scale and rotation angle
        return lambda p, t, i: bp_corners_1_roughness(p, t, i, config['kwargs']), initial_params

    else:
        raise ValueError('Invalid input_fn type: ' + config['type'])

def load_measurements(config, config_model):
    if config['type'] == 'cfd':
        dataset = nn_data.HDF5Dataset(config['cfd']['filename'], config_model['input_channels'],
                                      config_model['label_channels'], augmentation = False,
                                      return_grid_size = True, **config['cfd']['kwargs'])
        data = dataset[config['cfd']['index']]

        input_channels = dataset.get_input_channels()

        input = data[0]
        label = data[1].unsqueeze(0)

        terrain = input[0].unsqueeze(0).unsqueeze(0)
        if 'mask' in input_channels:
            measurement = input[1:-1].unsqueeze(0)
            mask = input[-1].unsqueeze(0)
        else:
            measurement = input[1:].unsqueeze(0)
            mask = None

    elif config['type'] == 'log':

        wind_data = extract_wind_data(config['log']['filename'], False)

        # determine the grid dimension
        if 'cosmo_file' in config['log'].keys():
            grid_dimensions = get_cosmo_cell(config['log']['cosmo_file'], wind_data['lat'][0], wind_data['lon'][0],
                                             wind_data['alt'].min() - config['log']['alt_offset'], config['log']['d_horizontal'], config['log']['d_vertical'])
            grid_dimensions['n_cells'] = config['log']['num_cells']
    
        else:
            grid_dimensions = {
                'n_cells': config['log']['num_cells'],
                'x_min': wind_data['x'].min() - 1.0,
                'x_max': wind_data['x'].max() + 1.0,
                'y_min': wind_data['y'].min() - 1.0,
                'y_max': wind_data['y'].max() + 1.0,
                'z_min': wind_data['alt'].min() - 20.0,
                'z_max': wind_data['alt'].max() + 1.0,
            }
    
            # force the grid to be square
            if (grid_dimensions['x_max'] - grid_dimensions['x_min']) > (grid_dimensions['y_max'] - grid_dimensions['y_min']):
                diff = (grid_dimensions['x_max'] - grid_dimensions['x_min']) - (grid_dimensions['y_max'] - grid_dimensions['y_min'])
                grid_dimensions['y_min'] -= 0.5 * diff
                grid_dimensions['y_max'] += 0.5 * diff
            else:
                diff = (grid_dimensions['y_max'] - grid_dimensions['y_min']) - (grid_dimensions['x_max'] - grid_dimensions['x_min'])
                grid_dimensions['x_min'] -= 0.5 * diff
                grid_dimensions['x_max'] += 0.5 * diff

        x_terr, y_terr, z_terr, h_terr, full_block = \
            get_terrain(config['log']['geotiff_file'],
                        [grid_dimensions['x_min'], grid_dimensions['x_max']],
                        [grid_dimensions['y_min'], grid_dimensions['y_max']],
                        [grid_dimensions['z_min'], grid_dimensions['z_max']],
                        (config['log']['num_cells'], config['log']['num_cells'], config['log']['num_cells']))

        terrain = torch.from_numpy(np.logical_not(full_block).astype('float'))

        measurement, variance, mask = bin_log_data(wind_data, grid_dimensions)

        terrain = terrain.unsqueeze(0).unsqueeze(0).float()
        measurement = measurement.unsqueeze(0).float()

        label = None

    else:
        raise ValueError('Invalid measurement type: ' + config['type'])

    if config_model['autoscale']:
        if mask is not None:
            scale = measurement.norm(dim=1).sum() / mask.sum()
            scale.clamp_(min = 1.0)
        else:
            scale = dataset.get_scale(measurement[0])
    else:
        scale = None

    return measurement, terrain, label, mask, scale

def predict(net, input, scale, config):
    if input.shape[1] != len(config['input_channels']):
        raise ValueError('The number of input channels is inconsistent')

    if scale is None:
        scale = torch.Tensor([1.0])

    for i, channel in enumerate(config['input_channels']):
        if 'ux' in channel or 'uy' in channel or 'uz' in channel:
            input[:, i] /= scale * config[channel + '_scaling']
        elif 'terrain' in channel:
            input[:, i] /= config[channel + '_scaling']
        else:
            raise ValueError('Unknown channel: ' + channel)

    prediction = net(input)

    for i, channel in enumerate(config['label_channels']):
        if 'ux' in channel or 'uy' in channel or 'uz' in channel:
            prediction['pred'][:, i] *= scale * config[channel + '_scaling']
        elif 'turb' in channel:
            prediction['pred'][:, i] *= scale * scale * config[channel + '_scaling']
        else:
            raise ValueError('Unknown channel: ' + channel)

    return prediction
