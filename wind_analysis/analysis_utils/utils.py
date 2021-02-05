import numpy as np
import os
from scipy import ndimage
import torch
from torch.optim.optimizer import Optimizer

import nn_wind_prediction.data as nn_data
from .bin_log_data import bin_log_data
from .extract_cosmo_data import get_cosmo_cell
from .get_mapgeo_terrain import get_terrain
from .input_fn_definitions import *
from .ulog_utils import extract_wind_data

class SimpleStepOptimizer(Optimizer):
    def __init__(self, params, lr=1e-4):
        self.lr = lr
        defaults = dict(lr=lr)
        super(SimpleStepOptimizer, self).__init__(params, defaults)

    def __str__(self):
        return 'SimpleStepOptimizer, lr={0:0.2e}'.format(self.lr)

    def zero_grad(self):
        pass

    def step(self):
        # Basic gradient step
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    p -= group['lr']* p.grad

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
    elif config['name'] == 'Ranger':
        from ranger import Ranger
        return Ranger(params, **config['kwargs'])
    elif config['name'] == 'RAdam':
        from radam import RAdam
        return RAdam(params, **config['kwargs'])
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
        #scale = measurement.norm(dim=1).sum() / mask.sum()
        # starting from a underestimation of the wind magnitude seems to be more stable
        scale = 1.0
    else:
        scale = 1.0

    idx = mask.squeeze().nonzero(as_tuple=True)
    
    angle = torch.atan2(measurement[:,1, idx[0], idx[1], idx[2]], measurement[:, 0, idx[0], idx[1], idx[2]]).mean().cpu().item()
    
    if config['type'] == 'bp_corners_1':
        initial_params = torch.Tensor([[scale, angle]]).to(measurement.device).requires_grad_()

        return lambda p, t, i: bp_corners(p, t, i, config['kwargs']), initial_params

    elif config['type'] == 'bp_corners_4':
        initial_params = torch.Tensor([[scale, angle]]).repeat(4, 1).clone().to(measurement.device).requires_grad_()

        return lambda p, t, i: bp_corners(p, t, i, config['kwargs']), initial_params

    elif config['type'] == 'bp_corners_1_roughness':
        roughness = mask.squeeze().shape[0]
        initial_params = torch.Tensor([[scale, angle, roughness]]).to(measurement.device).requires_grad_()

        return lambda p, t, i: bp_corners(p, t, i, config['kwargs']), initial_params

    elif config['type'] == 'bp_corners_4_roughness':
        roughness = mask.squeeze().shape[0]
        initial_params = torch.Tensor([[scale, angle, roughness]]).repeat(4, 1).clone().to(measurement.device).requires_grad_()

        return lambda p, t, i: bp_corners(p, t, i, config['kwargs']), initial_params

    elif config['type'] == 'splines_corner':
        if config['kwargs']['uz_zero']:
            num_channels = config['kwargs']['num_channels'] - 1
        else:
            num_channels = config['kwargs']['num_channels']

        mean_vel = (measurement.sum(dim=-1).sum(dim=-1).sum(dim=-1) / mask.sum()).unsqueeze(-1)[:,:num_channels]
        scale = ((torch.arange(config['kwargs']['num_control_points']) + 1.0) / config['kwargs']['num_control_points']).to(measurement.device)
        initial_params = (mean_vel * scale.unsqueeze(0).unsqueeze(0)).repeat(4, 1, 1).clone().requires_grad_()

        return lambda p, t, i: splines_corner(p, t, i, config['kwargs']), initial_params

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

        wind_data = None

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

        is_wind = np.logical_not(full_block).astype('float')
        if config['log']['distance_field']:
            terrain = torch.from_numpy(ndimage.distance_transform_edt(is_wind).astype(np.float32))
        else:
            terrain = torch.from_numpy(is_wind.astype(np.float32))


        measurement, variance, mask, prediction = bin_log_data(wind_data, grid_dimensions)

        terrain = terrain.unsqueeze(0).unsqueeze(0).float()
        measurement = measurement.unsqueeze(0).float()
        mask = mask.unsqueeze(0).float()

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

    return measurement, terrain, label, mask, scale, wind_data

def predict(net, input, scale, config):
    if input.shape[1] != len(config['input_channels']):
        raise ValueError('The number of input channels is inconsistent')

    if scale is None:
        scale = torch.Tensor([1.0]).to(input.device)

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
