import numpy as np
import torch

from .interpolation import interpolate_sparse_data
from .HDF5Dataset import HDF5Dataset
import windseer.utils as utils


def load_measurements(config, config_model):
    # manually add uz if it was not present in the input channels since we want to return all the velocities as measurements
    measurement_channels = ['terrain', 'ux', 'uy', 'uz']
    return_variance = False
    if 'turb' in config_model['label_channels']:
        return_variance = True
        measurement_channels += ['turb']

    for ch in config_model['input_channels']:
        if not ch in measurement_channels:
            raise ValueError(
                'Model has an input channel that is currently not supported: ', ch,
                'supported: ', measurement_channels
                )

    if config['type'] == 'cfd':
        dataset = HDF5Dataset(
            config['cfd']['filename'],
            measurement_channels,
            config_model['label_channels'],
            augmentation=False,
            return_grid_size=True,
            **config['cfd']['kwargs']
            )
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

        grid_dimensions = None

    elif config['type'] == 'log':
        wind_data = utils.extract_wind_data(config['log']['filename'], False)

        if config['log']['filter_window_size'] > 0:
            wind_data = utils.filter_wind_data(
                wind_data, config['log']['filter_window_size']
                )

        # determine the grid dimension
        if config['log']['use_cosmo_grid']:
            if not 'cosmo_file' in config['log'].keys():
                print('Cosmo file not specified, using the manually defined grid')
                config['log']['use_cosmo_grid'] = False

        if config['log']['use_cosmo_grid']:
            grid_dimensions = utils.get_cosmo_cell(
                config['log']['cosmo_file'], wind_data['lat'][0], wind_data['lon'][0],
                wind_data['alt'].min() - config['log']['alt_offset'],
                config['log']['d_horizontal'], config['log']['d_vertical']
                )
            grid_dimensions['n_cells'] = config['log']['num_cells']

        else:
            if config['log']['enforce_grid_size']:
                # center the grid around the takeoff location
                grid_dimensions = {
                    'n_cells':
                        config['log']['num_cells'],
                    'x_min':
                        wind_data['x'][0] - 0.5 * config['log']['d_horizontal'],
                    'x_max':
                        wind_data['x'][0] + 0.5 * config['log']['d_horizontal'],
                    'y_min':
                        wind_data['y'][0] - 0.5 * config['log']['d_horizontal'],
                    'y_max':
                        wind_data['y'][0] + 0.5 * config['log']['d_horizontal'],
                    'z_min':
                        wind_data['alt'].min() - config['log']['alt_offset'],
                    'z_max':
                        wind_data['alt'].min() - config['log']['alt_offset'] +
                        config['log']['d_vertical'],
                    }

            else:
                # adjust the grid size based on the flight data
                print(
                    'Warning: This does not preserve the normal grid size and could lead to issues when using the network for a prediction'
                    )
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
            if (grid_dimensions['x_max'] - grid_dimensions['x_min']
                ) > (grid_dimensions['y_max'] - grid_dimensions['y_min']):
                diff = (grid_dimensions['x_max'] - grid_dimensions['x_min']) - (
                    grid_dimensions['y_max'] - grid_dimensions['y_min']
                    )
                grid_dimensions['y_min'] -= 0.5 * diff
                grid_dimensions['y_max'] += 0.5 * diff
            else:
                diff = (grid_dimensions['y_max'] - grid_dimensions['y_min']) - (
                    grid_dimensions['x_max'] - grid_dimensions['x_min']
                    )
                grid_dimensions['x_min'] -= 0.5 * diff
                grid_dimensions['x_max'] += 0.5 * diff

        _, _, _, _, full_block = \
            utils.get_terrain(config['log']['geotiff_file'],
                        [grid_dimensions['x_min'], grid_dimensions['x_max']],
                        [grid_dimensions['y_min'], grid_dimensions['y_max']],
                        [grid_dimensions['z_min'], grid_dimensions['z_max']],
                        (config['log']['num_cells'], config['log']['num_cells'], config['log']['num_cells']),
                        build_block = True,
                        return_is_wind = True,
                        distance_field = config['log']['distance_field'],
                        horizontal_overflow = config['log']['horizontal_overflow'])

        terrain = torch.from_numpy(full_block.astype(np.float32))

        t_start = None
        t_end = None
        if config['log']['t_start'] is not None:
            t_start = config['log']['t_start'] + wind_data['time'][0] * 1e-6

        if config['log']['t_end'] is not None:
            t_end = config['log']['t_end'] + wind_data['time'][0] * 1e-6

        measurement, variance, mask, prediction = utils.bin_log_data(
            wind_data, grid_dimensions, method='binning', t_start=t_start, t_end=t_end
            )

        terrain = terrain.unsqueeze(0).unsqueeze(0).float()
        measurement = measurement.unsqueeze(0).float()
        if return_variance:
            measurement = torch.cat([
                measurement,
                variance.float().sum(dim=0).unsqueeze(0).unsqueeze(0)
                ],
                                    dim=1)
        mask = mask.unsqueeze(0).float()

        # mask the measurements by the terrain to avoid having nonzero measurements inside the terrain
        # which was not seen by the models during training
        terrain_mask = (terrain > 0).float()
        measurement *= terrain_mask
        mask *= terrain_mask[0]

        label = None

    else:
        raise ValueError('Invalid measurement type: ' + config['type'])

    if config_model['autoscale']:
        if mask is not None:
            scale = measurement.norm(dim=1).sum() / mask.sum()
            scale.clamp_(min=1.0)
        else:
            scale = dataset.get_scale(measurement[0])
    else:
        scale = None

    return measurement, terrain, label, mask, scale, wind_data, grid_dimensions
