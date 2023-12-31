import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import torch

import windseer.utils as utils
import windseer.plotting as plotting

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        'Plot the measured wind during the flight in the same grid as the prediction is made'
        )
    parser.add_argument('input_file', help='Input file, either ulog or hdf5')
    parser.add_argument(
        '-n',
        '--n_cells',
        type=int,
        default=64,
        help='Number of grid cells in one dimension'
        )
    parser.add_argument(
        '-b',
        '--binnning_method',
        default='binning',
        help='Method used to bin the flight data into the grid'
        )
    parser.add_argument(
        '-d_hor',
        '--d_horizontal',
        type=float,
        default=1100,
        help='Horizontal side length of the grid [m]'
        )
    parser.add_argument(
        '-d_ver',
        '--d_vertical',
        type=float,
        default=733,
        help='Horizontal side length of the grid [m]'
        )
    parser.add_argument(
        '-est',
        '--estimation',
        action='store_true',
        help=
        'In case of the ulog file take the wind from the EKF instead of the postprocessed one'
        )
    parser.add_argument(
        '-cosmo', '--cosmo_file', help='Cosmo file name to determine the grid size'
        )
    parser.add_argument(
        '-tiff', '--geotiff_file', required=True, help='Filename of the geotiff file'
        )
    parser.add_argument(
        '-p',
        '--predict',
        action='store_true',
        help='Predict the full field in case the binning method supports it'
        )

    args = parser.parse_args()

    # load the measured wind data from the log files
    wind_data = utils.extract_wind_data(args.input_file, args.estimation)

    # determine the grid dimension
    if (args.cosmo_file):
        print('Parsing COSMO file to determine grid location...', end='', flush=True)
        t_start = time.time()
        grid_dimensions = utils.get_cosmo_cell(
            args.cosmo_file, wind_data['lat'][0], wind_data['lon'][0],
            wind_data['alt'].min() - 20.0, args.d_horizontal, args.d_vertical
            )
        grid_dimensions['n_cells'] = args.n_cells

        print(' done [{:.2f} s]'.format(time.time() - t_start))
    else:
        print('Defining grid solely based on flight data...', end='', flush=True)
        t_start = time.time()
        grid_dimensions = {
            'n_cells': args.n_cells,
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

        print(' done [{:.2f} s]'.format(time.time() - t_start))

    # bin the data into the regular grid
    print('Binning wind data...', end='', flush=True)
    t_start = time.time()
    wind, variance, measurements_mask, prediction = utils.bin_log_data(
        wind_data, grid_dimensions, args.binnning_method, True, args.predict
        )
    print(' done [{:.2f} s]'.format(time.time() - t_start))

    # get the terrain data
    if (args.geotiff_file):
        print('Extracting terrain from geotiff...', end='', flush=True)
        t_start = time.time()
        x_terr, y_terr, z_terr, h_terr, full_block = \
            utils.get_mapgeo_terrain.get_terrain(args.geotiff_file, [grid_dimensions['x_min'], grid_dimensions['x_max']], [grid_dimensions['y_min'], grid_dimensions['y_max']],
                    [grid_dimensions['z_min'], grid_dimensions['z_max']], (args.n_cells, args.n_cells, args.n_cells))

        # convert to torch tensor since the plottools expect the terrain to be a tensor
        terrain = torch.from_numpy(np.logical_not(full_block).astype('float'))
        print(' done [{:.2f} s]'.format(time.time() - t_start))
    else:
        # no terrain
        terrain = torch.ones_like(wind[0])

    plotting.plot_measurement(['ux', 'uy', 'uz'],
                              wind,
                              measurements_mask,
                              terrain=terrain,
                              variance=variance,
                              prediction=prediction)
