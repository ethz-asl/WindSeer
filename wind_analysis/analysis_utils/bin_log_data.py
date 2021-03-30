import math
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import time
import torch

def bin_log_data_binning(wind_data, grid_dimensions, verbose = False):
    '''
    Bins the input wind data into a grid specified by the input grid dimensions.
    Currently assumes that in all 3 dimensions the number of cells is equal.

    After the binning is done the mean velocity and variance of each grid cell is computed.

    @out [wind]: (3,n,n,n) tensor containing the mean velocities of each cell
    @out [variance]: (3,n,n,n) tensor containing the velocity variance of each cell
    @out [mask]: (n,n,n) 1 indicates that wind was measured in that cell, 0 equals no measurements
    @out [prediction]: None since this method does not predict the flow over the full field
    '''
    # initialize lists for binning the wind data
    wx = [[[[] for k in range(grid_dimensions['n_cells'])] for j in range(grid_dimensions['n_cells'])] for i in range(grid_dimensions['n_cells'])]
    wy = [[[[] for k in range(grid_dimensions['n_cells'])] for j in range(grid_dimensions['n_cells'])] for i in range(grid_dimensions['n_cells'])]
    wz = [[[[] for k in range(grid_dimensions['n_cells'])] for j in range(grid_dimensions['n_cells'])] for i in range(grid_dimensions['n_cells'])]

    # determine the resolution of the grid
    x_res = (grid_dimensions['x_max'] - grid_dimensions['x_min']) / grid_dimensions['n_cells']
    y_res = (grid_dimensions['y_max'] - grid_dimensions['y_min']) / grid_dimensions['n_cells']
    z_res = (grid_dimensions['z_max'] - grid_dimensions['z_min']) / grid_dimensions['n_cells']

    # loop over the data and bin it
    for i in range(len(wind_data['x'])):
        if ((wind_data['x'][i] > grid_dimensions['x_min']) and
            (wind_data['x'][i] < grid_dimensions['x_max']) and
            (wind_data['y'][i] > grid_dimensions['y_min']) and
            (wind_data['y'][i] < grid_dimensions['y_max']) and
            (wind_data['alt'][i] > grid_dimensions['z_min']) and
            (wind_data['alt'][i] < grid_dimensions['z_max'])):

            idx_x = int((wind_data['x'][i] - grid_dimensions['x_min']) / x_res)
            idx_y = int((wind_data['y'][i] - grid_dimensions['y_min']) / y_res)
            idx_z = int((wind_data['alt'][i] - grid_dimensions['z_min']) / z_res)

            wx[idx_z][idx_y][idx_x].append(wind_data['we'][i])
            wy[idx_z][idx_y][idx_x].append(wind_data['wn'][i])
            wz[idx_z][idx_y][idx_x].append(-wind_data['wd'][i])

    wind = torch.zeros((3, grid_dimensions['n_cells'],grid_dimensions['n_cells'],grid_dimensions['n_cells']))
    variance = torch.zeros((3, grid_dimensions['n_cells'],grid_dimensions['n_cells'],grid_dimensions['n_cells']))
    mask = torch.zeros((grid_dimensions['n_cells'],grid_dimensions['n_cells'],grid_dimensions['n_cells']))

    vals_per_cell = []
    for i in range(grid_dimensions['n_cells']):
        for j in range(grid_dimensions['n_cells']):
            for k in range(grid_dimensions['n_cells']):
                if wx[i][j][k]:
                    wind[0, i, j, k] = np.mean(wx[i][j][k])
                    wind[1, i, j, k] = np.mean(wy[i][j][k])
                    wind[2, i, j, k] = np.mean(wz[i][j][k])

                    variance[0, i, j, k] = np.var(wx[i][j][k])
                    variance[1, i, j, k] = np.var(wy[i][j][k])
                    variance[2, i, j, k] = np.var(wz[i][j][k])

                    mask[i, j, k] = 1
                    vals_per_cell.append(len(wx[i][j][k]))

    if verbose:
        print('')
        print('\tNumber of cells with values:     {}'.format(mask.sum()))
        print('\tPercentage of cells with values: {:.2f}'.format(100 * mask.sum() / (grid_dimensions['n_cells']*grid_dimensions['n_cells']*grid_dimensions['n_cells'])))
        print('\tNumber of values per cell (avg): {:.2f}'.format(np.mean(vals_per_cell)))
    return wind, variance, mask, None

def bin_log_data_idw_interpolation(wind_data, grid_dimensions, verbose = False):
    '''
    Compute the wind value at the center of each bin using an inverse distance
    weighted interpolation scheme.

    @out [wind]: (3,n,n,n) tensor containing the mean velocities of each cell
    @out [variance]: (3,n,n,n) tensor containing the velocity variance of each cell
    @out [mask]: (n,n,n) 1 indicates that wind was measured in that cell, 0 equals no measurements
    @out [prediction]: None since this method does not predict the flow over the full field
    '''
    # Initialize distance lists for the wind_data
    wx = [[[[] for k in range(grid_dimensions['n_cells'])] for j in range(grid_dimensions['n_cells'])] for i in range(grid_dimensions['n_cells'])]
    wy = [[[[] for k in range(grid_dimensions['n_cells'])] for j in range(grid_dimensions['n_cells'])] for i in range(grid_dimensions['n_cells'])]
    wz = [[[[] for k in range(grid_dimensions['n_cells'])] for j in range(grid_dimensions['n_cells'])] for i in range(grid_dimensions['n_cells'])]
    id = [[[[] for k in range(grid_dimensions['n_cells'])] for j in range(grid_dimensions['n_cells'])] for i in range(grid_dimensions['n_cells'])]

    # determine the resolution of the grid
    x_res = (grid_dimensions['x_max'] - grid_dimensions['x_min']) / grid_dimensions['n_cells']
    y_res = (grid_dimensions['y_max'] - grid_dimensions['y_min']) / grid_dimensions['n_cells']
    z_res = (grid_dimensions['z_max'] - grid_dimensions['z_min']) / grid_dimensions['n_cells']

    # Calculate distance of points to the center of the bin
    for i in range(len(wind_data['x'])):
        if ((wind_data['x'][i] > grid_dimensions['x_min']) and
            (wind_data['x'][i] < grid_dimensions['x_max']) and
            (wind_data['y'][i] > grid_dimensions['y_min']) and
            (wind_data['y'][i] < grid_dimensions['y_max']) and
            (wind_data['alt'][i] > grid_dimensions['z_min']) and
            (wind_data['alt'][i] < grid_dimensions['z_max'])):

            idx_x = int((wind_data['x'][i] - grid_dimensions['x_min']) / x_res)
            idx_y = int((wind_data['y'][i] - grid_dimensions['y_min']) / y_res)
            idx_z = int((wind_data['alt'][i] - grid_dimensions['z_min']) / z_res)

            x_cell = (idx_x + 0.5) * x_res + grid_dimensions['x_min']
            y_cell = (idx_y + 0.5) * y_res + grid_dimensions['y_min']
            z_cell = (idx_z + 0.5) * z_res + grid_dimensions['z_min']

            distance = math.sqrt((wind_data['x'][i] - x_cell) ** 2 +
                                 (wind_data['y'][i] - y_cell) ** 2 +
                                 (wind_data['alt'][i] - z_cell) ** 2)

            wx[idx_z][idx_y][idx_x].append(wind_data['we'][i])
            wy[idx_z][idx_y][idx_x].append(wind_data['wn'][i])
            wz[idx_z][idx_y][idx_x].append(-wind_data['wd'][i])
            id[idx_z][idx_y][idx_x].append(1.0 / distance)

    wind = torch.zeros((3, grid_dimensions['n_cells'], grid_dimensions['n_cells'], grid_dimensions['n_cells']))
    variance = torch.zeros((3, grid_dimensions['n_cells'], grid_dimensions['n_cells'], grid_dimensions['n_cells']))
    mask = torch.zeros((grid_dimensions['n_cells'],grid_dimensions['n_cells'],grid_dimensions['n_cells']))

    vals_per_cell = []
    for i in range(grid_dimensions['n_cells']):
        for j in range(grid_dimensions['n_cells']):
            for k in range(grid_dimensions['n_cells']):
                if wx[i][j][k]:
                    wind[0, i, j, k] = np.average(wx[i][j][k], axis=None, weights=id[i][j][k])
                    wind[1, i, j, k] = np.average(wy[i][j][k], axis=None, weights=id[i][j][k])
                    wind[2, i, j, k] = np.average(wz[i][j][k], axis=None, weights=id[i][j][k])

                    variance[0, i, j, k] = np.var(wx[i][j][k])
                    variance[1, i, j, k] = np.var(wy[i][j][k])
                    variance[2, i, j, k] = np.var(wz[i][j][k])

                    mask[i, j, k] = 1
                    vals_per_cell.append(len(wx[i][j][k]))

    if verbose:
        print('')
        print('\tNumber of cells with values:     {}'.format(mask.sum()))
        print('\tPercentage of cells with values: {:.2f}'.format(
            100 * mask.sum() / (grid_dimensions['n_cells']
                             * grid_dimensions['n_cells']
                             * grid_dimensions['n_cells'])))
        print('\tNumber of values per cell (avg): {:.2f}'.format(np.mean(vals_per_cell)))
    return wind, variance, mask, None

def interpolate_flight_data_gpr(wind_data, grid_dimensions, verbose = False, predict = False):
    '''
    Create a wind map from the wind measurements using Gaussian Process Regression
    for interpolation.
    Compute the mean velocity and variance at the center of each bin by
    evaluating the wind map.

    The return consists of three tensors:
    @out [wind]: (3,n,n,n) tensor containing the mean velocities of each cell
    @out [variance]: (3,n,n,n) tensor containing the velocity variance of each cell
    @out [mask]: (n,n,n) 1 indicates that wind was measured in that cell, 0 equals no measurements
    @out [prediction]: If predict is true then this tensor contains the flow of the full field according to the gp
    '''
    t_start = time.time()
    # Initialize empty wind, variance and predicted_flight_data
    wind = torch.zeros((3, grid_dimensions['n_cells'], grid_dimensions['n_cells'], grid_dimensions['n_cells']))
    variance = torch.zeros((3, grid_dimensions['n_cells'], grid_dimensions['n_cells'], grid_dimensions['n_cells']))
    mask = torch.zeros((grid_dimensions['n_cells'], grid_dimensions['n_cells'], grid_dimensions['n_cells']))

    # determine the resolution of the grid
    x_res = (grid_dimensions['x_max'] - grid_dimensions['x_min']) / grid_dimensions['n_cells']
    y_res = (grid_dimensions['y_max'] - grid_dimensions['y_min']) / grid_dimensions['n_cells']
    z_res = (grid_dimensions['z_max'] - grid_dimensions['z_min']) / grid_dimensions['n_cells']

    for i in range(len(wind_data['x'])):
        if ((wind_data['x'][i] > grid_dimensions['x_min']) and
            (wind_data['x'][i] < grid_dimensions['x_max']) and
            (wind_data['y'][i] > grid_dimensions['y_min']) and
            (wind_data['y'][i] < grid_dimensions['y_max']) and
            (wind_data['alt'][i] > grid_dimensions['z_min']) and
            (wind_data['alt'][i] < grid_dimensions['z_max'])):

            idx_x = int((wind_data['x'][i] - grid_dimensions['x_min']) / x_res)
            idx_y = int((wind_data['y'][i] - grid_dimensions['y_min']) / y_res)
            idx_z = int((wind_data['alt'][i] - grid_dimensions['z_min']) / z_res)

            mask[idx_z, idx_y, idx_x] = 1

    idx_mask = mask.nonzero(as_tuple=False)

    # Instantiate a Gaussian Process model for each direction
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(100, (1e-3, 1e3))
    gp_x = GPR(kernel=kernel, n_restarts_optimizer=10, alpha=0.2, normalize_y=True)
    gp_y = GPR(kernel=kernel, n_restarts_optimizer=10, alpha=0.2, normalize_y=True)
    gp_z = GPR(kernel=kernel, n_restarts_optimizer=10, alpha=0.2, normalize_y=True)

    max_meas = 100
    stride = int(np.ceil(len(wind_data['alt']) / max_meas))

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp_x.fit(np.column_stack([wind_data['alt'][::stride], wind_data['y'][::stride], wind_data['x'][::stride]]),
             wind_data['we'][::stride])
    gp_y.fit(np.column_stack([wind_data['alt'][::stride], wind_data['y'][::stride], wind_data['x'][::stride]]),
             wind_data['wn'][::stride])
    gp_z.fit(np.column_stack([wind_data['alt'][::stride], wind_data['y'][::stride], wind_data['x'][::stride]]),
             -wind_data['wd'][::stride])

    for idx in idx_mask:
        x = grid_dimensions['x_min'] + (idx[2] + 0.5) * x_res
        y = grid_dimensions['y_min'] + (idx[1] + 0.5) * x_res
        z = grid_dimensions['z_min'] + (idx[0] + 0.5) * x_res

        mean, var = gp_x.predict(np.column_stack([z, y, x]), return_std=True)
        wind[0, idx[0], idx[1], idx[2]] = mean.item()
        variance[0, idx[0], idx[1], idx[2]] = var.item()
        # y
        mean, var = gp_y.predict(np.column_stack([z, y, x]), return_std=True)
        wind[1, idx[0], idx[1], idx[2]] = mean.item()
        variance[1, idx[0], idx[1], idx[2]] = var.item()
        # z
        mean, var = gp_z.predict(np.column_stack([z, y, x]), return_std=True)
        wind[2, idx[0], idx[1], idx[2]] = mean.item()
        variance[2, idx[0], idx[1], idx[2]] = var.item()

    if predict:
        if verbose:
            print(' Interpolation is done [{:.2f} s], predicting full domain...'.format(time.time() - t_start), end='', flush=True)
        prediction = torch.zeros((3, grid_dimensions['n_cells'], grid_dimensions['n_cells'], grid_dimensions['n_cells']))

        for i in range(grid_dimensions['n_cells']):
            for j in range(grid_dimensions['n_cells']):
                for k in range(grid_dimensions['n_cells']):
                    x = grid_dimensions['x_min'] + (k + 0.5) * x_res
                    y = grid_dimensions['y_min'] + (j + 0.5) * y_res
                    z = grid_dimensions['z_min'] + (i + 0.5) * z_res

                    mean, var = gp_x.predict(np.column_stack([z, y, x]), return_std=True)
                    prediction[0, i, j, k] = mean.item()
                    # y
                    mean, var = gp_y.predict(np.column_stack([z, y, x]), return_std=True)
                    prediction[1, i, j, k] = mean.item()
                    # z
                    mean, var = gp_z.predict(np.column_stack([z, y, x]), return_std=True)
                    prediction[2, i, j, k] = mean.item()
    else:
        prediction = None

    if verbose:
        print('GPR interpolation is done [{:.2f} s]'.format(time.time() - t_start))

    return wind, variance, mask, prediction

def extract_window_wind_data(wind_data, t_start, t_end):
    t_init = wind_data['time'][0] * 1e-6

    # extract the relevant data if t_start or t_end are set
    if t_end is not None or t_start is not None:
        if t_start is None:
            t_start = 0.0
        elif t_start < 0.0:
            t_start = 0.0

        if t_end is None:
            t_end = np.inf
        elif t_end < 0.0:
            t_end = np.inf

        t_rel = wind_data['time'] * 1e-6 - t_init
        idx = np.logical_and(t_rel >= t_start, t_rel <= t_end)

        wind_out = {}
        for key in wind_data.keys():
            wind_out[key] = wind_data[key][idx]

        return wind_out

    else:
        return wind_data

def bin_log_data(wind_data, grid_dimensions, method = 'binning', verbose = False, full_field = False, t_start = None, t_end = None):
    wind_data = extract_window_wind_data(wind_data, t_start, t_end)

    if method == 'binning':
        return bin_log_data_binning(wind_data, grid_dimensions, verbose)

    elif method == 'interpolation':
        return bin_log_data_idw_interpolation(wind_data, grid_dimensions, verbose)

    elif method == 'gpr':
        return interpolate_flight_data_gpr(wind_data, grid_dimensions, verbose, full_field)

    else:
        raise ValueError('Unknown method: ' + method)
