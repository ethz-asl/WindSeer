import numpy as np
import torch

def bin_log_data(wind_data, grid_dimensions):
    '''
    Bins the input wind data into a grid specified by the input grid dimensions.
    Currently assumes that in all 3 dimensions the number of cells is equal.

    After the binning is done the mean velocity and variance of each grid cell is computed.

    The return consists of three tensors:
    @out [wind]: (3,n,n,n) tensor containing the mean velocities of each cell
    @out [variance]: (3,n,n,n) tensor containing the velocity variance of each cell
    @out [mask]: (n,n,n) 1 indicates that wind was measured in that cell, 0 equals no measurements
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

            wx[idx_z][idx_y][idx_x].append(wind_data['wn'][i])
            wy[idx_z][idx_y][idx_x].append(wind_data['we'][i])
            wz[idx_z][idx_y][idx_x].append(-wind_data['wd'][i])

    wind = torch.zeros((3, grid_dimensions['n_cells'],grid_dimensions['n_cells'],grid_dimensions['n_cells'])) * float('nan')
    variance = torch.zeros((3, grid_dimensions['n_cells'],grid_dimensions['n_cells'],grid_dimensions['n_cells'])) * float('nan')

    counter = 0
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

                    counter += 1
                    vals_per_cell.append(len(wx[i][j][k]))
    print('')
    print('\tNumber of cells with values:     {}'.format(counter))
    print('\tPercentage of cells with values: {:.2f}'.format(100* counter / (grid_dimensions['n_cells']*grid_dimensions['n_cells']*grid_dimensions['n_cells'])))
    print('\tNumber of values per cell (avg): {:.2f}'.format(np.mean(vals_per_cell)))
    return wind, variance
