import numpy as np
import torch
import math
from scipy.interpolate import RegularGridInterpolator


def interpolate_log_data_from_grid(terrain, predicted_wind_data, wind_data):
    my_interpolating_function_x = RegularGridInterpolator((terrain.x_terr, terrain.y_terr, terrain.z_terr),
                                                        predicted_wind_data[0, :, :, :].detach().cpu().numpy())
    # Initialize empty list of points to interpolate at
    pts_x = [[] for i in range(len(wind_data['x']))]
    for i in range(len(wind_data['x'])):
        if ((wind_data['x'][i] > terrain.x_terr[0]) and
                (wind_data['x'][i] < terrain.x_terr[-1]) and
                (wind_data['y'][i] > terrain.y_terr[0]) and
                (wind_data['y'][i] < terrain.y_terr[-1]) and
                (wind_data['alt'][i] > terrain.z_terr[0]) and
                (wind_data['alt'][i] < terrain.z_terr[-1])):
            pts_x[i].append([wind_data['x'][i], wind_data['y'][i], wind_data['alt'][i]])

    # Remove empty lists from pts
    pts_x = [x for x in pts_x if x != []]

    interpolated_log_data_x = my_interpolating_function_x(pts_x)
    return interpolated_log_data_x


def interpolate_log_data(wind_data, grid, terrain):
    # initialize wind lists for the wind data
    wx = [[[[] for k in range(grid['n_cells'])] for j in range(grid['n_cells'])] for i in range(grid['n_cells'])]
    wy = [[[[] for k in range(grid['n_cells'])] for j in range(grid['n_cells'])] for i in range(grid['n_cells'])]
    wz = [[[[] for k in range(grid['n_cells'])] for j in range(grid['n_cells'])] for i in range(grid['n_cells'])]

    # Initialize distance lists for the wind_data
    id_xyz = [[[[] for k in range(grid['n_cells'])] for j in range(grid['n_cells'])] for i in range(grid['n_cells'])]

    # determine the resolution of the grid
    x_res = (grid['x_max'] - grid['x_min']) / grid['n_cells']
    y_res = (grid['y_max'] - grid['y_min']) / grid['n_cells']
    z_res = (grid['z_max'] - grid['z_min']) / grid['n_cells']

    # loop over the data and bin it
    for i in range(len(wind_data['x'])):
        if ((wind_data['x'][i] > grid['x_min']) and
                (wind_data['x'][i] < grid['x_max']) and
                (wind_data['y'][i] > grid['y_min']) and
                (wind_data['y'][i] < grid['y_max']) and
                (wind_data['alt'][i] > grid['z_min']) and
                (wind_data['alt'][i] < grid['z_max'])):
            idx_x = int((wind_data['x'][i] - grid['x_min']) / x_res)
            idx_y = int((wind_data['y'][i] - grid['y_min']) / y_res)
            idx_z = int((wind_data['alt'][i] - grid['z_min']) / z_res)

            distance = math.sqrt((wind_data['x'][i] - terrain.x_terr[idx_x]) ** 2
                                 + (wind_data['y'][i] - terrain.y_terr[idx_y]) ** 2
                                 + (wind_data['alt'][i] - terrain.z_terr[idx_z]) ** 2)

            wx[idx_z][idx_y][idx_x].append(wind_data['wn'][i])
            wy[idx_z][idx_y][idx_x].append(wind_data['we'][i])
            wz[idx_z][idx_y][idx_x].append(-wind_data['wd'][i])
            id_xyz[idx_z][idx_y][idx_x].append(1 / distance)

    wind = torch.zeros((3, grid['n_cells'], grid['n_cells'], grid['n_cells']))
    for i in range(grid['n_cells']):
        for j in range(grid['n_cells']):
            for k in range(grid['n_cells']):
                if wx[i][j][k]:
                    wind[0, i, j, k] = np.average(wx[i][j][k], axis=None, weights=id_xyz[i][j][k])
                    wind[1, i, j, k] = np.average(wy[i][j][k], axis=None, weights=id_xyz[i][j][k])
                    wind[2, i, j, k] = np.average(wz[i][j][k], axis=None, weights=id_xyz[i][j][k])

    return wind


def krig_log_data(wind_data, grid, terrain, OK3d_north, OK3d_east, OK3d_down):
    x_res = (grid['x_max'] - grid['x_min']) / grid['n_cells']
    y_res = (grid['y_max'] - grid['y_min']) / grid['n_cells']
    z_res = (grid['z_max'] - grid['z_min']) / grid['n_cells']

    # Initialize empty wind and variance
    wind = torch.zeros(
        (3, grid['n_cells'], grid['n_cells'], grid['n_cells'])) * float('nan')
    variance = torch.zeros(
        (3, grid['n_cells'], grid['n_cells'], grid['n_cells'])) * float('nan')

    # loop over the data and create the kriged grid and the variance grid
    for i in range(len(wind_data['x'])):
        if ((wind_data['x'][i] > grid['x_min']) and
                (wind_data['x'][i] < grid['x_max']) and
                (wind_data['y'][i] > grid['y_min']) and
                (wind_data['y'][i] < grid['y_max']) and
                (wind_data['alt'][i] > grid['z_min']) and
                (wind_data['alt'][i] < grid['z_max'])):
            idx_x = int((wind_data['x'][i] - grid['x_min']) / x_res)
            idx_y = int((wind_data['y'][i] - grid['y_min']) / y_res)
            idx_z = int((wind_data['alt'][i] - grid['z_min']) / z_res)
            gridx = terrain.x_terr[idx_x]
            gridy = terrain.y_terr[idx_y]
            gridz = terrain.z_terr[idx_z]
            k3d, ss3d = OK3d_north.execute('grid', gridx, gridy, gridz)
            wind[0, idx_x, idx_y, idx_z] = k3d[0][0][0]; variance[0, idx_x, idx_y, idx_z] = ss3d[0][0][0]
            k3d, ss3d = OK3d_east.execute('grid', gridx, gridy, gridz)
            wind[1, idx_x, idx_y, idx_z] = k3d[0][0][0]; variance[1, idx_x, idx_y, idx_z] = ss3d[0][0][0]
            k3d, ss3d = OK3d_down.execute('grid', gridx, gridy, gridz)
            wind[2, idx_x, idx_y, idx_z] = k3d[0][0][0]; variance[2, idx_x, idx_y, idx_z] = ss3d[0][0][0]

    return wind, variance
