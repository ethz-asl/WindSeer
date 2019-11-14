import numpy as np
import torch
import math
from scipy.interpolate import RegularGridInterpolator as RGI


class UlogInterpolation(object):
    def __init__(self, grid):
        self.wx = [[[[] for k in range(grid['n_cells'])] for j in range(grid['n_cells'])] for i in range(grid['n_cells'])]
        self.wy = [[[[] for k in range(grid['n_cells'])] for j in range(grid['n_cells'])] for i in range(grid['n_cells'])]
        self.wz = [[[[] for k in range(grid['n_cells'])] for j in range(grid['n_cells'])] for i in range(grid['n_cells'])]

    def bin_log_data(self, wind_data, grid_dimensions):
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
        wx = self.wx
        wy = self.wy
        wz = self.wz

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

        wind = torch.zeros((3, grid_dimensions['n_cells'], grid_dimensions['n_cells'], grid_dimensions['n_cells']))
        variance = torch.zeros(
            (3, grid_dimensions['n_cells'], grid_dimensions['n_cells'], grid_dimensions['n_cells'])) * float(
            'nan')

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
        print('\tPercentage of cells with values: {:.2f}'.format(
            100 * counter / (grid_dimensions['n_cells'] * grid_dimensions['n_cells'] * grid_dimensions['n_cells'])))
        print('\tNumber of values per cell (avg): {:.2f}'.format(np.mean(vals_per_cell)))
        return wind, variance

    def interpolate_log_data_krigging(self, wind_data, grid, terrain, OK3d_north, OK3d_east, OK3d_down):
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

    def interpolate_log_data_idw(self, wind_data, grid, terrain):
        # Initialize wind lists for the wind data
        wx = [[[[] for k in range(grid['n_cells'])] for j in range(grid['n_cells'])] for i in range(grid['n_cells'])]
        wy = [[[[] for k in range(grid['n_cells'])] for j in range(grid['n_cells'])] for i in range(grid['n_cells'])]
        wz = [[[[] for k in range(grid['n_cells'])] for j in range(grid['n_cells'])] for i in range(grid['n_cells'])]

        # Initialize distance lists for the wind_data
        id_xyz = [[[[] for k in range(grid['n_cells'])] for j in range(grid['n_cells'])] for i in range(grid['n_cells'])]

        # Determine the resolution of the grid
        x_res = (grid['x_max'] - grid['x_min']) / grid['n_cells']
        y_res = (grid['y_max'] - grid['y_min']) / grid['n_cells']
        z_res = (grid['z_max'] - grid['z_min']) / grid['n_cells']

        # Loop over the data and bin it
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

    def interpolate_log_data_from_grid(self, terrain, predicted_wind_data, wind_data):
        '''
        The output of the NN is taken as input and an interpolated value of the
        wind is calculated based at location along the trajectory
        '''

        interpolating_function_x = RGI((terrain.x_terr, terrain.y_terr, terrain.z_terr),
                                       predicted_wind_data[0, :, :, :].detach().cpu().numpy())
        interpolating_function_y = RGI((terrain.x_terr, terrain.y_terr, terrain.z_terr),
                                       predicted_wind_data[1, :, :, :].detach().cpu().numpy())
        interpolating_function_z = RGI((terrain.x_terr, terrain.y_terr, terrain.z_terr),
                                       predicted_wind_data[2, :, :, :].detach().cpu().numpy())

        # Initialize empty list of points where the wind is interpolated
        pts = [[] for i in range(len(wind_data['x']))]
        for i in range(len(wind_data['x'])):
            if ((wind_data['x'][i] > terrain.x_terr[0]) and
                    (wind_data['x'][i] < terrain.x_terr[-1]) and
                    (wind_data['y'][i] > terrain.y_terr[0]) and
                    (wind_data['y'][i] < terrain.y_terr[-1]) and
                    (wind_data['alt'][i] > terrain.z_terr[0]) and
                    (wind_data['alt'][i] < terrain.z_terr[-1])):
                pts[i].append([wind_data['x'][i], wind_data['y'][i], wind_data['alt'][i]])

        # Remove empty lists from pts
        pts = [x for x in pts if x != []]

        interpolated_log_data_x = interpolating_function_x(pts)
        interpolated_log_data_y = interpolating_function_y(pts)
        interpolated_log_data_z = interpolating_function_z(pts)
        return interpolated_log_data_x, interpolated_log_data_y, interpolated_log_data_z
