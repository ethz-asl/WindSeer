import numpy as np
import torch
import math
from pykrige.ok3d import OrdinaryKriging3D
from scipy.interpolate import RegularGridInterpolator as RGI


class UlogInterpolation:
    def __init__(self, wind_data, grid_dimensions=None, terrain=None):
        self._terrain = terrain
        self._wind_data = wind_data
        self._grid_dimensions = grid_dimensions
        self._x_res, self._y_res, self._z_res = self.get_grid_resolution()
        self._idx_x, self._idx_y, self._idx_z, \
        self._wx, self._wy, self._wz = self.get_bin_indices_and_winds()
        self._x_coord, self._y_coord, self._z_coord = self.get_bin_coordinates()

    def get_grid_resolution(self):
        x_res = 0; y_res = 0; z_res = 0
        if self._grid_dimensions is not None:
            x_res = (self._grid_dimensions['x_max'] - self._grid_dimensions['x_min']) / self._grid_dimensions['n_cells']
            y_res = (self._grid_dimensions['y_max'] - self._grid_dimensions['y_min']) / self._grid_dimensions['n_cells']
            z_res = (self._grid_dimensions['z_max'] - self._grid_dimensions['z_min']) / self._grid_dimensions['n_cells']
        return x_res, y_res, z_res

    def get_bin_indices_and_winds(self):
        '''
        Bin the grid specified by the input grid dimensions.
        Get bins which contain the points along the trajectory specified by wind_data.
        Get the bin indices and wind speeds of the points inside each bin.

        Currently assumes that in all 3 dimensions the number of cells is equal.
        '''
        # Indices of bins which contain the points along the trajectory
        idx_x = []; idx_y = []; idx_z = []
        # Initialize empty list if grid dimensions is not provided
        wx = []; wy = []; wz = []

        if self._grid_dimensions is not None:
            # Initialize lists for binning the wind data
            wx = [[[[] for k in range(self._grid_dimensions['n_cells'])]
                   for j in range(self._grid_dimensions['n_cells'])]
                  for i in range(self._grid_dimensions['n_cells'])]
            wy = [[[[] for k in range(self._grid_dimensions['n_cells'])]
                   for j in range(self._grid_dimensions['n_cells'])]
                  for i in range(self._grid_dimensions['n_cells'])]
            wz = [[[[] for k in range(self._grid_dimensions['n_cells'])]
                   for j in range(self._grid_dimensions['n_cells'])]
                  for i in range(self._grid_dimensions['n_cells'])]

            for i in range(len(self._wind_data['x'])):
                if ((self._wind_data['x'][i] > self._grid_dimensions['x_min']) and
                        (self._wind_data['x'][i] < self._grid_dimensions['x_max']) and
                        (self._wind_data['y'][i] > self._grid_dimensions['y_min']) and
                        (self._wind_data['y'][i] < self._grid_dimensions['y_max']) and
                        (self._wind_data['alt'][i] > self._grid_dimensions['z_min']) and
                        (self._wind_data['alt'][i] < self._grid_dimensions['z_max'])):
                    id_x = (int((self._wind_data['x'][i] - self._grid_dimensions['x_min']) / self._x_res))
                    id_y = (int((self._wind_data['y'][i] - self._grid_dimensions['y_min']) / self._y_res))
                    id_z = (int((self._wind_data['alt'][i] - self._grid_dimensions['z_min']) / self._z_res))

                    wx[id_z][id_y][id_x].append(self._wind_data['wn'][i])
                    wy[id_z][id_y][id_x].append(self._wind_data['we'][i])
                    wz[id_z][id_y][id_x].append(-self._wind_data['wd'][i])
                    idx_x.append(id_x)
                    idx_y.append(id_y)
                    idx_z.append(id_z)
        return idx_x, idx_y, idx_z, wx, wy, wz

    def get_bin_coordinates(self):
        x_coord = []; y_coord =[]; z_coord = []
        if self._terrain is not None:
            for i in range(len(self._idx_x)):
                x_coord.append(self._terrain.x_terr[self._idx_x[i]])
                y_coord.append(self._terrain.y_terr[self._idx_y[i]])
                z_coord.append(self._terrain.z_terr[self._idx_z[i]])
        return x_coord, y_coord, z_coord

    def bin_log_data(self):
        '''
        Compute the mean velocity and variance of wind values at the center of
        each bin.

        The return consists of three tensors:
        @out [wind]: (3,n,n,n) tensor containing the mean velocities of each cell
        @out [variance]: (3,n,n,n) tensor containing the velocity variance of each cell
        '''
        # Initialize wind and variance
        wind = torch.zeros((3,
                            self._grid_dimensions['n_cells'],
                            self._grid_dimensions['n_cells'],
                            self._grid_dimensions['n_cells']))
        variance = torch.zeros((3,
                                self._grid_dimensions['n_cells'],
                                self._grid_dimensions['n_cells'],
                                self._grid_dimensions['n_cells']))
        counter = 0
        vals_per_cell = []
        for i in range(self._grid_dimensions['n_cells']):
            for j in range(self._grid_dimensions['n_cells']):
                for k in range(self._grid_dimensions['n_cells']):
                    if self._wx[i][j][k]:
                        wind[0, i, j, k] = np.mean(self._wx[i][j][k])
                        wind[1, i, j, k] = np.mean(self._wy[i][j][k])
                        wind[2, i, j, k] = np.mean(self._wz[i][j][k])

                        variance[0, i, j, k] = np.var(self._wx[i][j][k])
                        variance[1, i, j, k] = np.var(self._wy[i][j][k])
                        variance[2, i, j, k] = np.var(self._wz[i][j][k])

                        counter += 1
                        vals_per_cell.append(len(self._wx[i][j][k]))
        print('')
        print('\tNumber of cells with values:     {}'.format(counter))
        print('\tPercentage of cells with values: {:.2f}'.format(
            100 * counter / (self._grid_dimensions['n_cells']
                             * self._grid_dimensions['n_cells']
                             * self._grid_dimensions['n_cells'])))
        print('\tNumber of values per cell (avg): {:.2f}'.format(np.mean(vals_per_cell)))
        return wind, variance

    def interpolate_log_data_krigging(self):
        '''
        Create a wind map from the wind measurements using krigging interpolation
        (or Gaussian regression process).
        Compute the mean velocity and variance at the center of each bin by
        evaluating the wind map.

        The return consists of three tensors:
        @out [wind]: (3,n,n,n) tensor containing the mean velocities of each cell
        @out [variance]: (3,n,n,n) tensor containing the velocity variance of each cell
        '''

        # Create the ordinary kriging object
        OK3d_north = OrdinaryKriging3D(self._wind_data['x'], self._wind_data['y'], self._wind_data['alt'],
                                      self._wind_data['wn'], variogram_model='linear')
        OK3d_east = OrdinaryKriging3D(self._wind_data['x'], self._wind_data['y'], self._wind_data['alt'],
                                      self._wind_data['we'], variogram_model='linear')
        OK3d_down = OrdinaryKriging3D(self._wind_data['x'], self._wind_data['y'], self._wind_data['alt'],
                                      self._wind_data['wd'], variogram_model='linear')

        # Initialize empty wind and variance
        wind = torch.zeros(
            (3, self._grid_dimensions['n_cells'], self._grid_dimensions['n_cells'], self._grid_dimensions['n_cells']))
        variance = torch.zeros(
            (3, self._grid_dimensions['n_cells'], self._grid_dimensions['n_cells'], self._grid_dimensions['n_cells']))

        # Loop over the data and create the krigged grid and the variance grid
        for i in range(len(self._x_coord)):
            # North
            k3d, ss3d = OK3d_north.execute('grid', self._x_coord[i], self._y_coord[i], self._z_coord[i])
            wind[0, self._idx_x[i], self._idx_y[i], self._idx_z[i]] = k3d[0][0][0]
            variance[0, self._idx_x[i], self._idx_y[i], self._idx_z[i]] = ss3d[0][0][0]
            # East
            k3d, ss3d = OK3d_east.execute('grid', self._x_coord[i], self._y_coord[i], self._z_coord[i])
            wind[1, self._idx_x[i], self._idx_y[i], self._idx_z[i]] = k3d[0][0][0]
            variance[1, self._idx_x[i], self._idx_y[i], self._idx_z[i]] = ss3d[0][0][0]
            # Down
            k3d, ss3d = OK3d_down.execute('grid', self._x_coord[i], self._y_coord[i], self._z_coord[i])
            wind[2, self._idx_x[i], self._idx_y[i], self._idx_z[i]] = k3d[0][0][0]
            variance[2, self._idx_x[i], self._idx_y[i], self._idx_z[i]] = ss3d[0][0][0]

        return wind, variance

    def interpolate_log_data_idw(self):
        '''
        Compute the wind value at the center of each bin using an inverse distance
        weighted interpolation scheme.

        The return consists of three tensors:
        @out [wind]: (3,n,n,n) tensor containing the mean velocities of each cell
        @out [variance]: (3,n,n,n) tensor containing the velocity variance of each cell
        '''
        # Initialize distance lists for the wind_data
        id_xyz = [[[[] for k in range(self._grid_dimensions['n_cells'])]
                   for j in range(self._grid_dimensions['n_cells'])]
                  for i in range(self._grid_dimensions['n_cells'])]

        # Calculate distance of points to the center of the bin
        for i in range(len(self._wind_data['x'])):
            if ((self._wind_data['x'][i] > self._grid_dimensions['x_min']) and
                    (self._wind_data['x'][i] < self._grid_dimensions['x_max']) and
                    (self._wind_data['y'][i] > self._grid_dimensions['y_min']) and
                    (self._wind_data['y'][i] < self._grid_dimensions['y_max']) and
                    (self._wind_data['alt'][i] > self._grid_dimensions['z_min']) and
                    (self._wind_data['alt'][i] < self._grid_dimensions['z_max'])):
                idx_x = int((self._wind_data['x'][i] - self._grid_dimensions['x_min']) / self._x_res)
                idx_y = int((self._wind_data['y'][i] - self._grid_dimensions['y_min']) / self._y_res)
                idx_z = int((self._wind_data['alt'][i] - self._grid_dimensions['z_min']) / self._z_res)

                distance = math.sqrt((self._wind_data['x'][i] - self._terrain.x_terr[idx_x]) ** 2
                                     + (self._wind_data['y'][i] - self._terrain.y_terr[idx_y]) ** 2
                                     + (self._wind_data['alt'][i] - self._terrain.z_terr[idx_z]) ** 2)
                id_xyz[idx_z][idx_y][idx_x].append(1 / distance)

        wind = torch.zeros((3,
                            self._grid_dimensions['n_cells'],
                            self._grid_dimensions['n_cells'],
                            self._grid_dimensions['n_cells']))
        variance = torch.zeros((3,
                                self._grid_dimensions['n_cells'],
                                self._grid_dimensions['n_cells'],
                                self._grid_dimensions['n_cells']))
        for i in range(self._grid_dimensions['n_cells']):
            for j in range(self._grid_dimensions['n_cells']):
                for k in range(self._grid_dimensions['n_cells']):
                    if self._wx[i][j][k]:
                        wind[0, i, j, k] = np.average(self._wx[i][j][k], axis=None, weights=id_xyz[i][j][k])
                        wind[1, i, j, k] = np.average(self._wy[i][j][k], axis=None, weights=id_xyz[i][j][k])
                        wind[2, i, j, k] = np.average(self._wz[i][j][k], axis=None, weights=id_xyz[i][j][k])

        return wind, variance

    def interpolate_log_data_from_grid(self, inferred_wind_data, predict_wind_data):
        '''
        Use inferred wind data from the output of the NN to interpolate the wind
        values at the points along the trajectory which are used for testing.
        '''

        interpolating_function_x = RGI((self._terrain.x_terr, self._terrain.y_terr, self._terrain.z_terr),
                                       inferred_wind_data[0, :, :, :].detach().cpu().numpy())
        interpolating_function_y = RGI((self._terrain.x_terr, self._terrain.y_terr, self._terrain.z_terr),
                                       inferred_wind_data[1, :, :, :].detach().cpu().numpy())
        interpolating_function_z = RGI((self._terrain.x_terr, self._terrain.y_terr, self._terrain.z_terr),
                                       inferred_wind_data[2, :, :, :].detach().cpu().numpy())

        # Initialize empty list of points where the wind is interpolated
        pts = [[] for i in range(len(predict_wind_data['x']))]
        for i in range(len(predict_wind_data['x'])):
            if ((predict_wind_data['x'][i] > self._terrain.x_terr[0]) and
                    (predict_wind_data['x'][i] < self._terrain.x_terr[-1]) and
                    (predict_wind_data['y'][i] > self._terrain.y_terr[0]) and
                    (predict_wind_data['y'][i] < self._terrain.y_terr[-1]) and
                    (predict_wind_data['alt'][i] > self._terrain.z_terr[0]) and
                    (predict_wind_data['alt'][i] < self._terrain.z_terr[-1])):
                pts[i].append([predict_wind_data['x'][i], predict_wind_data['y'][i], predict_wind_data['alt'][i]])

        # Remove empty lists from pts
        pts = [x for x in pts if x != []]

        interpolated_log_data_x = interpolating_function_x(pts)
        interpolated_log_data_y = interpolating_function_y(pts)
        interpolated_log_data_z = interpolating_function_z(pts)
        interpolated_log_data = [interpolated_log_data_x, interpolated_log_data_y, interpolated_log_data_z]
        return interpolated_log_data
