import numpy as np
import torch
import math
from pykrige.ok3d import OrdinaryKriging3D
from scipy.interpolate import RegularGridInterpolator as RGI
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import time


class FlightInterpolation:
    def __init__(self, wind_data, grid_dimensions=None, terrain=None, predict=False, wind_data_for_prediction=None):
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._wind_data = wind_data
        self._grid_dimensions = grid_dimensions
        self._terrain = terrain
        self.predict = predict
        self._wind_data_for_prediction = wind_data_for_prediction
        self._x_res, self._y_res, self._z_res = self.get_grid_resolution()
        self._idx_x, self._idx_y, self._idx_z, \
            self._wx, self._wy, self._wz = self.get_bin_indices_and_winds()
        self._bin_x_coord, self._bin_y_coord, self._bin_z_coord = self.get_bin_coordinates()

    def get_grid_resolution(self):
        x_res, y_res, z_res = 0, 0, 0
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
        x_coord, y_coord, z_coord = [], [], []
        if self._terrain is not None:
            for i in range(len(self._idx_x)):
                x_coord.append(self._terrain.x_terr[self._idx_x[i]])
                y_coord.append(self._terrain.y_terr[self._idx_y[i]])
                z_coord.append(self._terrain.z_terr[self._idx_z[i]])
        return x_coord, y_coord, z_coord

    def bin_flight_data(self):
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
                            self._grid_dimensions['n_cells'])) * float('nan')
        variance = torch.zeros((3,
                                self._grid_dimensions['n_cells'],
                                self._grid_dimensions['n_cells'],
                                self._grid_dimensions['n_cells'])) * float('nan')
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

    def interpolate_flight_data_idw(self):
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
                            self._grid_dimensions['n_cells'])) * float('nan')
        variance = torch.zeros((3,
                                self._grid_dimensions['n_cells'],
                                self._grid_dimensions['n_cells'],
                                self._grid_dimensions['n_cells'])) * float('nan')
        counter = 0
        vals_per_cell = []
        for i in range(self._grid_dimensions['n_cells']):
            for j in range(self._grid_dimensions['n_cells']):
                for k in range(self._grid_dimensions['n_cells']):
                    if self._wx[i][j][k]:
                        wind[0, i, j, k] = np.average(self._wx[i][j][k], axis=None, weights=id_xyz[i][j][k])
                        wind[1, i, j, k] = np.average(self._wy[i][j][k], axis=None, weights=id_xyz[i][j][k])
                        wind[2, i, j, k] = np.average(self._wz[i][j][k], axis=None, weights=id_xyz[i][j][k])

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

    def interpolate_flight_data_krigging(self):
        '''
        Create a wind map from the wind measurements using krigging interpolation.
        Compute the mean velocity and variance at the center of each bin by
        evaluating the wind map.

        The return consists of three tensors:
        @out [wind]: (3,n,n,n) tensor containing the mean velocities of each cell
        @out [variance]: (3,n,n,n) tensor containing the velocity variance of each cell

        (Requires installation of pykrige library)
        '''
        t_start = time.time()
        # Create the ordinary kriging object
        OK3d_north = OrdinaryKriging3D(self._wind_data['alt'], self._wind_data['y'], self._wind_data['x'],
                                       self._wind_data['wn'], variogram_model='linear')
        OK3d_east = OrdinaryKriging3D(self._wind_data['alt'], self._wind_data['y'], self._wind_data['x'],
                                      self._wind_data['we'], variogram_model='linear')
        OK3d_down = OrdinaryKriging3D(self._wind_data['alt'], self._wind_data['y'], self._wind_data['x'],
                                      -self._wind_data['wd'], variogram_model='linear')

        # Initialize empty wind, variance and predicted_flight_data
        if self._grid_dimensions is not None:
            wind = torch.zeros((3,
                                self._grid_dimensions['n_cells'],
                                self._grid_dimensions['n_cells'],
                                self._grid_dimensions['n_cells'])) * float('nan')
            variance = torch.zeros((3,
                                    self._grid_dimensions['n_cells'],
                                    self._grid_dimensions['n_cells'],
                                    self._grid_dimensions['n_cells'])) * float('nan')
        else:
            wind, variance = [], []
        predicted_flight_data = []

        if self.predict is True and self._wind_data_for_prediction:  # predict flight data
            # Make sure points are inside terrain
            pts = []
            for i in range(len(self._wind_data_for_prediction['x'])):
                if ((self._wind_data_for_prediction['x'][i] > self._terrain.x_terr[0]) and
                        (self._wind_data_for_prediction['x'][i] < self._terrain.x_terr[-1]) and
                        (self._wind_data_for_prediction['y'][i] > self._terrain.y_terr[0]) and
                        (self._wind_data_for_prediction['y'][i] < self._terrain.y_terr[-1]) and
                        (self._wind_data_for_prediction['alt'][i] > self._terrain.z_terr[0]) and
                        (self._wind_data_for_prediction['alt'][i] < self._terrain.z_terr[-1])):
                    pts.append([self._wind_data_for_prediction['alt'][i], self._wind_data_for_prediction['y'][i],
                                self._wind_data_for_prediction['x'][i]])
            if not pts:
                predicted_flight_data = [pts, pts, pts]
            else:
                predicted_flight_data_x, _ = OK3d_north.execute(
                    'points', np.array(pts)[:, 0], np.array(pts)[:, 1], np.array(pts)[:, 2])
                predicted_flight_data_y, _ = OK3d_north.execute(
                    'points', np.array(pts)[:, 0], np.array(pts)[:, 1], np.array(pts)[:, 2])
                predicted_flight_data_z, _ = OK3d_north.execute(
                    'points', np.array(pts)[:, 0], np.array(pts)[:, 1], np.array(pts)[:, 2])
                predicted_flight_data = [predicted_flight_data_x, predicted_flight_data_y, -predicted_flight_data_z]
                predicted_flight_data = torch.from_numpy(np.column_stack(predicted_flight_data))

        else:  # Bin data
            for i in range(len(self._bin_x_coord)):
                # x
                k3d, ss3d = OK3d_north.execute('grid', self._bin_z_coord[i], self._bin_y_coord[i], self._bin_x_coord[i])
                wind[0, self._idx_z[i], self._idx_y[i], self._idx_x[i]] = k3d[0][0][0]
                variance[0, self._idx_z[i], self._idx_y[i], self._idx_x[i]] = ss3d[0][0][0]
                # y
                k3d, ss3d = OK3d_east.execute('grid', self._bin_z_coord[i], self._bin_y_coord[i], self._bin_x_coord[i])
                wind[1, self._idx_z[i], self._idx_y[i], self._idx_x[i]] = k3d[0][0][0]
                variance[1, self._idx_z[i], self._idx_y[i], self._idx_x[i]] = ss3d[0][0][0]
                # z
                k3d, ss3d = OK3d_down.execute('grid', self._bin_z_coord[i], self._bin_y_coord[i], self._bin_x_coord[i])
                wind[2, self._idx_z[i], self._idx_y[i], self._idx_x[i]] = k3d[0][0][0]
                variance[2, self._idx_z[i], self._idx_y[i], self._idx_x[i]] = ss3d[0][0][0]

        print('Krigging interpolation is done [{:.2f} s]'.format(time.time() - t_start))
        return wind, variance, predicted_flight_data

    def interpolate_flight_data_gpr(self):
        '''
        Create a wind map from the wind measurements using Gaussian Process Regression
        for interpolation.
        Compute the mean velocity and variance at the center of each bin by
        evaluating the wind map.

        The return consists of three tensors:
        @out [wind]: (3,n,n,n) tensor containing the mean velocities of each cell
        @out [variance]: (3,n,n,n) tensor containing the velocity variance of each cell
        '''
        t_start = time.time()
        # Instantiate a Gaussian Process model for each direction
        kernel = C(1.0, (1e-3, 1e3)) * RBF(100, (1e-3, 1e3))
        gp_x = GPR(kernel=kernel, n_restarts_optimizer=10, alpha=0.2, normalize_y=True)
        gp_y = GPR(kernel=kernel, n_restarts_optimizer=10, alpha=0.2, normalize_y=True)
        gp_z = GPR(kernel=kernel, n_restarts_optimizer=10, alpha=0.2, normalize_y=True)

        # Fit to data using Maximum Likelihood Estimation of the parameters
        gp_x.fit(np.column_stack([self._wind_data['alt'], self._wind_data['y'], self._wind_data['x']]),
                 self._wind_data['wn'])
        gp_y.fit(np.column_stack([self._wind_data['alt'], self._wind_data['y'], self._wind_data['x']]),
                 self._wind_data['we'])
        gp_z.fit(np.column_stack([self._wind_data['alt'], self._wind_data['y'], self._wind_data['x']]),
                 -self._wind_data['wd'])
        # gp_x.kernel_.get_params()  # get kernel's hyperparameters

        # Initialize empty wind, variance, predicted_flight_data and predicted_wind_field
        if self._grid_dimensions is not None:
            wind = torch.zeros((3,
                                self._grid_dimensions['n_cells'],
                                self._grid_dimensions['n_cells'],
                                self._grid_dimensions['n_cells'])) * float('nan')
            variance = torch.zeros((3,
                                    self._grid_dimensions['n_cells'],
                                    self._grid_dimensions['n_cells'],
                                    self._grid_dimensions['n_cells'])) * float('nan')
        else:
            wind, variance = [], []
        predicted_flight_data = []
        predicted_wind_field = []

        if self.predict is True and self._wind_data_for_prediction:  # predict flight data
            # trajectory prediction
            # Make sure points are inside terrain
            pts = []
            for i in range(len(self._wind_data_for_prediction['x'])):
                if ((self._wind_data_for_prediction['x'][i] > self._terrain.x_terr[0]) and
                        (self._wind_data_for_prediction['x'][i] < self._terrain.x_terr[-1]) and
                        (self._wind_data_for_prediction['y'][i] > self._terrain.y_terr[0]) and
                        (self._wind_data_for_prediction['y'][i] < self._terrain.y_terr[-1]) and
                        (self._wind_data_for_prediction['alt'][i] > self._terrain.z_terr[0]) and
                        (self._wind_data_for_prediction['alt'][i] < self._terrain.z_terr[-1])):
                    pts.append([self._wind_data_for_prediction['alt'][i], self._wind_data_for_prediction['y'][i],
                                self._wind_data_for_prediction['x'][i]])
            if not pts:
                predicted_flight_data = [pts, pts, pts]
            else:
                predicted_flight_data_x = gp_x.predict(np.row_stack(pts))
                predicted_flight_data_y = gp_y.predict(np.row_stack(pts))
                predicted_flight_data_z = gp_z.predict(np.row_stack(pts))
                predicted_flight_data = [predicted_flight_data_x, predicted_flight_data_y, -predicted_flight_data_z]
                predicted_flight_data = torch.from_numpy(np.column_stack(predicted_flight_data))

            # wind field prediction
            x_terr, y_terr, z_terr = self._terrain.x_terr, self._terrain.y_terr, self._terrain.z_terr
            nx, ny, nz = x_terr.size, y_terr.size, z_terr.size
            xv_terr, yv_terr, zv_terr = np.meshgrid(x_terr, y_terr, z_terr)
            pts_wind_field = np.column_stack([xv_terr.flatten(), xv_terr.flatten(), xv_terr.flatten()])
            predicted_wind_field_x = (gp_x.predict(np.row_stack(pts_wind_field))).reshape((nz, ny, nx))
            predicted_wind_field_y = (gp_y.predict(np.row_stack(pts_wind_field))).reshape((nz, ny, nx))
            predicted_wind_field_z = (gp_z.predict(np.row_stack(pts_wind_field))).reshape((nz, ny, nx))
            predicted_wind_field_x = np.expand_dims(predicted_wind_field_x, axis=0)
            predicted_wind_field_y = np.expand_dims(predicted_wind_field_y, axis=0)
            predicted_wind_field_z = np.expand_dims(predicted_wind_field_z, axis=0)
            predicted_wind_field = np.concatenate(
                (predicted_wind_field_x, predicted_wind_field_y, predicted_wind_field_z), axis=0)
            # Use terrain mask on gpr predicted wind field
            is_wind = self._terrain.network_terrain.sign_()
            predicted_wind_field = is_wind.repeat(predicted_wind_field.shape[0], 1, 1, 1) \
                                   * torch.from_numpy(predicted_wind_field).to(self._device)

        else:  # bin data
            for i in range(len(self._bin_x_coord)):
                # x
                mean_x, var_x = gp_x.predict(np.column_stack(
                    [self._bin_z_coord[i], self._bin_y_coord[i], self._bin_x_coord[i]]),
                                             return_std=True)
                wind[0, self._idx_z[i], self._idx_y[i], self._idx_x[i]] = mean_x.item()
                variance[0, self._idx_z[i], self._idx_y[i], self._idx_x[i]] = var_x.item()
                # y
                mean_y, var_y = gp_y.predict(np.column_stack(
                    [self._bin_z_coord[i], self._bin_y_coord[i], self._bin_x_coord[i]]),
                                             return_std=True)
                wind[1, self._idx_z[i], self._idx_y[i], self._idx_x[i]] = mean_y.item()
                variance[1, self._idx_z[i], self._idx_y[i], self._idx_x[i]] = var_y.item()
                # z
                mean_z, var_z = gp_z.predict(np.column_stack(
                    [self._bin_z_coord[i], self._bin_y_coord[i], self._bin_x_coord[i]]),
                                             return_std=True)
                wind[2, self._idx_z[i], self._idx_y[i], self._idx_x[i]] = mean_z.item()
                variance[2, self._idx_z[i], self._idx_y[i], self._idx_x[i]] = var_z.item()

        print('GPR interpolation is done [{:.2f} s]'.format(time.time() - t_start))
        return wind, variance, predicted_flight_data, predicted_wind_field

    def interpolate_flight_data_from_grid(self, inferred_wind_data):
        '''
        Use inferred wind data from the output of the NN to interpolate the wind
        values at the points along the trajectory which are used for testing.
        '''

        interpolating_function_x = RGI((self._terrain.z_terr, self._terrain.y_terr, self._terrain.x_terr),
                                       inferred_wind_data[0, :].detach().cpu().numpy(), method='linear')
        interpolating_function_y = RGI((self._terrain.z_terr, self._terrain.y_terr, self._terrain.x_terr),
                                       inferred_wind_data[1, :].detach().cpu().numpy(), method='linear')
        interpolating_function_z = RGI((self._terrain.z_terr, self._terrain.y_terr, self._terrain.x_terr),
                                       inferred_wind_data[2, :].detach().cpu().numpy(), method='linear')

        # Initialize empty list of points where the wind is interpolated
        pts = []
        for i in range(len(self._wind_data_for_prediction['x'])):
            if ((self._wind_data_for_prediction['x'][i] > self._terrain.x_terr[0]) and
                    (self._wind_data_for_prediction['x'][i] < self._terrain.x_terr[-1]) and
                    (self._wind_data_for_prediction['y'][i] > self._terrain.y_terr[0]) and
                    (self._wind_data_for_prediction['y'][i] < self._terrain.y_terr[-1]) and
                    (self._wind_data_for_prediction['alt'][i] > self._terrain.z_terr[0]) and
                    (self._wind_data_for_prediction['alt'][i] < self._terrain.z_terr[-1])):
                pts.append([self._wind_data_for_prediction['alt'][i], self._wind_data_for_prediction['y'][i],
                            self._wind_data_for_prediction['x'][i]])

        interpolated_flight_data_x = interpolating_function_x(pts)
        interpolated_flight_data_y = interpolating_function_y(pts)
        interpolated_flight_data_z = interpolating_function_z(pts)
        interpolated_flight_data = [interpolated_flight_data_x, interpolated_flight_data_y, -interpolated_flight_data_z]
        interpolated_flight_data = torch.from_numpy(np.column_stack(interpolated_flight_data))
        return interpolated_flight_data
