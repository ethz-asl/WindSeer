import numpy as np
import nn_wind_prediction.utils as utils
import nn_wind_prediction.models as models
from analysis_utils import extract_cosmo_data as cosmo
from analysis_utils import ulog_utils, get_mapgeo_terrain
from analysis_utils.interpolate_log_data import UlogInterpolation
from analysis_utils.generate_trajectory import generate_trajectory
from nn_wind_prediction.utils.interpolation import DataInterpolation
import nn_wind_prediction.data as data
import nn_wind_prediction.data as nn_data
from sklearn import metrics
from datetime import datetime
from scipy import ndimage
from scipy.interpolate import CubicSpline
import random
import torch
import os
import time


class TerrainBlock(object):
    def __init__(self, x_terr, y_terr, z_terr, h_terr, full_block, device=None, boolean_terrain=False):
        self.x_terr = x_terr
        self.y_terr = y_terr
        self.z_terr = z_terr
        self.h_terr = h_terr
        self.is_wind = np.invert(full_block)
        self.terrain_corners = h_terr[::h_terr.shape[0] - 1, ::h_terr.shape[1] - 1]
        self.boolean_terrain = boolean_terrain
        if self.boolean_terrain:
            network_terrain = torch.from_numpy(np.invert(self.is_wind))
        else:
            network_terrain = torch.from_numpy(ndimage.distance_transform_edt(self.is_wind).astype(np.float32))
        if device is None:
            self.network_terrain = network_terrain.unsqueeze(0)
        else:
            self.network_terrain = network_terrain.unsqueeze(0).to(device)

    def get_dimensions(self):
        return len(self.x_terr), len(self.y_terr), len(self.z_terr)


class SimpleStepOptimiser:
    def __init__(self, variables, lr=1e-4, lr_decay=0.0):
        self.var = variables
        self.learning_rate = lr
        self.lr_decay = lr_decay
        self.iterations = 0

    def __str__(self):
        return 'SimpleStepOptimiser, lr={0:0.2e}'.format(self.learning_rate)

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


class OptTest(object):
    def __init__(self, opt, kwargs={}):
        self.opt = opt
        self.kwargs = kwargs

    def __str__(self):
        return str(self.opt)


class SelectionFlags(object):
    def __init__(self, test, cfd, ulog, wind, noise, optimisation):
        self.test_simulated_data = test.params['test_simulated_data']
        self.test_flight_data = test.params['test_flight_data']
        self.batch_test = cfd.params['batch_test']
        self.use_ekf_wind = ulog.params['use_ekf_wind']
        self.add_wind_measurements = wind.params['add_wind_measurement']
        self.use_scattered_points = wind.params['scattered_points']['use_scattered_points']
        self.use_trajectory = wind.params['trajectory']['use_trajectory_generation']
        self.predict_ulog = wind.params['ulog_flight']['predict_ulog']
        self.generate_turbulence = noise.params['generate_turbulence']
        self.add_gaussian_noise = noise.params['add_gaussian_noise']
        self.optimize_corners_individually = optimisation.params['optimise_corners_individually']
        self.use_scale_optimisation = optimisation.params['use_scale_optimisation']
        self.use_spline_optimisation = optimisation.params['use_spline_optimisation']


class WindOptimiser(object):
    _loss_fn = torch.nn.MSELoss()
    _optimisation_variables = None

    def __init__(self, config_yaml, resolution=64):
        # Configuration variables
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._config_yaml = config_yaml
        self._test_args = utils.BasicParameters(self._config_yaml, 'test')
        self._cfd_args = utils.BasicParameters(self._config_yaml, 'cfd')
        self._cosmo_args = utils.COSMOParameters(self._config_yaml)
        self._ulog_args = utils.UlogParameters(self._config_yaml)
        self._wind_args = utils.BasicParameters(self._config_yaml, 'wind')
        self._noise_args = utils.BasicParameters(self._config_yaml, 'noise')
        self._optimisation_args = utils.BasicParameters(self._config_yaml, 'optimisation')
        self._model_args = utils.BasicParameters(self._config_yaml, 'model')
        self._resolution = resolution
        # Selection flags
        self.flag = SelectionFlags(self._test_args, self._cfd_args, self._ulog_args,
                                   self._wind_args, self._noise_args, self._optimisation_args)
        # Load data
        if self.flag.test_simulated_data:
            self.original_input, self.labels, self.data_set_name, self.grid_size, self.binary_terrain \
                = self.load_data_set()
            self.terrain = self.get_cfd_terrain()
        if self.flag.test_flight_data:
            self._ulog_data = self.load_ulog_data()
            self._cosmo_wind = self.load_wind()
            self.terrain = self.load_cosmo_terrain()
        # Wind measurements variables
        if self.flag.test_simulated_data:
            if self.flag.use_scattered_points:
                self._wind_blocks, self._wind_zeros, self._wind_mask \
                    = self.get_scattered_wind_blocks()
            if self.flag.use_trajectory:
                self._wind_blocks, self._wind_zeros, self._wind_mask\
                    = self.get_trajectory_wind_blocks()
        if self.flag.test_flight_data:
            self._wind_blocks, self._var_blocks, self._wind_zeros, self._wind_mask \
                = self.get_ulog_wind_blocks()
        # Noise
        if self.flag.add_gaussian_noise:
            self._wind_blocks, self._wind_zeros = self.add_gaussian_noise()
        if self.flag.generate_turbulence:
            self._wind_blocks, self._wind_zeros = self.generate_turbulence()
        # Optimisation variables
        self._optimisation_variables = self.get_optimisation_variables()
        self.reset_optimisation_variables()
        if self.flag.test_simulated_data:
            self._base_cfd_corners = self.get_cfd_corners()
            # self._optimized_corners = self.get_optimized_corners()
        if self.flag.test_flight_data:
            self._base_cosmo_corners = self.get_cosmo_corners()
            if self.flag.predict_ulog:
                self._train_ulog_data, self._test_ulog_data = self.train_test_split()
        self._interpolator = DataInterpolation(self._device, 3, *self.terrain.get_dimensions())
        # Network model and loss function
        self.net = self.load_network_model()
        self.net.freeze_model()
        self._loss_fn = self.get_loss_function()

    # --- Load data ---

    def load_ulog_data(self):
        print('Loading ulog data...', end='', flush=True)
        t_start = time.time()
        self._ulog_args.print()
        ulog_data = ulog_utils.get_log_data(self._ulog_args.params['file'])

        if self.flag.use_ekf_wind:
            ulog_data['we'] = ulog_data['we_east']
            ulog_data['wn'] = ulog_data['we_north']
            ulog_data['wd'] = ulog_data['we_down']
        else:
            ulog_data['we'] = ulog_data['we']
            ulog_data['wn'] = ulog_data['wn']
            ulog_data['wd'] = ulog_data['wd']
        print(' done [{:.2f} s]'.format(time.time() - t_start))

        return ulog_data

    def load_wind(self):
        print('Loading COSMO wind...', end='', flush=True)
        t_start = time.time()
        self._cosmo_args.print()

        lat0, lon0 = self._ulog_data['lat'][0], self._ulog_data['lon'][0]

        # Get cosmo wind
        t0 = datetime.utcfromtimestamp(self._ulog_data['utc_microsec'][0] / 1e6)
        offset_cosmo_time = self._cosmo_args.get_cosmo_time(t0.hour)
        cosmo_wind = cosmo.extract_cosmo_data(self._cosmo_args.params['file'], lat0, lon0, offset_cosmo_time,
                                              terrain_file=self._cosmo_args.params['terrain_file'])
        print(' done [{:.2f} s]'.format(time.time() - t_start))
        return cosmo_wind

    def load_cosmo_terrain(self):
        print('Loading terrain...', end='', flush=True)
        t_start = time.time()
        # Get corresponding terrain
        # min_height = min(ulog_data['alt'].min(), h_terr.min())
        block_height = [1100.0 / 95 * 63]
        # x_terr, y_terr and z_terr are the (regular, monotonic) index arrays for the h_terr and full_block arrays
        # h_terr is the terrain height
        boolean_terrain = self._model_args.params['boolean_terrain']

        terrain = TerrainBlock(
            *get_mapgeo_terrain.get_terrain(self._cosmo_args.params['terrain_tiff'], self._cosmo_wind['x'][[0, 1], [0, 1]],
                               self._cosmo_wind['y'][[0, 1], [0, 1]],
                               block_height, (self._resolution, self._resolution, self._resolution)),
            device=self._device, boolean_terrain=boolean_terrain)
        print(' done [{:.2f} s]'.format(time.time() - t_start))
        return terrain

    def load_data_set(self):
        # Load data set
        test_set = nn_data.HDF5Dataset(self._cfd_args.params['testset_name'],
                                       self._cfd_args.params['input_channels'],
                                       self._cfd_args.params['label_channels'])

        # Get data set from test set
        data_set = test_set[self._cfd_args.params['index']]
        name = test_set.get_name(self._cfd_args.params['index'])
        print("Loading test set with name: ", str(name))
        grid_size = data.get_grid_size(self._cfd_args.params['testset_name'])
        # Convert distance transformed matrix back to binary matrix
        binary_terrain = data_set[0][0, :, :, :] <= 0
        return data_set[0], data_set[1], name, grid_size, binary_terrain

    def get_cfd_terrain(self):
        nx, ny, nz = self.binary_terrain.shape
        x_terr = torch.Tensor([self.grid_size[0]*i for i in range(nx)])
        y_terr = torch.Tensor([self.grid_size[1]*i for i in range(ny)])
        z_terr = torch.Tensor([self.grid_size[2]*i for i in range(nz)])
        h_terr = torch.Tensor(np.squeeze(self.binary_terrain.sum(axis=2, keepdims=True)*self.grid_size[2]))
        # Get terrain object
        boolean_terrain = self._model_args.params['boolean_terrain']
        terrain = TerrainBlock(x_terr, y_terr, z_terr, h_terr, self.binary_terrain,
                               device=self._device, boolean_terrain=boolean_terrain)
        return terrain

    # --- Wind generation ---

    def get_scattered_wind_blocks(self):
        # num_steps = self._wind_args.params['scattered_points']['num_steps']
        t = 0
        losses, percentages = [], []
        # Copy of the true wind labels
        wind = self.labels.clone().detach().cpu().numpy()
        # p = ((100 - self._wind_args.params['scattered_points']['initial_percentage'])/num_steps)
        # percentage = (self._wind_args.params['scattered_points']['initial_percentage'] + p) / 100
        percentage = self._wind_args.params['scattered_points']['initial_percentage'] / 100
        mask = self.binary_terrain.detach().cpu().numpy()
        # Change (percentage) of False values in binary terrain to True
        wind_mask = [not elem if (not elem and random.random() < percentage) else elem for elem in np.nditer(mask)]

        wind[wind_mask] = float('nan')
        wind_blocks = torch.Tensor(wind).to(self._device)
        wind[wind_mask] = 0
        wind_zeros = torch.Tensor(wind).to(self._device)
        return wind_blocks, wind_zeros, torch.Tensor(wind_mask).to(self._device)

    def get_trajectory_wind_blocks(self):
        x = self._wind_args.params['trajectory']['x']
        y = self._wind_args.params['trajectory']['y']
        z = self._wind_args.params['trajectory']['z']
        num_points = self._wind_args.params['trajectory']['num_points']
        x_traj, y_traj, z_traj = generate_trajectory(x, y, z, num_points, self.terrain)
        # Get the grid points and winds along the trajectory
        wind = torch.zeros(self.labels.shape) * float('nan')
        num_segments, segment_length = x_traj.shape
        counter = 0
        for i in range(num_segments):
            for j in range(segment_length):
                id_x = (int((x_traj[i][j] - self.terrain.x_terr[0]) / self.grid_size[0]))
                id_y = (int((y_traj[i][j] - self.terrain.y_terr[0]) / self.grid_size[1]))
                id_z = (int((z_traj[i][j] - self.terrain.z_terr[0]) / self.grid_size[2]))
                if all(v == 0 for v in wind[:, id_x, id_y, id_z]):
                    counter += 1
                wind[:, id_x, id_y, id_z] = self.labels[:, id_x, id_y, id_z]

        wind_mask = torch.isnan(wind)
        wind_zeros = wind.clone()
        wind_zeros[wind_mask] = 0
        return wind.to(self._device), wind_zeros.to(self._device), wind_mask.to(self._device)

    def get_ulog_wind_blocks(self):
        print('Getting binned wind blocks...', end='', flush=True)
        t_start = time.time()

        # Determine the grid dimensions
        dx = self.terrain.x_terr[[0, -1]]; ddx = (self.terrain.x_terr[1]-self.terrain.x_terr[0])/2.0
        dy = self.terrain.y_terr[[0, -1]]; ddy = (self.terrain.y_terr[1]-self.terrain.y_terr[0])/2.0
        dz = self.terrain.z_terr[[0, -1]]; ddz = (self.terrain.z_terr[1]-self.terrain.z_terr[0])/2.0
        grid_dimensions = {'x_min': dx[0] - ddx, 'x_max': dx[1] + ddx, 'y_min': dy[0] - ddy, 'y_max': dy[1] + ddy,
                           'z_min': dz[0] - ddz, 'z_max': dz[1] + ddz, 'n_cells': self.terrain.get_dimensions()[0]}
        if self._wind_args.params['ulog_flight']['predict_ulog']:
            UlogInterpolator = UlogInterpolation(self._train_ulog_data, grid_dimensions, self.terrain)
        else:
            UlogInterpolator = UlogInterpolation(self._ulog_data, grid_dimensions, self.terrain)

        # bin the data into the regular grid
        try:
            if self._wind_args.params['ulog_flight']['interpolation_method'].lower() == 'bin':
                wind, variance = UlogInterpolator.bin_log_data()
            elif self._wind_args.params['ulog_flight']['interpolation_method'].lower() == 'krigging':
                wind, variance = UlogInterpolator.interpolate_log_data_krigging()
            elif self._wind_args.params['ulog_flight']['interpolation_method'].lower() == 'idw':
                wind, variance = UlogInterpolator.interpolate_log_data_idw()
            else:
                print('Specified interpolation method: {0} unknown!'
                      .format(self._ulog_args.params['interpolation_method']))
                raise ValueError
        except KeyError:
            print('Interpolation method not specified in file: {0}'
                  .format(self._config_yaml))
        wind_mask = torch.isnan(wind)       # This is a binary mask with ones where there are invalid wind estimates
        wind_zeros = wind.clone()
        wind_zeros[wind_mask] = 0

        print(' done [{:.2f} s]'.format(time.time() - t_start))
        return wind, variance, wind_zeros.to(self._device), wind_mask.to(self._device)

    def run_original_wind_prediction(self, data_set):
        input_ = data_set[0].to(self._device)
        labels = data_set[1].to(self._device)
        output = self.get_wind_prediction(input_)
        loss = self._loss_fn(output, labels)
        print("Loss is: ", loss.item())

    # --- Add noise to data ---

    def add_gaussian_noise(self):
        # wind_noise = np.random.normal(0, 1, self._wind_zeros.shape)
        wind_noise = torch.randn(self._wind_zeros.shape).to(self._device)
        wind_zeros = self._wind_zeros.clone() + wind_noise
        wind_zeros[self._wind_mask] = 0
        wind_blocks = self._wind_blocks.clone() + wind_noise
        wind_blocks[self._wind_mask] = float('nan')
        return wind_blocks, wind_zeros

    def generate_turbulence(self):
        wind_blocks, wind_zeros = 0, 0
        return wind_blocks, wind_zeros

    # --- Wind optimisation ---

    def get_optimisation_variables(self):
        rotation = self._optimisation_args.params['rotation']
        if self.flag.use_scale_optimisation:
            scale = self._optimisation_args.params['scale']
        else:
            scale = []
        if self.flag.use_spline_optimisation:
            spline = 1
        else:
            spline = []
        opt_var = [rotation, scale, spline]
        optimisation_variables = [var for var in opt_var if var != []]
        # Replicate optimisation variables for each corner
        if self.flag.optimize_corners_individually:
            optimisation_variables = [optimisation_variables[:] for i in range(4)]

        return optimisation_variables

    def reset_optimisation_variables(self, optimisation_variables=None):
        if optimisation_variables is None:
            optimisation_variables = self._optimisation_variables

        self._optimisation_variables = torch.Tensor(optimisation_variables).to(self._device).requires_grad_()

    def get_optimized_corners(self):
        corner_winds = np.zeros((3, 64, 2, 2), dtype='float')
        wind_z = []
        num_points = self.optimisation_args.params['wind_profile']['polynomial_law_num_points']
        for yi in range(2):
            for xi in range(2):
                # wind_speed = self._optimisation_variables[i][0:num_points]
                wind_height_increment = ((self.terrain.z_terr[-1] + self.grid_size[2])
                                         - self.terrain.terrain_corners[yi, xi]) / num_points
                wind_z.append([self.terrain.terrain_corners[yi, xi] + j * wind_height_increment for j in range(num_points)])

        if self.optimisation_args.params['wind_profile']['optimized_corners'] > 0:
            for yi in range(2):
                for xi in range(2):
                    valid_z = ((self.terrain.z_terr+self.grid_size[2]) > self.terrain.terrain_corners[yi, xi])
                    if wind_z[xi*2+yi][1] > wind_z[xi*2+yi][0]:
                        corner_winds[0, valid_z, yi, xi] = \
                            CubicSpline(wind_z[xi*2+yi], self._optimisation_variables[xi*2+yi][0:num_points])(self.terrain.z_terr[valid_z])\
                            * 1/(np.sqrt(1+torch.tan(self._optimisation_variables[xi*2+yi][-1]**2)))
                        corner_winds[1, valid_z, yi, xi] = \
                            CubicSpline(wind_z[xi*2+yi], self._optimisation_variables[xi*2+yi][0:num_points])(self.terrain.z_terr[valid_z])\
                            * torch.tan(self._optimisation_variables[xi*2+yi][-1])/(np.sqrt(1+torch.tan(self._optimisation_variables[xi*2+yi][-1]**2)))

        else:
            for yi in range(2):
                for xi in range(2):
                    valid_z = ((self.terrain.z_terr+self.grid_size[2]) > self.terrain.h_terr[xi+yi])
                    if wind_z[xi*2+yi][1] > wind_z[xi*2+yi][0]:
                        corner_winds[0, valid_z, yi, xi] = \
                            CubicSpline(wind_z[xi+yi], self._optimisation_variables[0:num_points])(self.terrain.z_terr[valid_z])\
                            * 1/(np.sqrt(1+np.tan(self._optimisation_variables[-1]**2)))
                        corner_winds[1, valid_z, yi, xi] = \
                            CubicSpline(wind_z[xi+yi], self._optimisation_variables[0:num_points])(self.terrain.z_terr[valid_z])\
                            * np.tan(self._optimisation_variables[-1])/(np.sqrt(1+np.tan(self._optimisation_variables[-1]**2)))

        return torch.Tensor(corner_winds).to(self._device).requires_grad_()

    def get_cfd_corners(self):
        channels, nx, ny, nz = self.labels.shape
        corners = self.labels[:, ::nx-1, ::ny-1, :]
        cfd_corners = np.transpose(corners, (0, 3, 1, 2))
        return cfd_corners.to(self._device)

    def get_cosmo_corners(self):
        temp_cosmo = cosmo.cosmo_corner_wind(self._cosmo_wind, self.terrain.z_terr, rotate=0.0, scale=1.0,
                                             terrain_height=self.terrain.terrain_corners)
        cosmo_corners = torch.from_numpy(temp_cosmo.astype(np.float32)).to(self._device)
        return cosmo_corners

    def train_test_split(self):
        train_ulog = {}; test_ulog = {}
        if self._ulog_args.params['predict_ulog']:
            train_size = self._ulog_args.params['train_size']
            for keys, values in self._ulog_data.items():
                train_batch = values[:int(len(values) * train_size)]
                test_batch = values[int(len(values)*train_size):]
                train_ulog.update({keys: train_batch})
                test_ulog.update({keys: test_batch})

        return train_ulog, test_ulog

    # --- Network and loss function ---

    def load_network_model(self):
        print('Loading network model...', end='', flush=True)
        t_start = time.time()
        yaml_loc = os.path.join(self._model_args.params['location'], self._model_args.params['name'], 'params.yaml')
        params = utils.EDNNParameters(yaml_loc)  # load the model config
        NetworkType = getattr(models, params.model['model_type'])  # load the model
        net = NetworkType(**params.model_kwargs())  # load learnt parameters
        model_loc = os.path.join(self._model_args.params['location'], self._model_args.params['name'],
                                 self._model_args.params['version'] + '.model')
        net.load_state_dict(torch.load(model_loc, map_location=lambda storage, loc: storage))
        net = net.to(self._device)
        print(' done [{:.2f} s]'.format(time.time() - t_start))
        return net

    def get_loss_function(self):
        try:
            if self._model_args.params['loss'].lower() == 'l1':
                loss_fn = torch.nn.L1Loss()
            elif self._model_args.params['loss'].lower() == 'mse':
                loss_fn = torch.nn.MSELoss()
            else:
                print('Specified loss function: {0} unknown!'.format(self._model_args.params['loss']))
                raise ValueError
        except KeyError:
            print('Loss function not specified, using default: {0}'.format(str(self._loss_fn)))
        return loss_fn

    # --- Helper functions ---

    def get_rotated_wind(self):
        sr = []; cr = []
        for i in range(self._optimisation_variables.shape[0]):
            sr.append(torch.sin(self._optimisation_variables[i][0]))
            cr.append(torch.cos(self._optimisation_variables[i][0]))
        # Get corner winds for model inference, offset to actual terrain heights
        if self.flag.test_simulated_data:
            wind_corners = self._base_cfd_corners.clone()
            corners = self._base_cfd_corners
        if self.flag.test_flight_data:
            wind_corners = self._base_cosmo_corners.clone()
            corners = self._base_cosmo_corners
        if self.flag.use_scale_optimisation:
            for i in range(self._optimisation_variables.shape[0]):
                wind_corners[0, :, i//2, (i+2)%2] = self._optimisation_variables[i][1]*(
                        corners[0, :, i//2, (i+2)%2]*cr[i]
                        - corners[1, :, i//2, (i+2)%2]*sr[i])
                wind_corners[1, :, i//2, (i+2)%2] = self._optimisation_variables[i][1]*(
                        corners[0, :, i//2, (i+2)%2]*sr[i]
                        + corners[1, :, i//2, (i+2)%2]*cr[i])
        if self.flag.use_spline_optimisation:
            wind_corners = []
        return wind_corners

    def generate_wind_input(self):
        wind_corners = self.get_rotated_wind()
        interpolated_wind = self._interpolator.edge_interpolation(wind_corners)
        if self.flag.add_wind_measurements:
            interpolated_wind += self._wind_zeros
        input_ = torch.cat([self.terrain.network_terrain, interpolated_wind])
        return input_

    def generate_ulog_input(self):
        wind = self._wind_zeros.to(self._device)
        input = torch.cat([self.terrain.network_terrain, wind])
        return input

    def get_predicted_interpolated_ulog_data(self, output_wind):
        grid_dimensions = None
        UlogInterpolator = UlogInterpolation(self._train_ulog_data, grid_dimensions, self.terrain)
        interpolated_ulog_data = UlogInterpolator.interpolate_log_data_from_grid(output_wind, self._test_ulog_data)
        return interpolated_ulog_data

    def calculate_metrics(self, predicted_wind):
        test_wind_inside_terrain = np.zeros((3, len(predicted_wind[0])))
        j = 0
        for i in range(len(self._test_ulog_data['x'])):
            if ((self._test_ulog_data['x'][i] > self.terrain.x_terr[0]) and
                    (self._test_ulog_data['x'][i] < self.terrain.x_terr[-1]) and
                    (self._test_ulog_data['y'][i] > self.terrain.y_terr[0]) and
                    (self._test_ulog_data['y'][i] < self.terrain.y_terr[-1]) and
                    (self._test_ulog_data['alt'][i] > self.terrain.z_terr[0]) and
                    (self._test_ulog_data['alt'][i] < self.terrain.z_terr[-1])):
                test_wind_inside_terrain[0][j] = self._test_ulog_data['wn'][i]
                test_wind_inside_terrain[1][j] = self._test_ulog_data['we'][i]
                test_wind_inside_terrain[2][j] = self._test_ulog_data['wd'][i]
                j += 1

        test_wind_inside_terrain = list(test_wind_inside_terrain)
        mean_absolute_error_x = metrics.mean_absolute_error(test_wind_inside_terrain[0], predicted_wind[0])
        mean_absolute_error_y = metrics.mean_absolute_error(test_wind_inside_terrain[1], predicted_wind[1])
        mean_absolute_error_z = metrics.mean_absolute_error(test_wind_inside_terrain[2], predicted_wind[2])
        mean_squared_error_x = metrics.mean_squared_error(test_wind_inside_terrain[0], predicted_wind[0])
        mean_squared_error_y = metrics.mean_squared_error(test_wind_inside_terrain[1], predicted_wind[1])
        mean_squared_error_z = metrics.mean_squared_error(test_wind_inside_terrain[2], predicted_wind[2])
        mean_absolute_error = mean_absolute_error_x + mean_absolute_error_y + mean_absolute_error_z
        mean_squared_error = mean_squared_error_x + mean_squared_error_y + mean_squared_error_z
        print('Mean absolute error is: ', mean_absolute_error)
        print('Mean square error is: ', mean_squared_error)

    def build_csv(self):
        try:
            csv_args = utils.BasicParameters(self._config_yaml, 'csv')
            print('Saving csv to {0}'.format(csv_args.params['file']))
            utils.build_csv(self.terrain.x_terr, self.terrain.y_terr, self.terrain.z_terr, self.terrain.full_block,
                            self.terrain.cosmo_corners, csv_args.params['file'])
        except:
            print('CSV filename parameter (csv:file) not found in {0}, csv not saved'.format(self._config_yaml))

    def run_prediction(self, input):
        return self.net(input.unsqueeze(0)).squeeze(0)

    def evaluate_loss(self, output):
        # input = is_wind.repeat(1, self.__num_outputs, 1, 1, 1) * x
        if self.flag.test_simulated_data:
            nn_output = output[0:3, :, :, :]
            labels = self.labels.to(self._device)
        if self.flag.test_flight_data:
            nn_output = output[0:3, :, :, :]
            nn_output[self._wind_mask] = 0.0
            labels = self._wind_zeros
        return self._loss_fn(nn_output, labels)

    def get_prediction(self):
        input = self.generate_wind_input()       # Set new rotation
        output = self.run_prediction(input)      # Run network prediction with input csv
        return output

    def get_ulog_prediction(self):
        input = self.generate_ulog_input()
        output = self.run_prediction(input)
        return output

    # --- Optimisation functions ---

    def optimise_wind_variables(self, opt, n=1000, min_gradient=1e-5, opt_kwargs={'learning_rate':1e-5}, verbose=False):
        optimizer = opt([self._optimisation_variables], **opt_kwargs)
        print(optimizer)
        t0 = time.time()
        t = 0
        max_grad = min_gradient+1.0
        losses, grads, optimisation_variables_ = [], [], []
        while t < n and max_grad > min_gradient:
            # if self.flag.optimize_corners_individually:
            #     print(t,
            #               ' r: ', *self._optimisation_variables[0].detach().cpu().numpy() * 180.0 / np.pi,
            #               ' s: ', *self._optimisation_variables[1].detach().cpu().numpy(),
            #               ' ds: ', *self._optimisation_variables[2].detach().cpu().numpy() * 180.0 / (np.pi*10000),
            #               '')
            # else:
            #     print(t,
            #           ' r: ', self._optimisation_variables[0].detach().cpu().numpy() * 180.0 / np.pi,
            #           ' s: ', self._optimisation_variables[1].detach().cpu().numpy(),
            #           ' ds: ', self._optimisation_variables[2].detach().cpu().numpy() * 180.0 / (np.pi*10000),
            #           '')
            print(t, *self._optimisation_variables.detach().cpu().numpy())
            optimisation_variables_.append(self._optimisation_variables.clone().detach().cpu().numpy())
            optimizer.zero_grad()
            t1 = time.time()
            output = self.get_prediction()
            tp = time.time()

            loss = self.evaluate_loss(output)
            tl = time.time()

            losses.append(loss.item())
            print('loss={0:0.3e}, '.format(loss.item()), end='')

            # Calculate derivative of loss with respect to optimisation variables
            loss.backward(retain_graph=True)
            tb = time.time()

            max_grad = self._optimisation_variables.grad.abs().max()
            print('Max grad: {0:0.3e}'.format(max_grad))
            grads.append(max_grad)

            # Step with gradient
            optimizer.step()
            to = time.time()

            if verbose:
                print('Times: prediction: {0:6.3f}s'.format(tp - t1), end='')
                print(', loss: {0:6.3f}s'.format(tl - tp), end='')
                print(', backward: {0:6.3f}s'.format(tb - tl), end='')
                print(', opt step: {0:6.3f}s'.format(to - tb))
            t += 1
        tt = time.time()-t0
        if verbose:
            print('Total time: {0}s, avg. per step: {1}'.format(tt, tt/t))
        return np.array(optimisation_variables_), np.array(losses), np.array(grads)

    def run_original_input_prediction(self):
        input_ = self.original_input

        return output, loss
