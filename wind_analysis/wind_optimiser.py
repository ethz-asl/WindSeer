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
from datetime import datetime
from scipy import ndimage
import mlrose
from scipy.interpolate import CubicSpline
from scipy.spatial import distance
from scipy.interpolate import RegularGridInterpolator as RGI
import random
import torch
import os
import time
import copy
import math


def angle_between_vectors(v1, v2, deg_rad=0):
    """ Returns angle in deg or rad between two vectors 'v1' and 'v2' """
    # Unit vectors
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)

    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    angle_deg = angle_rad * 180 / math.pi

    if deg_rad == 0:
        return angle_deg
    else:
        return angle_rad

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
    def __init__(self, test, cfd, flight, wind, noise, window_split, optimisation):
        self.test_simulated_data = test.params['test_simulated_data']
        self.test_flight_data = test.params['test_flight_data']
        self.print_names = cfd.params['print_names']
        self.use_ekf_wind = flight.params['use_ekf_wind']
        self.add_wind_measurements = wind.params['add_wind_measurements']
        self.add_corners = wind.params['add_corners']
        self.use_sparse_data = wind.params['sparse_data']['use_sparse_data']
        self.terrain_percentage_correction = wind.params['sparse_data']['terrain_percentage_correction']
        self.sample_random_region = wind.params['sparse_data']['sample_random_region']
        self.use_trajectory = wind.params['trajectory']['use_trajectory_generation']
        self.use_simulated_trajectory = wind.params['trajectory']['use_simulated_trajectory']
        self.predict_flight = wind.params['flight']['predict_flight']
        self.generate_turbulence = noise.params['generate_turbulence']
        self.add_gaussian_noise = noise.params['add_gaussian_noise']
        self.use_window_split = window_split.params['use_window_split']
        self.use_optimized_corners = window_split.params['use_optimized_corners']
        self.use_gpr_prediction = window_split.params['use_gpr_prediction']
        self.use_krigging_prediction = window_split.params['use_krigging_prediction']
        self.rescale_prediction = window_split.params['rescale_prediction']
        self.optimise_corners_individually = optimisation.params['optimise_corners_individually']
        self.use_scale_optimisation = optimisation.params['use_scale_optimisation']
        self.use_spline_optimisation = optimisation.params['use_spline_optimisation']


class WindOptimiser(object):
    _loss_fn = torch.nn.MSELoss()
    _optimisation_variables = None

    def __init__(self, config_yaml, resolution=64):
        # Configuration variables
        # self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._device = torch.device("cpu")
        self._config_yaml = config_yaml
        self._test_args = utils.BasicParameters(self._config_yaml, 'test')
        self._cfd_args = utils.BasicParameters(self._config_yaml, 'cfd')
        self._cosmo_args = utils.COSMOParameters(self._config_yaml)
        self._flight_args = utils.FlightParameters(self._config_yaml)
        self._wind_args = utils.BasicParameters(self._config_yaml, 'wind')
        self._noise_args = utils.BasicParameters(self._config_yaml, 'noise')
        self._window_splits_args = utils.BasicParameters(self._config_yaml, 'window_split')
        self._optimisation_args = utils.BasicParameters(self._config_yaml, 'optimisation')
        self._model_args = utils.BasicParameters(self._config_yaml, 'model')
        self._second_model_args = utils.BasicParameters(self._config_yaml, 'second_model')
        self._resolution = resolution

        # Selection flags
        self.flag = SelectionFlags(self._test_args, self._cfd_args, self._flight_args, self._wind_args,
                                   self._noise_args, self._window_splits_args, self._optimisation_args)

        # Network model and loss function
        self.net, self.params = self.load_network_model()
        self.net.freeze_model()
        self.second_net, self.second_params = self.load_network_model(second_model=True)
        self.net.freeze_model()
        self._loss_fn = self.get_loss_function()

        # Load data
        if self.flag.test_simulated_data:
            self.original_input, self.labels, self.scale, self.data_set_name, self.grid_size, \
                self.binary_terrain, self.original_terrain \
                = self.load_data_set()
            self.terrain = self.get_cfd_terrain()
        if self.flag.test_flight_data:
            self._flight_data = self.load_flight_data()
            self._cosmo_wind = self.load_wind()
            self.terrain = self.load_cosmo_terrain()
            self.wind_vector_angles = self.get_angles_between_wind_vectors()

        # Wind measurements variables
        if self.flag.test_simulated_data:
            if self.flag.use_sparse_data:
                self._wind_zeros = self.get_sparse_wind_blocks()
            if self.flag.use_trajectory:
                self._wind_blocks, self._wind_zeros, self._wind_mask, self._simulated_flight_data\
                    = self.get_trajectory_wind_blocks()
        if self.flag.test_flight_data:
            input_flight_data = self._flight_data
            # input_flight_data, _ = self.sliding_window_split(self._flight_data, 300, 1, 600)
            self._wind_blocks, self._var_blocks, self._wind_zeros, self._wind_mask \
                = self.get_flight_wind_blocks(input_flight_data)
            # TODO: replace if there are actual labels for flight data
            self.labels = self._wind_zeros

        # Noise
        # if self.flag.add_gaussian_noise:
        #     self._wind_blocks, self._wind_zeros = self.add_gaussian_noise(self._wind_zeros, self._wind_blocks)
        # if self.flag.generate_turbulence:
        #     self._wind_blocks, self._wind_zeros = self.generate_turbulence(self._wind_zeros, self._wind_blocks)

        # Optimisation variables
        self._optimisation_variables = self.get_optimisation_variables()
        self.reset_optimisation_variables()
        if self.flag.test_simulated_data:
            self._base_cfd_corners = self.get_cfd_corners()
            # self._optimized_corners = self.get_optimized_corners()
        if self.flag.test_flight_data:
            self._base_cosmo_corners = self.get_cosmo_corners()
        self._interpolator = DataInterpolation(self._device, 3, *self.terrain.get_dimensions())

    # --- Network and loss function ---

    def load_network_model(self, second_model=False):
        if second_model:
            model_args = self._second_model_args
        else:
            model_args = self._model_args
        print('Loading network model...', end='', flush=True)
        t_start = time.time()
        yaml_loc = os.path.join(model_args.params['location'], model_args.params['name'], 'params.yaml')
        params = utils.EDNNParameters(yaml_loc)  # load the model config
        NetworkType = getattr(models, params.model['model_type'])  # load the model
        net = NetworkType(**params.model_kwargs())  # load learnt parameters
        model_loc = os.path.join(model_args.params['location'], model_args.params['name'],
                                 model_args.params['version'] + '.model')
        net.load_state_dict(torch.load(model_loc, map_location=lambda storage, loc: storage))
        net = net.to(self._device)
        print(' done [{:.2f} s]'.format(time.time() - t_start))
        return net, params

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

    # --- Load data ---

    def load_flight_data(self):
        self._flight_args.print()
        file_name = self._flight_args.params['file']
        extension = file_name.split('.')[-1]
        if extension == 'ulg':
            print('Loading ulog flight data...', end='', flush=True)
            t_start = time.time()
            ulog_data = ulog_utils.get_log_data(self._flight_args.params['file'])

            if self.flag.use_ekf_wind:
                ulog_data['wn'] = ulog_data['we_north']
                ulog_data['we'] = ulog_data['we_east']
                ulog_data['wd'] = ulog_data['we_down']
            else:
                ulog_data['wn'] = ulog_data['wn']
                ulog_data['we'] = ulog_data['we']
                ulog_data['wd'] = ulog_data['wd']
            ulog_data['time_microsec'] = ulog_data['utc_microsec']

            flight_data = ulog_data
            print(' done [{:.2f} s]'.format(time.time() - t_start))
        elif extension == 'hdf5':
            print('Loading hdf5 flight data...', end='', flush=True)
            t_start = time.time()
            hdf5_data = ulog_utils.read_filtered_hdf5(self._flight_args.params['file'], skip_amount=10)
            hdf5_data['wn'] = hdf5_data['wind_n']
            hdf5_data['we'] = hdf5_data['wind_e']
            hdf5_data['wd'] = hdf5_data['wind_d']
            hdf5_data['time_microsec'] = hdf5_data['time']

            flight_data = hdf5_data
            print(' done [{:.2f} s]'.format(time.time() - t_start))
        else:
            raise ValueError('Unknown file type: ' + extension + '. ' + file_name + ' should have a .ulg or .hdf5 extension')

        return flight_data

    def load_wind(self):
        print('Loading COSMO wind...', end='', flush=True)
        t_start = time.time()
        self._cosmo_args.print()

        lat0, lon0 = self._flight_data['lat'][0], self._flight_data['lon'][0]

        if 'utc_microsec' in self._flight_data.keys():
            # Get time of flight
            t0 = datetime.utcfromtimestamp(self._flight_data['utc_microsec'][0] / 1e6)
            time_of_flight = t0.hour
        else:
            # Get time of flight from file name
            try:
                file_name = self._flight_args.params['file']
                time_of_flight = int(file_name.split('/')[-1].split('_')[0])
            except ValueError:
                print('Cannot extract hour of flight from file name')
        # Get cosmo wind
        offset_cosmo_time = self._cosmo_args.get_cosmo_time(time_of_flight)
        cosmo_wind = cosmo.extract_cosmo_data(self._cosmo_args.params['file'], lat0, lon0, offset_cosmo_time,
                                              terrain_file=self._cosmo_args.params['terrain_file'])
        print(' done [{:.2f} s]'.format(time.time() - t_start))
        return cosmo_wind

    def load_cosmo_terrain(self):
        print('Loading terrain...', end='', flush=True)
        t_start = time.time()
        # Get corresponding terrain
        # min_height = min(flight_data['alt'].min(), h_terr.min())
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
                                       augmentation=False, return_grid_size=True,
                                       **self.params.Dataset_kwargs())
        # print names of files in test_set
        if self.flag.print_names:
            for i in range(len(test_set)):
                name = test_set.get_name(i)
                print(i, name)
        # Get data set from test set
        data_set = test_set[self._cfd_args.params['index']]
        input = data_set[0]
        labels = data_set[1]
        name = test_set.get_name(self._cfd_args.params['index'])
        scale = 1.0
        if self.params.data['autoscale']:
            scale = data_set[3].item()
        print("Loading test set (index) with name: ", str(name))
        grid_size = data.get_grid_size(self._cfd_args.params['testset_name'])
        # Convert distance transformed matrix back to binary matrix
        binary_terrain = data_set[0][0, :, :, :] <= 0
        terrain = data_set[0][0, :, :, :]
        return input, labels, scale, name, grid_size, binary_terrain, terrain

    def get_cfd_terrain(self):
        nx, ny, nz = self.binary_terrain.shape
        x_terr = torch.Tensor([self.grid_size[0]*i for i in range(nx)])
        y_terr = torch.Tensor([self.grid_size[1]*i for i in range(ny)])
        z_terr = torch.Tensor([self.grid_size[2]*i for i in range(nz)])
        h_terr = torch.Tensor(np.squeeze(self.binary_terrain.sum(axis=2, keepdims=True)*self.grid_size[2]))
        h_terr = torch.clamp(h_terr, 0, z_terr[-1])  # make sure height is not bigger than max z_terr
        # Get terrain object
        boolean_terrain = self._model_args.params['boolean_terrain']
        terrain = TerrainBlock(x_terr, y_terr, z_terr, h_terr, self.binary_terrain,
                               device=self._device, boolean_terrain=boolean_terrain)
        return terrain

    # --- Wind generation ---

    def get_sparse_wind_blocks(self, p=0):
        wind = self.labels.clone()
        channels, nz, ny, nx = wind.shape
        terrain = self.original_terrain
        boolean_terrain = terrain > 0
        use_nonzero_idx = False
        if use_nonzero_idx:
            numel = int(terrain.numel() * p)
            idx = torch.nonzero(terrain)
            select = torch.randperm(idx.shape[0])
            mask = torch.zeros_like(terrain)
            mask[idx[select][:numel].split(1, dim=1)] = 1
        else:
            # terrain correction
            if self.flag.terrain_percentage_correction:
                terrain_percentage = 1 - boolean_terrain.sum().item() / boolean_terrain.numel()
                correctected_percentage = p / (1 - terrain_percentage)
                percentage = correctected_percentage
            else:
                percentage = p

            # sample random region
            if self.flag.sample_random_region:
                p_sample = 0.2 + random.random() * 0.3  # sample between 0.2 and 0.5
                unifrom_dist_region = torch.zeros((nz, ny, nx)).float()
                l = int(np.sqrt(nx * ny * p_sample))
                x0 = int(l / 2) + 1
                x1 = nx - (int(l / 2) + 1)
                rx = random.randint(x0, x1)
                ry = random.randint(x0, x1)
                unifrom_dist_region[:, ry - int((l + 1) / 2):ry + int(l / 2), rx - int((l + 1) / 2):rx + int(l / 2)] \
                    = torch.FloatTensor(nz, l, l).uniform_()
                terrain_uniform_mask = boolean_terrain.float() * unifrom_dist_region
                percentage = p / p_sample
            else:
                uniform_dist = torch.FloatTensor(nz, ny, nx).uniform_()
                terrain_uniform_mask = boolean_terrain.float() * uniform_dist

            # sparsity mask
            mask = terrain_uniform_mask > (1 - percentage)

        sparse_wind_mask = mask.float().unsqueeze(0)
        augmented_wind = torch.cat([wind, sparse_wind_mask])
        return augmented_wind.to(self._device)

    def get_trajectory_wind_blocks(self):
        # type of trajectories
        spiral_trajectory = False
        random_trajectory = True

        # spiral trajectory
        if spiral_trajectory:
            dot = 400
            d = dot * 0.1
            x = []
            y = []
            z = []
            for i in range(dot):
                t = i / d * np.pi
                x.append(750 + t * math.cos(t) * 19)
                y.append(750 + t * math.sin(t) * 19)
                z.append(0 + 700/dot*i)
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
                    wind[:, id_z, id_y, id_x] = self.labels[:, id_z, id_y, id_x]

        # random trajectory
        if random_trajectory:
            terrain = self.original_terrain.clone()
            numel = self._wind_args.params['trajectory']['num_points_generated']
            use_region = False
            if use_region:
                channels, nz, ny, nx = self.labels.shape
                p_sample = 0.2 + random.random() * 0.3  # sample between 0.2 and 0.5
                random_region = torch.zeros((nz, ny, nx)).float()
                l = int(np.sqrt(nx * ny * p_sample))
                x0 = int(l / 2) + 1
                x1 = nx - (int(l / 2) + 1)
                rx = random.randint(x0, x1)
                ry = random.randint(x0, x1)
                random_region[:, ry - int((l + 1) / 2):ry + int(l / 2), rx - int((l + 1) / 2):rx + int(l / 2)] = \
                    torch.ones(nz, l, l)
                terrain_region = terrain * random_region
            else:
                terrain_region = terrain
            idx = torch.nonzero(terrain_region)
            # set torch manual seed
            torch.manual_seed(7)
            select = torch.randperm(idx.shape[0])
            # 3D positions of the random points
            pos_x = idx[select][:numel][:, 2].float() * self.grid_size[0]
            pos_y = idx[select][:numel][:, 1].float() * self.grid_size[1]
            pos_z = idx[select][:numel][:, 0].float() * self.grid_size[2]
            # set random manual seed
            random.seed(7)
            # random starting point
            start = random.randint(0, numel)
            # get trajectory points
            points = torch.cat((pos_x.unsqueeze(1), pos_y.unsqueeze(1), pos_z.unsqueeze(1)), dim=1)
            distances = distance.squareform(distance.pdist(points))
            closest = np.argsort(distances, axis=1)
            x, y, z = [], [], []
            closest_point = 0
            current_points = []
            for i in range(closest.shape[0]):
                if i == 0:
                    closest_point = start
                    current_points.append(closest_point)
                else:
                    n = 1
                    for j in range(closest.shape[1]):
                        closest_point = closest[closest_point, n]
                        # check that we don't go back to the same points
                        if closest_point in current_points:
                            n += 1
                            if n == closest.shape[1]-1:
                                break
                        else:
                            current_points.append(closest_point)
                            break
                x.append(pos_x[closest_point])
                y.append(pos_y[closest_point])
                z.append(pos_z[closest_point])

            uav_speed = self._wind_args.params['trajectory']['uav_speed']
            dt = self._wind_args.params['trajectory']['dt']
            x_traj, y_traj, z_traj = generate_trajectory(x, y, z, uav_speed, dt, self.terrain, z_above_terrain=True)

            # generate simulated flight data
            simulated_flight_data = {}
            interpolating_function_x = RGI((self.terrain.z_terr, self.terrain.y_terr, self.terrain.x_terr),
                                           self.labels[0, :].detach().cpu().numpy(), method='nearest')
            interpolating_function_y = RGI((self.terrain.z_terr, self.terrain.y_terr, self.terrain.x_terr),
                                           self.labels[1, :].detach().cpu().numpy(), method='nearest')
            interpolating_function_z = RGI((self.terrain.z_terr, self.terrain.y_terr, self.terrain.x_terr),
                                           self.labels[2, :].detach().cpu().numpy(), method='nearest')
            pts = np.column_stack((z_traj, y_traj, x_traj))
            # distances = np.sqrt(np.sum(np.diff(pts, axis=0)**2, 1))

            interpolated_flight_data_x = interpolating_function_x(pts)
            interpolated_flight_data_y = interpolating_function_y(pts)
            interpolated_flight_data_z = interpolating_function_z(pts)
            simulated_flight_data['x'] = pts[:, 2]
            simulated_flight_data['y'] = pts[:, 1]
            simulated_flight_data['alt'] = pts[:, 0]
            simulated_flight_data['wn'] = interpolated_flight_data_x.astype(float)
            simulated_flight_data['we'] = interpolated_flight_data_y.astype(float)
            simulated_flight_data['wd'] = -interpolated_flight_data_z.astype(float)
            simulated_flight_data['time_microsec'] = np.array([i*dt*1e6 for i in range(x_traj.size)])


            # Get the bins and wind along the trajectory
            wind = torch.zeros(self.labels.shape) * float('nan')
            for i in range(x_traj.size):
                    id_x = (int((x_traj[i] - self.terrain.x_terr[0]) / self.grid_size[0]))
                    id_y = (int((y_traj[i] - self.terrain.y_terr[0]) / self.grid_size[1]))
                    id_z = (int((z_traj[i] - self.terrain.z_terr[0]) / self.grid_size[2]))
                    wind[:, id_z, id_y, id_x] = self.labels[:, id_z, id_y, id_x]

        wind_mask = torch.isnan(wind)
        wind_zeros = wind.clone()
        wind_zeros[wind_mask] = 0
        # if self.flag.add_gaussian_noise:
        #     wind_zeros = self.add_gaussian_noise(wind_zeros, wind_mask)
        return wind.to(self._device), wind_zeros.to(self._device), wind_mask.to(self._device), simulated_flight_data

    def get_flight_wind_blocks(self, flight_data):
        #print('Getting binned wind blocks...', end='', flush=True)
        #t_start = time.time()

        # Determine the grid dimensions
        dx = self.terrain.x_terr[[0, -1]]; ddx = (self.terrain.x_terr[1]-self.terrain.x_terr[0])/2.0
        dy = self.terrain.y_terr[[0, -1]]; ddy = (self.terrain.y_terr[1]-self.terrain.y_terr[0])/2.0
        dz = self.terrain.z_terr[[0, -1]]; ddz = (self.terrain.z_terr[1]-self.terrain.z_terr[0])/2.0
        grid_dimensions = {'x_min': dx[0] - ddx, 'x_max': dx[1] + ddx, 'y_min': dy[0] - ddy, 'y_max': dy[1] + ddy,
                           'z_min': dz[0] - ddz, 'z_max': dz[1] + ddz, 'n_cells': self.terrain.get_dimensions()[0]}

        FlightInterpolator = UlogInterpolation(flight_data, grid_dimensions, self.terrain)

        # bin the data into the regular grid
        try:
            if self._wind_args.params['flight']['interpolation_method'].lower() == 'bin':
                wind, variance = FlightInterpolator.bin_log_data()
            elif self._wind_args.params['flight']['interpolation_method'].lower() == 'idw':
                wind, variance = FlightInterpolator.interpolate_log_data_idw()
            elif self._wind_args.params['flight']['interpolation_method'].lower() == 'krigging':
                wind, variance, _ = FlightInterpolator.interpolate_log_data_krigging()
            elif self._wind_args.params['flight']['interpolation_method'].lower() == 'gpr':
                wind, variance, _ = FlightInterpolator.interpolate_log_data_gpr()
            else:
                print('Specified interpolation method: {0} unknown!'
                      .format(self._flight_args.params['interpolation_method']))
                raise ValueError
        except KeyError:
            print('Interpolation method not specified in file: {0}'
                  .format(self._config_yaml))

        wind_mask = torch.isnan(wind)       # This is a binary mask with ones where there are invalid wind estimates
        wind_zeros = wind.clone()
        wind_zeros[wind_mask] = 0

        # print(' done [{:.2f} s]'.format(time.time() - t_start))
        return wind, variance, wind_zeros.to(self._device), wind_mask.to(self._device)

    # --- Add noise to data ---

    def add_gaussian_noise(self, wind_zeros, wind_mask):
        wind_noise = torch.randn(wind_zeros.shape).to(self._device) / self.scale
        wind_zeros_noise = wind_zeros + wind_noise
        wind_zeros_noise[wind_mask] = 0
        # wind_blocks_noise = wind_blocks + wind_noise
        # wind_blocks_noise[wind_mask] = float('nan')
        return wind_zeros_noise

    def generate_turbulence(self, wind_zeros, wind_blocks):
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
        if self.flag.optimise_corners_individually:
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
        input_ = copy.deepcopy(self.original_input)
        # input_ = copy.deepcopy(self.labels)
        corners = input_[1:4, ::nx-1, ::ny-1, :]
        cfd_corners = np.transpose(corners, (0, 3, 1, 2))
        return cfd_corners.to(self._device)

    def get_cosmo_corners(self):
        temp_cosmo = cosmo.cosmo_corner_wind(self._cosmo_wind, self.terrain.z_terr, rotate=0.0, scale=1.0,
                                             terrain_height=self.terrain.terrain_corners)
        cosmo_corners = torch.from_numpy(temp_cosmo.astype(np.float32)).to(self._device)
        return cosmo_corners

    # --- Helper functions ---
    def add_mask_to_wind(self, wind_provided=None, wind_mask=None):
        if wind_provided is None:
            wind = self._wind_zeros.clone()
        else:
            wind = wind_provided
        if wind_mask is None:
            sparse_mask = (~self._wind_mask[0]).float()
        else:
            sparse_mask = (~wind_mask[0]).float()
        augmented_wind = torch.cat(([wind, sparse_mask.unsqueeze(0)]))
        return augmented_wind.to(self._device)

    def get_angles_between_wind_vectors(self):
        wind_e = self._flight_data['we']
        wind_n = self._flight_data['wn']
        wind_d = self._flight_data['wd']
        wind_vectors = [[wind_e[i], wind_n[i], wind_d[i]] for i in range(len(wind_e))]
        angles = []
        for i in range(len(wind_e)-1):
            angles.append(angle_between_vectors(wind_vectors[i], wind_vectors[i+1]))
        return angles

    def get_rotated_wind(self):
        sr, cr = [], []
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
        # if self.flag.add_wind_measurements:
        #     interpolated_wind += self._wind_zeros
        # wind_mask = self.terrain.network_terrain <= 0
        # wind = self.add_mask_to_wind(interpolated_wind, wind_mask)
        input = torch.cat([self.terrain.network_terrain, interpolated_wind])
        return input

    def build_csv(self):
        try:
            csv_args = utils.BasicParameters(self._config_yaml, 'csv')
            print('Saving csv to {0}'.format(csv_args.params['file']))
            utils.build_csv(self.terrain.x_terr, self.terrain.y_terr, self.terrain.z_terr, self.terrain.full_block,
                            self.terrain.cosmo_corners, csv_args.params['file'])
        except:
            print('CSV filename parameter (csv:file) not found in {0}, csv not saved'.format(self._config_yaml))

    def get_prediction(self):
        input = self.generate_wind_input()       # Set new rotation
        output = self.run_prediction(input, second_model=True)      # Run network prediction with input csv
        return input, output

    def run_prediction(self, input, second_model=False):
        if second_model:
            return self.second_net(input.unsqueeze(0))['pred'].squeeze(0)
        else:
            return self.net(input.unsqueeze(0))['pred'].squeeze(0)

    def evaluate_loss(self, output):
        # input = is_wind.repeat(1, self.__num_outputs, 1, 1, 1) * x
        if self.flag.test_simulated_data:
            nn_output = output[0:3, :, :, :]
            labels = self.labels.to(self._device)
        if self.flag.test_flight_data:
            nn_output = output[0:3, :, :, :]
            # nn_output[self._wind_mask] = 0.0
            labels = self._wind_zeros
        return self._loss_fn(nn_output, labels)

    def sliding_window_split(self, flight_data, window_size=1, response_size=1, step_size=1):
        input_flight, output_flight = {}, {}
        for keys, values in flight_data.items():
            input_batch = values[step_size : step_size+window_size]
            output_batch = values[step_size+window_size : step_size+window_size+response_size]
            input_flight.update({keys: input_batch})
            output_flight.update({keys: output_batch})

        return copy.deepcopy(input_flight), copy.deepcopy(output_flight)

    def rescale_prediction(self, output=None, label=None):
        if output is not None:
            output = output.squeeze()
        channels_to_predict = self.params.data['label_channels']
        # make sure the channels to predict exist and are properly ordered
        default_channels = ['terrain', 'ux', 'uy', 'uz', 'turb', 'p', 'epsilon', 'nut']
        for channel in channels_to_predict:
            if channel not in default_channels:
                raise ValueError('Incorrect label_channel detected: \'{}\', '
                                 'correct channels are {}'.format(channel, default_channels))
        channels_to_predict = [x for x in default_channels if x in channels_to_predict]

        # rescale the labels and predictions
        for i, channel in enumerate(channels_to_predict):
            if channel == 'terrain':
                if output is not None:
                    output[i] *= self.params.data[channel + '_scaling']
                if label is not None:
                    label[i] *= self.params.data[channel + '_scaling']
            elif channel.startswith('u') or channel == 'nut':
                if output is not None:
                    output[i] *= self.scale * self.params.data[channel + '_scaling']
                if label is not None:
                    label[i] *= self.scale * self.params.data[channel + '_scaling']
            elif channel == 'p' or channel == 'turb':
                if output is not None:
                    output[i] *= self.scale * self.scale * self.params.data[channel + '_scaling']
                if label is not None:
                    label[i] *= self.scale * self.scale * self.params.data[channel + '_scaling']
            elif channel == 'epsilon':
                if output is not None:
                    output[i] *= self.scale * self.scale * self.scale * self.params.data[channel + '_scaling']
                if label is not None:
                    label[i] *= self.scale * self.scale * self.scale * self.params.data[channel + '_scaling']

    # --- Optimisation functions ---

    def optimise_wind_corners(self, opt, n=1000, min_gradient=1e-5, opt_kwargs={'learning_rate':1e-5},
                              wind_zeros=None, wind_mask=None,
                              print_steps=False, verbose=False):
        optimizer = opt([self._optimisation_variables], **opt_kwargs)
        if print_steps:
            print(optimizer)
        t0 = time.time()
        t = 0
        max_grad = min_gradient+1.0
        output, input = [], []
        losses, grads, opt_var = [], [], []
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
            if print_steps:
                print(t, *self._optimisation_variables.detach().cpu().numpy())
            opt_var.append(self._optimisation_variables.clone().detach().cpu().numpy())
            optimizer.zero_grad()
            t1 = time.time()
            input, output = self.get_prediction()
            output_copy = output.detach().clone()
            # outputs.append(output)
            # inputs.append(input)
            tp = time.time()

            if wind_zeros is None and wind_mask is None:
                loss = self.evaluate_loss(output)
            else:
                output[wind_mask] = 0.0
                labels = wind_zeros
                loss = self._loss_fn(output, labels)
            tl = time.time()

            losses.append(loss.item())
            if print_steps:
                print('loss={0:0.3e}, '.format(loss.item()), end='')

            # Calculate derivative of loss with respect to optimisation variables
            loss.backward(retain_graph=True)
            tb = time.time()

            max_grad = self._optimisation_variables.grad.abs().max()
            if print_steps:
                print('Max grad: {0:0.3e}'.format(max_grad))
            grads.append(max_grad)

            # Step with gradient
            optimizer.step()
            to = time.time()

            if t == n-1 or max_grad < min_gradient:
                last_output = output_copy.clone()

            if verbose:
                print('Times: prediction: {0:6.3f}s'.format(tp - t1), end='')
                print(', loss: {0:6.3f}s'.format(tl - tp), end='')
                print(', backward: {0:6.3f}s'.format(tb - tl), end='')
                print(', opt step: {0:6.3f}s'.format(to - tb))
            t += 1

        tt = time.time()-t0
        if verbose:
            print('Total time: {0}s, avg. per step: {1}'.format(tt, tt/t))
        print('Corner optimisation is done: [{:.2f} s]'.format(tt))
        print('loss={0:0.3e}, '.format(losses[-1]))
        return last_output, input, np.array(opt_var), np.array(losses), np.array(grads)

    def function_window_split(self, window_split_variables):
        t0 = time.time()

        # window split variables
        window_size = window_split_variables[0]
        response_size = window_split_variables[1]
        step_size = window_split_variables[2]

        # calculate average nn loss
        if self.flag.test_flight_data:
            flight_data = self._flight_data
        if self.flag.test_simulated_data:
            flight_data = self._simulated_flight_data

        max_num_windows = int((flight_data['x'].size - (window_size + response_size)) / step_size + 1)
        nn_losses = 0
        n = 0
        for i in range(max_num_windows):
            # split data into input and label based on window variables
            input_flight_data, label_flight_data = \
                self.sliding_window_split(flight_data, window_size,
                                          response_size, i * step_size)

            # labels
            # filter out points in the labels which are outside the terrain sample
            valid_labels = []
            for j in range(len(label_flight_data['x'])):
                if ((label_flight_data['x'][j] > self.terrain.x_terr[0]) and
                        (label_flight_data['x'][j] < self.terrain.x_terr[-1]) and
                        (label_flight_data['y'][j] > self.terrain.y_terr[0]) and
                        (label_flight_data['y'][j] < self.terrain.y_terr[-1]) and
                        (label_flight_data['alt'][j] > self.terrain.z_terr[0]) and
                        (label_flight_data['alt'][j] < self.terrain.z_terr[-1])):
                    valid_labels.append([label_flight_data['wn'][j], label_flight_data['we'][j],
                                         label_flight_data['wd'][j]])
            if len(valid_labels) == 0:
                labels = []
                continue
            else:
                labels = np.row_stack(valid_labels)

            # --- Network prediction ---
            # bin input flight data
            wind_blocks, var_blocks, wind_zeros, wind_mask \
                = self.get_flight_wind_blocks(input_flight_data)

            # get prediction
            if self.flag.add_gaussian_noise:
                wind_zeros = self.add_gaussian_noise(wind_zeros, wind_mask)
            wind = wind_zeros.to(self._device)
            if self.flag.add_corners:
                interpolated_cosmo_corners = self._interpolator.edge_interpolation(self._base_cosmo_corners)
                wind += interpolated_cosmo_corners
            wind = self.add_mask_to_wind(wind, wind_mask)
            input = torch.cat([self.terrain.network_terrain, wind])
            output = self.run_prediction(input)
            # loss_org = self.evaluate_loss(output)
            # print(' loss: ', loss_org.item())

            # interpolate prediction to scattered flight data locations corresponding to the labels
            FlightInterpolator = UlogInterpolation(input_flight_data, terrain=self.terrain,
                                                   wind_data_for_prediction=label_flight_data)
            interpolated_output = FlightInterpolator.interpolate_log_data_from_grid(output)
            nn_output = np.column_stack(interpolated_output)
            nn_loss = self._loss_fn(torch.Tensor(nn_output).to(self._device),
                                    torch.Tensor(labels).to(self._device))
            # print('NN loss is: ', nn_loss)
            # interpolated_output = np.resize(np.array(interpolated_output), (3, len(interpolated_output[0])))

            # --- Average wind prediction ---
            average_wind_input = np.array([input_flight_data['wn'], input_flight_data['we'], input_flight_data['wd']])
            average_wind_output = np.ones_like(labels) * [average_wind_input[0].mean(),
                                                          average_wind_input[1].mean(),
                                                          average_wind_input[2].mean()]
            average_wind_loss = self._loss_fn(torch.Tensor(average_wind_output).to(self._device),
                                 torch.Tensor(labels).to(self._device))

            nn_losses += (nn_loss.item() - average_wind_loss.item())
            # nn_losses += nn_loss.item()
            n += 1

        average_nn_loss = nn_losses/n
        t1 = time.time()
        print('Time for 1 iteration', t1-t0)
        print('Window: ', window_size, 'Response: ', response_size ,' Step: ', step_size)
        print('Average loss is: ', average_nn_loss)
        # Calculate derivative of loss with respect to window split variables
        return average_nn_loss

    def window_split_optimisation(self, n=100):
        window = 60
        response = 40
        step = 100
        window_split_variables = [window, response, step]

        # Set up solver
        if self.flag.test_flight_data:
            flight_data = self._flight_data
        if self.flag.test_simulated_data:
            flight_data = self._simulated_flight_data
        max_val = int(flight_data['x'].size / 3)

        fitness_cust = mlrose.CustomFitness(self.function_window_split)
        problem = mlrose.DiscreteOpt(length=3, fitness_fn=fitness_cust, maximize=False, max_val=max_val)
        schedule = mlrose.ExpDecay()
        init_state = np.array(window_split_variables)
        best_state, best_fitness = mlrose.simulated_annealing(problem, schedule=schedule,
                                                              max_attempts=int(n/10), max_iters=n,
                                                              init_state=init_state, random_state=1)
        print('The best state found is: ', best_state)
        print('The fitness at the best state is: ', best_fitness)
        return best_state

    # --- Prediction functions ---

    def get_original_input_prediction(self):
        original_input = self.original_input.to(self._device)
        labels = self.labels.to(self._device)
        output = self.run_prediction(original_input)
        # self.rescale_prediction(output, labels)
        loss = self._loss_fn(output, labels)
        print('Loss value for original input is: ', loss.item())
        return output, loss

    def sparse_data_prediction(self):
        outputs, losses, inputs = [], [], []

        if self.flag.add_corners:
            wind_corners = self.get_rotated_wind()
            interpolated_wind = self._interpolator.edge_interpolation(wind_corners)

        p = (self._wind_args.params['sparse_data']['percentage_of_sparse_data'])
        wind_zeros = self.get_sparse_wind_blocks(p)
        # interpolated_wind[self._wind_mask] = 0
        if self.flag.add_corners and not self.flag.add_wind_measurements:
            wind_input = interpolated_wind
        elif self.flag.add_wind_measurements and not self.flag.add_corners:
            wind_input = wind_zeros
        else:
            wind_input = interpolated_wind + wind_zeros

        # get prediction
        input = torch.cat([self.terrain.network_terrain, wind_input])
        original_input = input.clone()
        output = self.run_prediction(input)
        loss = self.evaluate_loss(output)
        # self.rescale_prediction(output, self.labels)
        outputs.append(output)
        losses.append(loss)
        inputs.append(original_input)
        print(' percentage: ', p,
              ' loss: ', loss.item())

        return outputs, losses, inputs

    def cfd_trajectory_prediction(self):
        outputs, losses, inputs = [], [], []
        if self.flag.use_simulated_trajectory:
            # bin trajectory points
            input_flight_data = self._simulated_flight_data
            wind_blocks, var_blocks, wind_zeros, wind_mask \
                = self.get_flight_wind_blocks(input_flight_data)
            # get prediction
            wind = wind_zeros.to(self._device)
            if self.flag.add_corners:
                interpolated_cosmo_corners = self._interpolator.edge_interpolation(self._base_cosmo_corners)
                wind += interpolated_cosmo_corners
            wind = self.add_mask_to_wind(wind, wind_mask)
            input = torch.cat([self.terrain.network_terrain, wind])
            output = self.run_prediction(input)
            # self.rescale_prediction(output, self.labels)
            loss = self.evaluate_loss(output)
        else:
            # get wind zeros
            wind_blocks = self._wind_blocks.clone()
            wind_zeros = self._wind_zeros.clone()
            # add noise
            # if self.flag.add_gaussian_noise:
            #     wind_blocks, wind_zeros = self.add_gaussian_noise(wind_zeros, wind_blocks)
            # create mask
            wind_input = self.add_mask_to_wind()
            # run prediction
            input = torch.cat([self.terrain.network_terrain, wind_input])
            output = self.run_prediction(input)
            # self.rescale_prediction(output, self.labels)
            loss = self.evaluate_loss(output)
        outputs.append(output)
        losses.append(loss)
        inputs.append(input)
        print(' loss: ', loss.item())
        return outputs, losses, inputs

    def flight_prediction(self):
        outputs, losses, inputs = [], [], []

        # add mask to input
        wind_input = self.add_mask_to_wind()

        # run prediction
        input = torch.cat([self.terrain.network_terrain, wind_input])
        output = self.run_prediction(input)
        loss = 1
        outputs.append(output)
        losses.append(loss)
        inputs.append(input)
        return outputs, losses, inputs

    def window_split_prediction(self):
        if self.flag.test_flight_data:
            flight_data = self._flight_data
        if self.flag.test_simulated_data:
            flight_data = self._simulated_flight_data

        # sliding window variables (in seconds)
        window_time = self._window_splits_args.params['window_time']  # seconds
        response_time = self._window_splits_args.params['response_time']  # seconds
        step_time = self._window_splits_args.params['step_time']  # seconds

        # time interval between two consecutive flight measurements
        dt = ((flight_data['time_microsec'][-1]-flight_data['time_microsec'][0])/flight_data['time_microsec'].size)/1e6

        # sliding window variables
        window_size = int(window_time/dt)
        response_size = int(response_time/dt)
        step_size = int(step_time/dt)

        # total time of flight and maximum number of windows
        max_num_windows = int((flight_data['x'].size - (window_size + response_size)) / step_size + 1)
        # if max_num_windows < 0:
        #     raise ValueError('window time exceeds flight time')
        total_time_of_flight = (flight_data['time_microsec'][-1] - flight_data['time_microsec'][0]) / 1e6
        print('')
        print('Number of windows: ', max_num_windows)
        print('Number of measurements per window: ', window_size)
        print('Number of measurements per response: ', response_size)
        print('Number of measurements per step: ', step_size)
        print('Total time of flight', total_time_of_flight)
        print('Time interval between two consecutive flight measurements', dt, '\n')

        inputs, outputs = [], []
        nn_losses, zero_wind_losses, average_wind_losses = [], [], []
        gpr_wind_losses, krigging_wind_losses, optimized_corners_losses = [], [], []
        for i in range(max_num_windows):
            # split data into input and label based on window variables
            input_flight_data, label_flight_data = \
                self.sliding_window_split(flight_data, window_size, response_size, i*step_size)
            if self.flag.test_simulated_data and self.flag.rescale_prediction:  # only for simulated trajectory
                label_flight_data['wn'] *= self.scale * self.params.data['ux' + '_scaling']
                label_flight_data['we'] *= self.scale * self.params.data['uy' + '_scaling']
                label_flight_data['wd'] *= self.scale * self.params.data['uz' + '_scaling']

            # labels
            # filter out points in the labels which are outside the terrain sample
            valid_labels = []
            for j in range(len(label_flight_data['x'])):
                if ((label_flight_data['x'][j] > self.terrain.x_terr[0]) and
                        (label_flight_data['x'][j] < self.terrain.x_terr[-1]) and
                        (label_flight_data['y'][j] > self.terrain.y_terr[0]) and
                        (label_flight_data['y'][j] < self.terrain.y_terr[-1]) and
                        (label_flight_data['alt'][j] > self.terrain.z_terr[0]) and
                        (label_flight_data['alt'][j] < self.terrain.z_terr[-1])):
                    valid_labels.append([label_flight_data['wn'][j], label_flight_data['we'][j],
                                         label_flight_data['wd'][j]])
            if len(valid_labels) == 0:
                print(i, 'Labels outside terrain dimensions')
                print('')
                continue
            labels = np.row_stack(valid_labels)
            # labels = np.array([label_flight_data['wn'], label_flight_data['we'], label_flight_data['wd']])

            # --- Network prediction ---
            # bin input flight data
            wind_blocks, var_blocks, wind_zeros, wind_mask \
                = self.get_flight_wind_blocks(input_flight_data)

            # get prediction
            if self.flag.add_gaussian_noise:
                wind_zeros = self.add_gaussian_noise(wind_zeros, wind_mask)
            wind = wind_zeros.to(self._device)
            if self.flag.add_corners:
                interpolated_cosmo_corners = self._interpolator.edge_interpolation(self._base_cosmo_corners)
                wind += interpolated_cosmo_corners
            wind = self.add_mask_to_wind(wind, wind_mask)
            input = torch.cat([self.terrain.network_terrain, wind])
            nn_output = self.run_prediction(input)
            if self.flag.test_simulated_data and self.flag.rescale_prediction:  # only for simulated trajectory
                if i == 0:
                    self.rescale_prediction(nn_output, self.labels)  # rescale labels only once
                else:
                    self.rescale_prediction(nn_output)
            # loss_org = self.evaluate_loss(output)
            # print(' loss: ', loss_org.item())

            # interpolate prediction to scattered flight data locations corresponding to the labels
            FlightInterpolator = UlogInterpolation(input_flight_data, terrain=self.terrain,
                                                   wind_data_for_prediction=label_flight_data)
            interpolated_output = FlightInterpolator.interpolate_log_data_from_grid(nn_output)
            interpolated_nn_output = np.column_stack(interpolated_output)

            # interpolated_output = np.resize(np.array(interpolated_output), (3, len(interpolated_output[0])))

            # --- Zero wind prediction---
            interpolated_zero_wind_output = np.zeros_like(labels)

            # --- Average wind prediction ---
            average_wind_input = np.array([input_flight_data['wn'], input_flight_data['we'], input_flight_data['wd']])
            interpolated_average_wind_output = np.ones_like(labels) * [average_wind_input[0].mean(),
                                                                       average_wind_input[1].mean(),
                                                                       average_wind_input[2].mean()]
            average_wind_output = torch.ones_like(nn_output)
            average_wind_output[0, :] *= average_wind_input[0].mean()
            average_wind_output[1, :] *= average_wind_input[1].mean()
            average_wind_output[2, :] *= average_wind_input[2].mean()

            # --- Optimized corner prediction ---
            if self.flag.use_optimized_corners:
                optimiser = OptTest(torch.optim.Adagrad, {'lr': 1.0, 'lr_decay': 0.1})
                n_steps = self._optimisation_args.params['num_of_optimisation_steps']
                corners_output, _, _, _, _ \
                    = self.optimise_wind_corners(optimiser.opt, n=n_steps, opt_kwargs=optimiser.kwargs,
                                                 wind_zeros=wind_zeros, wind_mask=wind_mask,
                                                 print_steps=False, verbose=False)
                # interpolate prediction to scattered flight data locations corresponding to the labels
                FlightInterpolator = UlogInterpolation(input_flight_data, terrain=self.terrain,
                                                       wind_data_for_prediction=label_flight_data)
                interpolated_corners = FlightInterpolator.interpolate_log_data_from_grid(corners_output)
                interpolated_corners_output = np.column_stack(interpolated_corners)

            # --- Interpolation (Krigging) prediction ---
            if self.flag.use_krigging_prediction:
                FlightInterpolator = UlogInterpolation(input_flight_data, terrain=self.terrain, predict=True,
                                                       wind_data_for_prediction=label_flight_data)
                _, _, predicted_output = FlightInterpolator.interpolate_log_data_krigging()
                interpolated_krigging_output = np.column_stack(predicted_output)

            # --- Interpolation (Gaussian Process Regression) prediction ---
            if self.flag.use_gpr_prediction:
                FlightInterpolator = UlogInterpolation(input_flight_data, terrain=self.terrain, predict=True,
                                                       wind_data_for_prediction=label_flight_data)
                _, _, predicted_output = FlightInterpolator.interpolate_log_data_gpr()
                interpolated_gpr_output = np.column_stack(predicted_output)

            # losses
            nn_loss = self._loss_fn(torch.Tensor(interpolated_nn_output).to(self._device),
                                 torch.Tensor(labels).to(self._device))
            loss_axis = True
            if loss_axis:
                nn_loss_x = self._loss_fn(torch.Tensor(interpolated_nn_output[:, 0]).to(self._device),
                                        torch.Tensor(labels[:, 0]).to(self._device))
                nn_loss_y = self._loss_fn(torch.Tensor(interpolated_nn_output[:, 1]).to(self._device),
                                        torch.Tensor(labels[:, 1]).to(self._device))
                nn_loss_z = self._loss_fn(torch.Tensor(interpolated_nn_output[:, 2]).to(self._device),
                                        torch.Tensor(labels[:, 2]).to(self._device))
            zero_wind_loss = self._loss_fn(torch.Tensor(interpolated_zero_wind_output).to(self._device),
                                 torch.Tensor(labels).to(self._device))
            average_wind_loss = self._loss_fn(torch.Tensor(interpolated_average_wind_output).to(self._device),
                                 torch.Tensor(labels).to(self._device))
            if self.flag.use_krigging_prediction:
                krigging_wind_loss = self._loss_fn(torch.Tensor(interpolated_krigging_output).to(self._device),
                                     torch.Tensor(labels).to(self._device))
            if self.flag.use_gpr_prediction:
                gpr_wind_loss = self._loss_fn(torch.Tensor(interpolated_gpr_output).to(self._device),
                                     torch.Tensor(labels).to(self._device))
            if self.flag.use_optimized_corners:
                optimized_corners_loss = self._loss_fn(torch.Tensor(interpolated_corners_output).to(self._device),
                                     torch.Tensor(labels).to(self._device))
            print(i, ' NN loss is: ', nn_loss.item())
            if loss_axis:
                print(i, ' NN loss x is: ', nn_loss_x.item())
                print(i, ' NN loss y is: ', nn_loss_y.item())
                print(i, ' NN loss z is: ', nn_loss_z.item())
            print(i, ' Zero wind loss is: ', zero_wind_loss.item())
            print(i, ' Average wind loss is: ', average_wind_loss.item())
            if self.flag.use_krigging_prediction:
                print(i, ' Krigging wind loss is: ', krigging_wind_loss.item())
            if self.flag.use_gpr_prediction:
                print(i, ' GPR wind loss is: ', gpr_wind_loss.item())
            if self.flag.use_optimized_corners:
                print(i, ' Optimized corners wind loss is: ', optimized_corners_loss.item())

            # Average wind direction and speed
            print_direction_and_speed = False
            if print_direction_and_speed:
                # input
                input_wind_abs = np.sqrt(input_flight_data['wn']**2 + input_flight_data['we']**2)
                input_wind_abs_3d = np.sqrt(input_flight_data['wn']**2 + input_flight_data['we']**2
                                              + input_flight_data['wd']**2)
                input_wind_dir_trig_to = np.arctan2(input_flight_data['wn'] / input_wind_abs,
                                                  input_flight_data['we'] / input_wind_abs)
                input_wind_dir_trig_to_degrees = input_wind_dir_trig_to * 180 / np.pi
                input_wind_dir_cardinal = input_wind_dir_trig_to_degrees + 180
                print('Average input wind speed: ', input_wind_abs_3d.mean())
                print('Average input wind direction: ', input_wind_dir_cardinal.mean())
                # label
                label_wind_abs = np.sqrt(label_flight_data['wn']**2 + label_flight_data['we']**2)
                label_wind_abs_3d = np.sqrt(label_flight_data['wn']**2 + label_flight_data['we']**2
                                            + label_flight_data['wd']**2)
                label_wind_dir_trig_to = np.arctan2(label_flight_data['wn'] / label_wind_abs,
                                                  label_flight_data['we'] / label_wind_abs)
                label_wind_dir_trig_to_degrees = label_wind_dir_trig_to * 180 / np.pi
                label_wind_dir_cardinal = label_wind_dir_trig_to_degrees + 180
                print('Average label wind speed: ', label_wind_abs_3d.mean())
                print('Average label wind direction: ', label_wind_dir_cardinal.mean())
                # NN
                nn_wind_abs = np.sqrt(interpolated_nn_output[:, 0]**2 + interpolated_nn_output[:, 1]**2)
                nn_wind_abs_3d = np.sqrt(interpolated_nn_output[:, 0] ** 2
                                         + interpolated_nn_output[:, 1] ** 2
                                         + interpolated_nn_output[:, 2] ** 2)
                nn_wind_dir_trig_to = np.arctan2(interpolated_nn_output[:, 0] / nn_wind_abs,
                                                  nn_output[:, 1] / nn_wind_abs)
                nn_wind_dir_trig_to_degrees = nn_wind_dir_trig_to * 180 / np.pi
                nn_wind_dir_cardinal = nn_wind_dir_trig_to_degrees + 180
                print('Average NN wind speed: ', nn_wind_abs_3d.mean())
                print('Average NN wind direction: ', nn_wind_dir_cardinal.mean())
                # Average
                average_wind_abs = np.sqrt(interpolated_average_wind_output[:, 0]**2
                                           + interpolated_average_wind_output[:, 1]**2)
                average_wind_abs_3d = np.sqrt(interpolated_average_wind_output[:, 0] ** 2
                                              + interpolated_average_wind_output[:, 1] ** 2
                                              + interpolated_average_wind_output[:, 2] ** 2)
                average_wind_dir_trig_to = np.arctan2(interpolated_average_wind_output[:, 0] / average_wind_abs,
                                                  interpolated_average_wind_output[:, 1] / average_wind_abs)
                average_wind_dir_trig_to_degrees = average_wind_dir_trig_to * 180 / np.pi
                average_wind_dir_cardinal = average_wind_dir_trig_to_degrees + 180
                print('Average  wind average speed: ', average_wind_abs_3d.mean())
                print('Average  wind average direction: ', average_wind_dir_cardinal.mean())

            # outputs
            inputs.append(input)
            outputs.append(nn_output)
            nn_losses.append(nn_loss)
            zero_wind_losses.append(zero_wind_loss)
            average_wind_losses.append(average_wind_loss)
            if self.flag.use_krigging_prediction:
                krigging_wind_losses.append(krigging_wind_loss)
            if self.flag.use_gpr_prediction:
                gpr_wind_losses.append(gpr_wind_loss)
            if self.flag.use_optimized_corners:
                optimized_corners_losses.append(optimized_corners_loss)

        print('')
        losses_items = [nn_losses[i].item() for i in range(len(nn_losses))]
        average_loss = sum(losses_items)/len(losses_items)
        print('Average NN loss is: ', average_loss)
        losses_items = [zero_wind_losses[i].item() for i in range(len(zero_wind_losses))]
        average_loss = sum(losses_items)/len(losses_items)
        print('Average zero wind loss is: ', average_loss)
        losses_items = [average_wind_losses[i].item() for i in range(len(average_wind_losses))]
        average_loss = sum(losses_items)/len(losses_items)
        print('Average wind average loss is: ', average_loss)
        if self.flag.use_krigging_prediction:
            losses_items = [krigging_wind_losses[i].item() for i in range(len(krigging_wind_losses))]
            average_loss = sum(losses_items)/len(losses_items)
            print('Krigging wind average loss is: ', average_loss)
        if self.flag.use_gpr_prediction:
            losses_items = [gpr_wind_losses[i].item() for i in range(len(gpr_wind_losses))]
            average_loss = sum(losses_items)/len(losses_items)
            print('GPR wind average loss is: ', average_loss)
        if self.flag.use_optimized_corners:
            losses_items = [optimized_corners_losses[i].item() for i in range(len(optimized_corners_losses))]
            average_loss = sum(losses_items)/len(losses_items)
            print('Optimized corners average loss is: ', average_loss)

        return outputs, nn_losses, inputs
