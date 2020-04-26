import numpy as np
import sys
sys.path.append('../wind_prediction/')
import nn_wind_prediction.utils as utils
import nn_wind_prediction.models as models
from analysis_utils import extract_cosmo_data as cosmo
from analysis_utils import ulog_utils, get_mapgeo_terrain
from analysis_utils import generate_turbulence
from analysis_utils.interpolate_flight_data import FlightInterpolation
from analysis_utils.generate_trajectory import generate_trajectory
from analysis_utils.interpolate_wind_profile import interpolate_wind_profile
from nn_wind_prediction.utils.interpolation import DataInterpolation
import nn_wind_prediction.data as data
import nn_wind_prediction.data as nn_data
from datetime import datetime
from scipy import ndimage
from scipy.spatial import distance
from scipy.interpolate import RegularGridInterpolator as RGI
import random
import torch
import os
import time
import copy
import math
import warnings


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
    def __init__(self, test, cfd, flight, wind, noise, window_split, optimisation, model):
        self.test_simulated_data = test.params['test_simulated_data']
        self.test_flight_data = test.params['test_flight_data']
        self.print_names = cfd.params['print_names']
        self.use_ekf_wind = flight.params['use_ekf_wind']
        self.use_sparse_data = wind.params['sparse_data']['use_sparse_data']
        self.terrain_percentage_correction = wind.params['sparse_data']['terrain_percentage_correction']
        self.sample_random_region = wind.params['sparse_data']['sample_random_region']
        self.use_trajectory = wind.params['trajectory']['use_trajectory_generation']
        self.use_simulated_trajectory = wind.params['trajectory']['use_simulated_trajectory']
        self.predict_flight = wind.params['flight']['predict_flight']
        self.add_turbulence = noise.params['add_turbulence']
        self.add_gaussian_noise = noise.params['add_gaussian_noise']
        self.use_window_split = window_split.params['use_window_split']
        self.batch_test = window_split.params['batch_test']
        self.incremental_input_prediction = window_split.params['incremental_input_prediction']
        self.sliding_input_prediction = window_split.params['sliding_input_prediction']
        self.normalize_losses = window_split.params['normalize_losses']
        self.use_average_wind = window_split.params['use_average_wind']
        self.use_optimized_corners = window_split.params['use_optimized_corners']
        self.use_optimized_corners_polynomial = window_split.params['use_optimized_corners_polynomial']
        self.use_gpr_prediction = window_split.params['use_gpr_prediction']
        self.use_krigging_prediction = window_split.params['use_krigging_prediction']
        self.rescale_prediction = window_split.params['rescale_prediction']
        self.optimise_corners_individually = optimisation.params['optimise_corners_individually']
        self.use_hybrid_model = model.params['hybrid_model']['use_hybrid_model']


class WindOptimiser(object):

    def __init__(self, config_yaml, resolution=64):
        # Configuration variables
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self._device = torch.device("cpu")
        self._config_yaml = config_yaml
        self._test_args = utils.BasicParameters(self._config_yaml, 'test')
        self._cfd_args = utils.BasicParameters(self._config_yaml, 'cfd')
        self._cosmo_args = utils.COSMOParameters(self._config_yaml)
        self._flight_args = utils.FlightParameters(self._config_yaml)
        self._wind_args = utils.BasicParameters(self._config_yaml, 'wind')
        self._noise_args = utils.BasicParameters(self._config_yaml, 'noise')
        self._window_splits_args = utils.BasicParameters(self._config_yaml, 'window_split')
        self._corner_optimisation_args = utils.BasicParameters(self._config_yaml, 'corner_optimisation')
        self._model_args = utils.BasicParameters(self._config_yaml, 'model')
        self._resolution = resolution

        # Selection flags
        self.flag = SelectionFlags(self._test_args, self._cfd_args, self._flight_args, self._wind_args,
                                   self._noise_args, self._window_splits_args, self._corner_optimisation_args,
                                   self._model_args)

        # Network model and loss function
        self.net, self.params = self.load_network_model()
        self._loss_fn = self.get_loss_function()
        # custom loss functions
        self._loss_fn_mae = torch.nn.L1Loss()
        self._loss_fn_mse = torch.nn.MSELoss()

        # Load data
        if self.flag.test_simulated_data:
            self.original_input, self.labels, self.scale, self.data_set_file_name, self.grid_size, \
                self.original_terrain, self.test_set_length \
                = self.load_data_set()
            # Noise
            self.noisy_labels = self.labels.clone()
            if self.flag.add_gaussian_noise:
                self.noisy_labels = self.add_gaussian_noise(self.noisy_labels)
            if self.flag.add_turbulence:
                self.noisy_labels = self.add_turbulence(self.noisy_labels)
            # self._optimized_corners = self.get_optimized_corners()
            self.terrain = self.get_cfd_terrain()
            self._base_cfd_corners = self.get_cfd_corners()
        if self.flag.test_flight_data:
            self._flight_data = self.load_flight_data()
            self._cosmo_wind = self.load_wind()
            self.terrain = self.load_cosmo_terrain()
            if len(self._cosmo_wind) > 1:
                self._base_cosmo_corners = self.get_cosmo_corners()

        # Noise
        if self.flag.test_simulated_data:
            self.noisy_labels = self.labels.clone()
        if self.flag.test_simulated_data and self.flag.add_gaussian_noise:
            self.noisy_labels = self.add_gaussian_noise(self.noisy_labels)
        if self.flag.test_simulated_data and self.flag.add_turbulence:
            self.noisy_labels = self.add_turbulence(self.noisy_labels)

        # Wind data variables
        if self.flag.test_simulated_data:
            if self.flag.use_sparse_data:
                self._wind_input = self.get_sparse_wind_blocks()
            if self.flag.use_trajectory:
                self._wind_input, self._simulated_flight_data = self.get_trajectory_wind_blocks()
        if self.flag.test_flight_data:
            input_flight_data = self._flight_data
            # input_flight_data, _ = self.sliding_window_split(self._flight_data, 300, 1, 600)
            self._wind_blocks, self._var_blocks, self._wind_zeros, self._wind_mask \
                = self.get_flight_wind_blocks(input_flight_data)
            # TODO: replace if there are actual labels for flight data
            self.labels = self._wind_zeros

        # Angles
        # self.wind_vector_angles, self.input_angles, self.output_angles = self.get_angles_between_wind_vectors()

        # Optimisation variables
        self._optimisation_variables = self.get_optimisation_variables()
        self.reset_optimisation_variables()
        self._interpolator = DataInterpolation(self._device, 3, *self.terrain.get_dimensions())

    # --- Network and loss function ---

    def load_network_model(self):
        networks, parameters = [], []
        for i in range(len(self._model_args.params['name'])):
            model_args = self._model_args
            print('Loading network model...', end='', flush=True)
            t_start = time.time()
            yaml_loc = os.path.join(model_args.params['location'], model_args.params['name'][i], 'params.yaml')
            params = utils.EDNNParameters(yaml_loc)  # load the model config
            NetworkType = getattr(models, params.model['model_type'])  # load the model
            net = NetworkType(**params.model_kwargs())  # load learnt parameters
            model_loc = os.path.join(model_args.params['location'], model_args.params['name'][i],
                                     model_args.params['version'][i] + '.model')
            # load params
            net.load_state_dict(torch.load(model_loc, map_location=lambda storage, loc: storage))
            net = net.to(self._device)
            net.eval()
            print(' done [{:.2f} s]'.format(time.time() - t_start))
            networks.append(net)
            parameters.append(params)
        return networks, parameters

    def get_loss_function(self, index=0):
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

    def load_flight_data(self, index=None, flight_dir=None):
        self._flight_args.print()
        if index is not None:
            file_index = index
        else:
            file_index = self._flight_args.params['index']
        if flight_dir is not None:
            flight_dir = flight_dir
        else:
            flight_dir = self._flight_args.params['flight_data_dir']
        file_name = self._flight_args.params['files'][file_index]
        file_and_dir = flight_dir + file_name
        extension = file_name.split('.')[-1]
        if extension == 'ulg':
            print('Loading ulog flight data...', end='', flush=True)
            t_start = time.time()
            ulog_data = ulog_utils.get_log_data(file_and_dir)

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
            hdf5_data = ulog_utils.read_filtered_hdf5(file_and_dir, skip_amount=10)
            hdf5_data['wn'] = hdf5_data['wind_n']
            hdf5_data['we'] = hdf5_data['wind_e']
            hdf5_data['wd'] = hdf5_data['wind_d']
            hdf5_data['time_microsec'] = hdf5_data['time']

            flight_data = hdf5_data
            print(' done [{:.2f} s]'.format(time.time() - t_start))
        else:
            raise ValueError('Unknown file type: ' + extension + '. ' + file_name + ' should have a .ulg or .hdf5 extension')

        return flight_data

    def load_wind(self, cosmo_file=None, index=None):
        if cosmo_file is not None:
            cosmo_file = cosmo_file
        else:
            cosmo_file = self._cosmo_args.params['file']

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
                file_name = self._flight_args.params['files'][self._flight_args.params['index']]
                time_of_flight = int(file_name.split('/')[-1].split('_')[0])
            except ValueError:
                print('Cannot extract hour of flight from file name')
        # Get cosmo wind
        # TODO: temporary due to lack of utc data in filtered flight logs
        if index is not None:
            index = index
        else:
            index = self._flight_args.params['index']
        if '20181123_flight01' in self._flight_args.params['files'][index]:
            time_of_flight = 10
        elif '20181123_flight02' in self._flight_args.params['files'][index]:
            time_of_flight = 11
        elif '20181123_flight03' in self._flight_args.params['files'][index]:
            time_of_flight = 13
        elif '20181123_flight04' in self._flight_args.params['files'][index]:
            time_of_flight = 13
        elif '20181123_flight05' in self._flight_args.params['files'][index]:
            time_of_flight = 14
        elif '20181123_flight06' in self._flight_args.params['files'][index]:
            time_of_flight = 14

        if self.flag.batch_test:
            # TODO: workaround to enable batch test
            if 'cosmo-1_ethz_fcst_2018112309' in cosmo_file:
                offset_cosmo_time = time_of_flight - 9
            elif 'cosmo-1_ethz_fcst_2018112312' in cosmo_file:
                offset_cosmo_time = time_of_flight - 12
        else:
            offset_cosmo_time = self._cosmo_args.get_cosmo_time(time_of_flight)

        cosmo_wind = cosmo.extract_cosmo_data(cosmo_file, lat0, lon0, offset_cosmo_time,
                                              terrain_file=self._cosmo_args.params['terrain_file'])
        print(' done [{:.2f} s]'.format(time.time() - t_start))
        return cosmo_wind

    def load_cosmo_terrain(self, terrain_tiff=None):
        if terrain_tiff is not None:
            terrain_tiff = terrain_tiff
        else:
            terrain_tiff = self._cosmo_args.params['terrain_tiff']

        print('Loading terrain...', end='', flush=True)
        t_start = time.time()
        # Get corresponding terrain
        # min_height = min(flight_data['alt'].min(), h_terr.min())
        block_height = [1100.0 / 95 * 63]
        # x_terr, y_terr and z_terr are the (regular, monotonic) index arrays for the h_terr and full_block arrays
        # h_terr is the terrain height
        boolean_terrain = self._model_args.params['boolean_terrain']
        if len(self._cosmo_wind) == 1:
            # print warning since we are creating predefined terrain limits
            warnings.warn('Warning!: no cosmo data for selected flight')

            x_max_distance = self._flight_data['x'].max() - self._flight_data['x'].min()
            if x_max_distance > 1100:
                x = [self._flight_data['x'].min(), self._flight_data['x'].min()]
            else:
                x = [self._flight_data['x'].min() - 200, self._flight_data['x'].min() + 900]
            y_max_distance = self._flight_data['y'].max() - self._flight_data['y'].min()
            if y_max_distance > 1100:
                y = [self._flight_data['y'].min(), self._flight_data['y'].min()]
            else:
                y = [self._flight_data['y'].min() - 300, self._flight_data['y'].min() + 800]
            terrain = TerrainBlock(
                *get_mapgeo_terrain.get_terrain(terrain_tiff, x, y, block_height,
                                                (self._resolution, self._resolution, self._resolution)),
                device=self._device, boolean_terrain=boolean_terrain)
        else:
            terrain = TerrainBlock(
                *get_mapgeo_terrain.get_terrain(terrain_tiff, self._cosmo_wind['x'][[0, 1], [0, 1]],
                                                self._cosmo_wind['y'][[0, 1], [0, 1]], block_height,
                                                (self._resolution, self._resolution, self._resolution)),
                device=self._device, boolean_terrain=boolean_terrain)
        print(' done [{:.2f} s]'.format(time.time() - t_start))
        return terrain

    def load_data_set(self, index=None):
        # Load data set
        test_set = nn_data.HDF5Dataset(self._cfd_args.params['testset_name'],
                                       augmentation=False, return_grid_size=True,
                                       **self.params[0].Dataset_kwargs())
        test_set_length = len(test_set)
        # print names of files in test_set
        if self.flag.print_names:
            for i in range(test_set_length):
                name = test_set.get_name(i)
                print(i, name)
        # Get data set from test set
        if index is not None:
            file_index = index
        else:
            file_index = self._cfd_args.params['index']
        data_set = test_set[file_index]
        input = data_set[0]
        labels = data_set[1]
        name = test_set.get_name(file_index)
        scale = 1.0
        if self.params[0].data['autoscale']:
            scale = data_set[3].item()
        print("Loading test set (index) with name: ", str(name))
        grid_size = data.get_grid_size(self._cfd_args.params['testset_name'])
        terrain = data_set[0][0, :, :, :]
        return input, labels, scale, name, grid_size, terrain, test_set_length

    def get_cfd_terrain(self):
        terrain = self.original_terrain.clone()
        binary_terrain = terrain <= 0
        nz, ny, nx = terrain.shape
        x_terr = np.asarray([self.grid_size[0]*i for i in range(nx)])
        y_terr = np.asarray([self.grid_size[1]*i for i in range(ny)])
        z_terr = np.asarray([self.grid_size[2]*i for i in range(nz)])
        h_terr = (binary_terrain.sum(0, keepdim=True)*self.grid_size[2]).squeeze(0).detach().cpu().numpy()
        h_terr = np.clip(h_terr, 0, z_terr[-1])  # make sure height is not bigger than max z_terr
        # Get terrain object
        boolean_terrain = self._model_args.params['boolean_terrain']
        terrain = TerrainBlock(x_terr, y_terr, z_terr, h_terr, binary_terrain,
                               device=self._device, boolean_terrain=boolean_terrain)
        return terrain

    # --- Add noise to data ---

    def add_gaussian_noise(self, labels):
        eps = 1
        noise = eps * torch.rand(labels.shape)
        # apply scale
        noise /= self.scale
        labels += noise
        return labels

    def add_turbulence(self, labels):
        # get turbulence field
        turbulence, _ = generate_turbulence.generate_turbulence_spectral()
        _, turb_nx, turb_ny, turb_nz = turbulence.shape
        # subsample turbulent velocity field with the same shape as labels
        _, nz, ny, nx = labels.shape
        start_x = np.random.randint(0, turb_nx - nx)
        start_y = np.random.randint(0, turb_ny - ny)
        start_z = np.random.randint(0, turb_nz - nz)  # triangle distribution
        turbulence = \
            turbulence[:, start_z:start_z + nz, start_y:start_y + ny, start_x:start_x + nx]
        # apply scale
        turbulence /= self.scale
        eps = 1
        turbulence = eps * torch.from_numpy(turbulence).float()
        labels += turbulence
        return labels

    # --- Create wind data ---

    def get_sparse_wind_blocks(self, p=0):
        time_start = time.time()
        wind = self.noisy_labels.clone()
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

        augmented_wind = torch.cat([wind, mask.float().unsqueeze(0)])
        # print('Time to create sparse mask: ', time.time()-time_start)
        return augmented_wind.to(self._device)

    def get_trajectory_wind_blocks(self):
        # copy labels
        wind = self.noisy_labels.clone()

        # copy terrain
        terrain = self.original_terrain.clone()

        # define validity mask (1s where there is wind data 0s everywhere else)
        mask = torch.zeros_like(terrain)

        # initialize trajectory list
        sequential_input = []

        # initialize simulated flight data
        simulated_flight_data = {}

        # spiral trajectory
        if 'spiral' in self._wind_args.params['trajectory']['trajectory_type']:
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
            num_segments, segment_length = x_traj.shape
            counter = 0
            for i in range(num_segments):
                for j in range(segment_length):
                    id_x = (int((x_traj[i][j] - self.terrain.x_terr[0]) / self.grid_size[0]))
                    id_y = (int((y_traj[i][j] - self.terrain.y_terr[0]) / self.grid_size[1]))
                    id_z = (int((z_traj[i][j] - self.terrain.z_terr[0]) / self.grid_size[2]))
                    if all(v == 0 for v in wind[:, id_z, id_y, id_x]):
                        counter += 1
                    mask[id_z, id_y, id_x] = 1.0

        # random trajectory
        if 'random' in self._wind_args.params['trajectory']['trajectory_type']:
            time_start = time.time()
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
            time_inter = time.time()
            x_traj, y_traj, z_traj = generate_trajectory(x, y, z, uav_speed, dt, self.terrain, z_above_terrain=True)
            time_inter2 = time.time()

            # Get the bins and wind along the trajectory
            for i in range(x_traj.size):
                    id_x = (int((x_traj[i] - self.terrain.x_terr[0]) / self.grid_size[0]))
                    id_y = (int((y_traj[i] - self.terrain.y_terr[0]) / self.grid_size[1]))
                    id_z = (int((z_traj[i] - self.terrain.z_terr[0]) / self.grid_size[2]))
                    mask[id_z, id_y, id_x] = 1.0

            # generate simulated flight data
            interpolating_function_x = RGI((self.terrain.z_terr, self.terrain.y_terr, self.terrain.x_terr),
                                           wind[0, :].detach().cpu().numpy(), method='nearest')
            interpolating_function_y = RGI((self.terrain.z_terr, self.terrain.y_terr, self.terrain.x_terr),
                                           wind[1, :].detach().cpu().numpy(), method='nearest')
            interpolating_function_z = RGI((self.terrain.z_terr, self.terrain.y_terr, self.terrain.x_terr),
                                           wind[2, :].detach().cpu().numpy(), method='nearest')
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

        # segments trajectory
        if 'segments' in self._wind_args.params['trajectory']['trajectory_type']:
            time_start = time.time()
            # boolean terrain
            boolean_terrain = (self.original_terrain.clone() <= 0).float()  # true where there is terrain
            # network terrain height
            h_terrain = boolean_terrain.sum(0, keepdim=True).squeeze(0)
            # random starting point
            non_zero = torch.nonzero(terrain)
            random.seed(7)
            start = non_zero[random.randint(0, non_zero.shape[0]-1)]
            idx_start = start[2].detach().cpu().numpy()
            idy_start = start[1].detach().cpu().numpy()
            idz_start = start[0].detach().cpu().numpy()
            # number of bins along direction
            dir_1 = 10
            dir_2 = 2
            # Random number of segments, each with a length between 10 and 11 bins
            # num_of_segments = random.randint(2, 20)
            num_of_segments = 60
            # first axis to go along
            direction_axis = 1
            forward_axis = 'x'
            # trajectory bin points
            trajectory_bin_points = []
            # trajectory points
            trajectory_points = []
            for i in range(num_of_segments):
                feasible_points = []
                # current point
                current_idx = idx_start
                current_idy = idy_start
                current_idz = idz_start
                # find feasible next points
                if 'x' in forward_axis:
                    for m in range(-2, 3, 4):
                        for n in range(0, 3, 1):
                            for o in range(-1, 2, 2):
                                if (0 <= current_idx + o*dir_1 < 64 and
                                    0 <= current_idy + direction_axis * n * dir_2 < 64 and
                                    h_terrain[current_idy + direction_axis*n*dir_2, current_idx + o*dir_1]
                                        <= current_idz + m < 64):
                                    feasible_points.append(
                                        [current_idz + m, current_idy + direction_axis*n*dir_2, current_idx + o*dir_1])
                else:
                    for m in range(-2, 3, 4):
                        for n in range(0, 3, 1):
                            for o in range(-1, 2, 2):
                                if (0 <= current_idy + o*dir_1 < 64 and
                                    0 <= current_idx + direction_axis * n * dir_2 < 64 and
                                    h_terrain[current_idy + o*dir_1, current_idx + direction_axis*n*dir_2]
                                        <= current_idz + m < 64):
                                    feasible_points.append(
                                        [current_idz + m, current_idy + o*dir_1, current_idx + direction_axis*n*dir_2])
                # randomly choose next point from the feasible points
                if len(feasible_points) == 0:
                    # stop if no feasible point was found
                    print('No feasible next point found at iteration: ', i)
                    mask[current_idz, current_idy, current_idx] = 1.0
                    trajectory_bin_points.append([current_idz, current_idy, current_idx])
                    break
                else:
                    # time_inter = time.time()
                    random.seed(7)
                    next_feasible_point = feasible_points[random.randint(0, len(feasible_points)-1)]
                    next_idx = next_feasible_point[2]
                    next_idy = next_feasible_point[1]
                    next_idz = next_feasible_point[0]

                    # bins along trajectory
                    points_along_traj = 12
                    n = 1
                    for j in range(0, points_along_traj):
                        t = n / (points_along_traj + 1)
                        traj_x = current_idx + t * (next_idx - current_idx)
                        traj_y = current_idy + t * (next_idy - current_idy)
                        traj_z = current_idz + t * (next_idz - current_idz)
                        trajectory_points.append([traj_z, traj_y, traj_x])
                        id_x = int(traj_x)
                        id_y = int(traj_y)
                        id_z = int(traj_z)
                        if any(self.flag.use_hybrid_model) and mask[id_z, id_y, id_x] != 1.0:
                            # append bin points only once
                            trajectory_bin_points.append([id_z, id_y, id_x])
                        mask[id_z, id_y, id_x] = 1.0
                        n += 1

                # prepare next iteration
                idx_start = next_idx
                idy_start = next_idy
                idz_start = next_idz
                if 'x' in forward_axis:
                    forward_axis = 'y'
                    if next_idx < current_idx:
                        direction_axis = -direction_axis
                    else:
                        direction_axis = direction_axis
                else:
                    forward_axis = 'x'
                    if next_idy < current_idy:
                        direction_axis = -direction_axis
                    else:
                        direction_axis = direction_axis
            # print('Time to create feasible points:', time_inter - time_start)
            if any(self.flag.use_hybrid_model):
                if len(trajectory_bin_points) == 1:
                    # in case no feasible trajectory points were found make sure there is at least one measurement
                    # (the starting point) for each mask in each input sequence
                    sequence_length = self._model_args.params['hybrid_model']['sequence_length']
                    for i in range(sequence_length):
                        mask_segment = torch.zeros_like(terrain)
                        mask_segment[
                            trajectory_bin_points[0][0], trajectory_bin_points[0][1], trajectory_bin_points[0][2]] = 1.0
                        sequential_input.append(torch.cat([wind, mask_segment.float().unsqueeze(0)]))
                else:
                    # divide trajectory into sequences
                    sequence_length = self._model_args.params['hybrid_model']['sequence_length']
                    traj_seg_len = int((len(trajectory_bin_points) / sequence_length) + 1)
                    for i in range(sequence_length):
                        if i == 0:
                            trajectory_segment = torch.from_numpy(np.asarray(trajectory_bin_points[:traj_seg_len]))
                        elif i == sequence_length-1:
                            trajectory_segment = torch.from_numpy(np.asarray(trajectory_bin_points[((sequence_length-1)*traj_seg_len):]))
                        else:
                            trajectory_segment = torch.from_numpy(np.asarray(trajectory_bin_points[((i-1)*traj_seg_len):(i*traj_seg_len)]))
                        mask_segment = torch.zeros_like(terrain)
                        mask_segment[trajectory_segment.split(1, dim=1)] = 1.0
                        sequential_input.append(torch.cat([wind, mask_segment.float().unsqueeze(0)]))
            print('Time to create traj:', time.time()-time_start)

            # generate simulated trajectory
            dt = self._wind_args.params['trajectory']['dt']
            trajectory_points = np.asarray(trajectory_points)
            trajectory_points[:, 0] = trajectory_points[:, 0] * self.grid_size[2]
            trajectory_points[:, 1] = trajectory_points[:, 1] * self.grid_size[1]
            trajectory_points[:, 2] = trajectory_points[:, 2] * self.grid_size[0]
            interpolating_function_x = RGI((self.terrain.z_terr, self.terrain.y_terr, self.terrain.x_terr),
                                           wind[0, :].detach().cpu().numpy(), method='nearest')
            interpolating_function_y = RGI((self.terrain.z_terr, self.terrain.y_terr, self.terrain.x_terr),
                                           wind[1, :].detach().cpu().numpy(), method='nearest')
            interpolating_function_z = RGI((self.terrain.z_terr, self.terrain.y_terr, self.terrain.x_terr),
                                           wind[2, :].detach().cpu().numpy(), method='nearest')

            interpolated_flight_data_x = interpolating_function_x(trajectory_points)
            interpolated_flight_data_y = interpolating_function_y(trajectory_points)
            interpolated_flight_data_z = interpolating_function_z(trajectory_points)
            simulated_flight_data['x'] = trajectory_points[:, 2] - 0.5 * self.grid_size[0]
            simulated_flight_data['y'] = trajectory_points[:, 1] - 0.5 * self.grid_size[1]
            simulated_flight_data['alt'] = trajectory_points[:, 0] - 0.5 * self.grid_size[2]
            simulated_flight_data['wn'] = interpolated_flight_data_x.astype(float)
            simulated_flight_data['we'] = interpolated_flight_data_y.astype(float)
            simulated_flight_data['wd'] = -interpolated_flight_data_z.astype(float)
            simulated_flight_data['time_microsec'] = np.array([i*dt*1e6 for i in range(trajectory_points.shape[0])])

        # print('Time to create points:', time_inter - time_start)
        # print('Time to generate trajectory points:', time_inter2 - time_inter)
        # print('Time to bin points:', time.time() - time_inter)
        # print('Time to generate trajectory:', time.time()-time_start)
        if self.flag.use_hybrid_model[0]:
            augmented_wind = torch.stack(sequential_input, dim=0)
        else:
            augmented_wind = torch.cat([wind, mask.float().unsqueeze(0)])
        return augmented_wind.to(self._device), simulated_flight_data

    def get_flight_wind_blocks(self, flight_data):
        #print('Getting binned wind blocks...', end='', flush=True)
        #t_start = time.time()

        # Determine the grid dimensions
        dx = self.terrain.x_terr[[0, -1]]; ddx = (self.terrain.x_terr[1]-self.terrain.x_terr[0])/2.0
        dy = self.terrain.y_terr[[0, -1]]; ddy = (self.terrain.y_terr[1]-self.terrain.y_terr[0])/2.0
        dz = self.terrain.z_terr[[0, -1]]; ddz = (self.terrain.z_terr[1]-self.terrain.z_terr[0])/2.0
        grid_dimensions = {'x_min': dx[0] - ddx, 'x_max': dx[1] + ddx, 'y_min': dy[0] - ddy, 'y_max': dy[1] + ddy,
                           'z_min': dz[0] - ddz, 'z_max': dz[1] + ddz, 'n_cells': self.terrain.get_dimensions()[0]}

        FlightInterpolator = FlightInterpolation(flight_data, grid_dimensions, self.terrain)

        # bin the data into the regular grid
        try:
            if self._wind_args.params['flight']['interpolation_method'].lower() == 'bin':
                wind, variance = FlightInterpolator.bin_flight_data()
            elif self._wind_args.params['flight']['interpolation_method'].lower() == 'idw':
                wind, variance = FlightInterpolator.interpolate_flight_data_idw()
            elif self._wind_args.params['flight']['interpolation_method'].lower() == 'krigging':
                wind, variance, _ = FlightInterpolator.interpolate_flight_data_krigging()
            elif self._wind_args.params['flight']['interpolation_method'].lower() == 'gpr':
                wind, variance, _ = FlightInterpolator.interpolate_flight_data_gpr()
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
        return wind.to(self._device), variance.to(self._device), wind_zeros.to(self._device), wind_mask.to(self._device)

    # --- Corner optimisation variables ---

    def get_optimisation_variables(self, use_average_wind_norm=False, average_wind_norm=None):
        rotation = self._corner_optimisation_args.params['rotation']
        scale = self._corner_optimisation_args.params['scale']
        if use_average_wind_norm:
            alpha = 1/7
            spline = [average_wind_norm * (i/self._corner_optimisation_args.params['polynomial_law_num_points'])**alpha
                      for i in range(self._corner_optimisation_args.params['polynomial_law_num_points'])]
        else:
            spline = [self._corner_optimisation_args.params['init_wind_speed']
                      for i in range(self._corner_optimisation_args.params['polynomial_law_num_points'])]
        opt_var = [rotation, scale]
        opt_var.extend(spline)
        # optimisation_variables = [var for var in opt_var if var != []]
        # Replicate optimisation variables for each corner
        if self.flag.optimise_corners_individually:
            optimisation_variables = [opt_var[:] for i in range(4)]

        return optimisation_variables

    def reset_optimisation_variables(self, optimisation_variables=None):
        if optimisation_variables is None:
            optimisation_variables = self._optimisation_variables

        self._optimisation_variables = torch.Tensor(optimisation_variables).to(self._device).requires_grad_()

    def get_rotated_polynomial_corners(self):
        sr, cr = [], []
        for xi in range(2):
            for yi in range(2):
                sr.append(torch.sin(self._optimisation_variables[xi*2+yi][0]))
                cr.append(torch.cos(self._optimisation_variables[xi*2+yi][0]))
        corner_winds = torch.zeros((3, 64, 2, 2)).to(self._device)
        wind_z = []
        num_points = self._corner_optimisation_args.params['polynomial_law_num_points']
        for yi in range(2):
            for xi in range(2):
                # wind_speed = self._optimisation_variables[i][0:num_points]
                wind_height_increment = (self.terrain.z_terr[-1] - self.terrain.terrain_corners[yi, xi]) / num_points
                wind_z.append([self.terrain.terrain_corners[yi, xi] + j * wind_height_increment
                               for j in range(num_points)])

        if self.flag.optimise_corners_individually:
            for yi in range(2):
                for xi in range(2):
                    valid_z = self.terrain.z_terr > self.terrain.terrain_corners[yi, xi]
                    if wind_z[xi*2+yi][1] > wind_z[xi*2+yi][0]:
                        interpolated_corners = interpolate_wind_profile(
                            wind_z[xi * 2 + yi],
                            self._optimisation_variables[xi * 2 + yi][-num_points:],
                            self.terrain.z_terr[valid_z], self._device)
                        corner_winds[0, -(valid_z.sum()):, yi, xi] = interpolated_corners * cr[xi*2+yi] - \
                                                     interpolated_corners * sr[xi*2+yi]
                        corner_winds[1, -(valid_z.sum()):, yi, xi] = interpolated_corners * sr[xi*2+yi] + \
                                                     interpolated_corners * cr[xi*2+yi]

        else:
            valid_z = torch.from_numpy((self.terrain.z_terr > self.terrain.terrain_corners[0, 0]).astype(np.uint8))
            if wind_z[1] > wind_z[0]:
                interpolated_corners = interpolate_wind_profile(
                    wind_z,
                    self._optimisation_variables[-num_points:],
                    self.terrain.z_terr[valid_z], self._device)
                corner_winds[0, -(valid_z.sum()):, :] = interpolated_corners * cr - interpolated_corners * sr
                corner_winds[1, -(valid_z.sum()):, :] = interpolated_corners * sr + interpolated_corners * cr

        return corner_winds

    def get_rotated_corners(self):
        sr, cr = [], []
        for i in range(self._optimisation_variables.shape[0]):
            sr.append(torch.sin(self._optimisation_variables[i][0]))
            cr.append(torch.cos(self._optimisation_variables[i][0]))
        # Get corner winds for model inference, offset to actual terrain heights
        if self.flag.test_simulated_data:
            wind_corners = self._base_cfd_corners.clone()
            corners = self._base_cfd_corners.clone()
        if self.flag.test_flight_data:
            wind_corners = self._base_cosmo_corners.clone()
            corners = self._base_cosmo_corners.clone()
        for i in range(self._optimisation_variables.shape[0]):
            wind_corners[0, :, i//2, (i+2)%2] = self._optimisation_variables[i][1]*(
                    corners[0, :, i//2, (i+2)%2]*cr[i]
                    - corners[1, :, i//2, (i+2)%2]*sr[i])
            wind_corners[1, :, i//2, (i+2)%2] = self._optimisation_variables[i][1]*(
                    corners[0, :, i//2, (i+2)%2]*sr[i]
                    + corners[1, :, i//2, (i+2)%2]*cr[i])
        return wind_corners

    def get_cfd_corners(self):
        channels, nx, ny, nz = self.labels.shape
        input = self.noisy_labels.clone()
        corners = input[:, ::nx-1, ::ny-1, :]
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
        if self.flag.test_simulated_data:
            flight_data = self._simulated_flight_data
        elif self.flag.test_flight_data:
            flight_data = self._flight_data
        wind_n = flight_data['wn']
        wind_e = flight_data['we']
        wind_d = flight_data['wd']
        wind_vectors = [[wind_n[i], wind_e[i], wind_d[i]] for i in range(len(wind_e))]
        angles = []
        for i in range(len(wind_e)-1):
            angles.append(angle_between_vectors(wind_vectors[i], wind_vectors[i+1]))
        angles = np.array([x for x in angles if ~np.isnan(x)])

        # angles between input flight and output flight
        input_flight, output_flight = {}, {}
        for keys, values in flight_data.items():
            input_batch = values[0: 1000]
            output_batch = values[0: 1000]
            input_flight.update({keys: input_batch})
            output_flight.update({keys: output_batch})

        wind_blocks, var_blocks, wind_zeros, wind_mask \
            = self.get_flight_wind_blocks(input_flight)
        # get prediction
        wind_input = self.add_mask_to_wind(wind_zeros, wind_mask)
        input = torch.cat([self.terrain.network_terrain, wind_input])
        nn_output = self.run_prediction(input)
        FlightInterpolator = FlightInterpolation(input_flight, terrain=self.terrain,
                                                 wind_data_for_prediction=output_flight)
        interpolated_nn_output = FlightInterpolator.interpolate_flight_data_from_grid(nn_output)

        # input
        input_wind_n = input_flight['wn']
        input_wind_e = input_flight['we']
        input_wind_d = input_flight['wd']
        wind_vectors = [[input_wind_n[i], input_wind_e[i], input_wind_d[i]] for i in range(len(input_wind_e))]
        input_angles = []
        for i in range(len(input_wind_n)-1):
            input_angles.append(angle_between_vectors(wind_vectors[i], wind_vectors[i+1]))
        input_angles = np.array([x for x in input_angles if ~np.isnan(x)])
        # output
        output_wind_n = interpolated_nn_output[:, 0]
        output_wind_e = interpolated_nn_output[:, 1]
        output_wind_d = interpolated_nn_output[:, 2]
        wind_vectors = [[output_wind_n[i], output_wind_e[i], output_wind_d[i]] for i in range(len(output_wind_e))]
        output_angles = []
        for i in range(len(output_wind_n)-1):
            output_angles.append(angle_between_vectors(wind_vectors[i], wind_vectors[i+1]))
        output_angles = np.array([x for x in output_angles if ~np.isnan(x)])

        # losses
        # labels = torch.from_numpy(np.column_stack([output_flight['wn'], output_flight['we'], output_flight['wd']]))
        # nn_loss = self._loss_fn(interpolated_nn_output.to(self._device),
        #                         labels.to(self._device))
        # print('Loss is: ', nn_loss.item())

        return angles, input_angles, output_angles

    def build_csv(self):
        try:
            csv_args = utils.BasicParameters(self._config_yaml, 'csv')
            print('Saving csv to {0}'.format(csv_args.params['file']))
            utils.build_csv(self.terrain.x_terr, self.terrain.y_terr, self.terrain.z_terr, self.terrain.full_block,
                            self.terrain.cosmo_corners, csv_args.params['file'])
        except:
            print('CSV filename parameter (csv:file) not found in {0}, csv not saved'.format(self._config_yaml))

    def run_prediction(self, input, index=0):
        return self.net[index](input.unsqueeze(0))['pred'].squeeze(0)

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
            if self.flag.incremental_input_prediction and not self.flag.sliding_input_prediction:
                input_batch = values[:step_size+window_size]
                output_batch = values[step_size+window_size:]
            elif self.flag.sliding_input_prediction and not self.flag.incremental_input_prediction:
                input_batch = values[step_size:step_size+window_size]
                output_batch = values[step_size+window_size:step_size+window_size+response_size]
            input_flight.update({keys: input_batch})
            output_flight.update({keys: output_batch})

        return copy.deepcopy(input_flight), copy.deepcopy(output_flight)

    def rescale_prediction(self, output=None, label=None):
        if output is not None:
            output = output.squeeze()
        channels_to_predict = self.params[0].data['label_channels']
        # make sure the channels to predict exist a[0]nd are properly ordered
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
                    output[i] *= self.params[0].data[channel + '_scaling']
                if label is not None:
                    label[i] *= self.params[0].data[channel + '_scaling']
            elif channel.startswith('u') or channel == 'nut':
                if output is not None:
                    output[i] *= self.scale * self.params[0].data[channel + '_scaling']
                if label is not None:
                    label[i] *= self.scale * self.params[0].data[channel + '_scaling']
            elif channel == 'p' or channel == 'turb':
                if output is not None:
                    output[i] *= self.scale * self.scale * self.params[0].data[channel + '_scaling']
                if label is not None:
                    label[i] *= self.scale * self.scale * self.params[0].data[channel + '_scaling']
            elif channel == 'epsilon':
                if output is not None:
                    output[i] *= self.scale * self.scale * self.scale * self.params[0].data[channel + '_scaling']
                if label is not None:
                    label[i] *= self.scale * self.scale * self.scale * self.params[0].data[channel + '_scaling']

    # --- Optimisation functions ---

    def optimise_wind_corners(self, opt, n=1000, min_gradient=1e-5, opt_kwargs={'learning_rate':1e-5},
                              wind_zeros=None, wind_mask=None,
                              corners_polynomial=False, use_average_wind_norm=False, wind_norm=None,
                              print_steps=False, verbose=False):
        self._optimisation_variables = self.get_optimisation_variables(use_average_wind_norm=use_average_wind_norm,
                                                                       average_wind_norm=wind_norm)
        self.reset_optimisation_variables()
        optimizer = opt([self._optimisation_variables], **opt_kwargs)
        if print_steps:
            print(optimizer)
        t0 = time.time()
        t = 0
        max_grad = min_gradient+1.0
        output, input = [], []
        losses, grads, opt_var = [], [], []
        # additional stopping criteria
        loss_tolerance = 1e-6  # stop iterating if difference between two consecutive losses is smaller than this
        loss_difference = 1
        while t < n and max_grad > min_gradient and loss_difference > loss_tolerance:
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
            # get prediction
            if corners_polynomial:
                wind_corners = self.get_rotated_polynomial_corners()
            else:
                wind_corners = self.get_rotated_corners()
            interpolated_wind = self._interpolator.edge_interpolation(wind_corners)
            input = torch.cat([self.terrain.network_terrain, interpolated_wind])
            output = self.run_prediction(input, index=-1)  # Run network prediction with input csv

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

            # loss difference
            if t > 1:
                loss_difference = np.abs((loss.item() - losses[-1]))
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

            if t == n-1 or max_grad < min_gradient or loss_difference < loss_tolerance:
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

        p = (self._wind_args.params['sparse_data']['percentage_of_sparse_data'])
        wind_input = self.get_sparse_wind_blocks(p)

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
            if self.flag.use_hybrid_model[0]:
                sequence_length = self._model_args.params['hybrid_model']['sequence_length']
                trajectory_length = int((self._simulated_flight_data['x'].size/sequence_length) + 1)
                trajectory_input = []
                flight_data = self._simulated_flight_data
                for i in range(sequence_length):
                    trajectory_sequence = {}
                    for keys, values in flight_data.items():
                        if i == 0:
                            trajectory_batch = values[:trajectory_length]
                            trajectory_sequence.update({keys: trajectory_batch})
                        elif i == sequence_length-1:
                            trajectory_batch = values[i*trajectory_length:]
                            trajectory_sequence.update({keys: trajectory_batch})
                        else:
                            trajectory_batch = values[((i-1)*trajectory_length):(i*trajectory_length)]
                            trajectory_sequence.update({keys: trajectory_batch})
                    wind_blocks, var_blocks, wind_zeros, wind_mask \
                        = self.get_flight_wind_blocks(trajectory_sequence)
                    traj_wind_input = self.add_mask_to_wind(wind_zeros, wind_mask)
                    trajectory_wind_input = torch.cat([self.terrain.network_terrain, traj_wind_input])
                    trajectory_input.append(trajectory_wind_input)
                input = torch.stack(trajectory_input, dim=0)
                output = self.run_prediction(input)
                # self.rescale_prediction(output, self.labels)
            else:
                # bin trajectory points
                input_flight_data = self._simulated_flight_data
                wind_blocks, var_blocks, wind_zeros, wind_mask = self.get_flight_wind_blocks(input_flight_data)
                # get prediction
                wind_input = self.add_mask_to_wind(wind_zeros, wind_mask)
                input = torch.cat([self.terrain.network_terrain, wind_input])
                output = self.run_prediction(input)
        else:
            wind_input = self._wind_input.clone()
            # run prediction
            if self.flag.use_hybrid_model[0]:
                terrain_for_sequence = self.terrain.network_terrain.repeat(wind_input.shape[0], 1, 1, 1).unsqueeze(1)
                input = torch.cat([terrain_for_sequence, wind_input], dim=1)
            else:
                input = torch.cat([self.terrain.network_terrain, wind_input])
            output = self.run_prediction(input)
            # self.rescale_prediction(output, self.labels)

        loss = self.evaluate_loss(output)
        outputs.append(output)
        losses.append(loss)
        if self.flag.use_hybrid_model[0]:
            inputs.append(input[0, :])
        else:
            inputs.append(input)
        print(' loss: ', loss.item())

        return outputs, losses, inputs

    def flight_prediction(self):
        outputs, losses, inputs = [], [], []

        if self.flag.use_hybrid_model[0]:
            sequence_length = self._model_args.params['hybrid_model']['sequence_length']
            trajectory_length = int((self._flight_data['x'].size/sequence_length) + 1)
            trajectory_input = []
            flight_data = self._flight_data
            for i in range(sequence_length):
                trajectory_sequence = {}
                for keys, values in flight_data.items():
                    if i == 0:
                        trajectory_batch = values[:trajectory_length]
                        trajectory_sequence.update({keys: trajectory_batch})
                    elif i == sequence_length-1:
                        trajectory_batch = values[i*trajectory_length:]
                        trajectory_sequence.update({keys: trajectory_batch})
                    else:
                        trajectory_batch = values[((i-1)*trajectory_length):(i*trajectory_length)]
                        trajectory_sequence.update({keys: trajectory_batch})
                wind_blocks, var_blocks, wind_zeros, wind_mask \
                    = self.get_flight_wind_blocks(trajectory_sequence)
                traj_wind_input = self.add_mask_to_wind(wind_zeros, wind_mask)
                trajectory_wind_input = torch.cat([self.terrain.network_terrain, traj_wind_input])
                trajectory_input.append(trajectory_wind_input)
            wind_input = torch.stack(trajectory_input, dim=0)
            output = self.run_prediction(wind_input)
            # self.rescale_prediction(output, self.labels)
            loss = self.evaluate_loss(output)
            outputs.append(output)
            losses.append(loss)
            inputs.append(wind_input[0, :])
            print(' loss: ', loss.item())
        else:
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
        test_set_range = 1
        if self.flag.batch_test and self.flag.test_simulated_data:
            test_set_range = self.test_set_length
        elif self.flag.batch_test and self.flag.test_flight_data:
            test_set_range = len(self._flight_args.params['files'])

        # batch variables
        inputs, outputs = [], []
        losses_dict = []

        # load data for each sample
        for t in range(test_set_range):
            if self.flag.batch_test:
                if self.flag.test_simulated_data:
                    # get flight data
                    self.original_input, self.labels, self.scale, self.data_set_file_name, self.grid_size, \
                        self.original_terrain, self.test_set_length \
                        = self.load_data_set(t)
                    # Noise
                    self.noisy_labels = self.labels.clone()
                    if self.flag.add_gaussian_noise:
                        self.noisy_labels = self.add_gaussian_noise(self.noisy_labels)
                    if self.flag.add_turbulence:
                        self.noisy_labels = self.add_turbulence(self.noisy_labels)
                    _, self._simulated_flight_data = self.get_trajectory_wind_blocks()
                    flight_data = self._simulated_flight_data
                    self.terrain = self.get_cfd_terrain()
                    self._base_cfd_corners = self.get_cfd_corners()
                if self.flag.test_flight_data:
                    # get flight data
                    if 0 <= t < 4:
                        cosmo_file = 'data/cosmo-1_ethz_fcst_2018112312.nc'
                        terrain_tiff_file = 'data/riemenstalden_full.tif'
                        flight_data_dir = 'data/riemenstalden/'
                    elif 4 <= t < 6:
                        cosmo_file = 'data/cosmo-1_ethz_fcst_2018112309.nc'
                        terrain_tiff_file = 'data/fluelen_full.tif'
                        flight_data_dir = 'data/fluelen/'
                    elif 6 <= t < 7:
                        terrain_tiff_file = 'data/tobelhof.tif'
                        flight_data_dir = 'data/tobelhof/'
                    else:
                        print('Too many flight files in the config file!')
                        raise ValueError

                    self._flight_data = self.load_flight_data(index=t, flight_dir=flight_data_dir)
                    flight_data = self._flight_data
                    self._cosmo_wind = self.load_wind(cosmo_file=cosmo_file, index=t)
                    self.terrain = self.load_cosmo_terrain(terrain_tiff=terrain_tiff_file)
                    if len(self._cosmo_wind) > 1:
                        self._base_cosmo_corners = self.get_cosmo_corners()
            else:
                if self.flag.test_simulated_data:
                    # get flight data
                    flight_data = self._simulated_flight_data
                if self.flag.test_flight_data:
                    # get flight data
                    index = self._flight_args.params['index']
                    flight_data = self.load_flight_data(index=index)

            # sliding window variables (in seconds)
            window_time = self._window_splits_args.params['window_time']  # seconds
            response_time = self._window_splits_args.params['response_time']  # seconds
            step_time = self._window_splits_args.params['step_time']  # seconds

            # time interval between two consecutive flight measurements
            dt = ((flight_data['time_microsec'][-1]-flight_data['time_microsec'][0])/
                  flight_data['time_microsec'].size)/1e6

            # sliding window variables
            window_size = int(window_time/dt)
            response_size = int(response_time/dt)
            step_size = int(step_time/dt)

            # total time of flight and maximum number of windows
            if self.flag.sliding_input_prediction and not self.flag.incremental_input_prediction:
                max_num_windows = int((flight_data['x'].size - (window_size + response_size)) / step_size + 1)
            elif self.flag.incremental_input_prediction and not self.flag.sliding_input_prediction:
                max_num_windows = int((flight_data['x'].size - window_size) / step_size + 1)
            if max_num_windows < 1:
                # raise ValueError('window time exceeds flight time')
                print('Not enough windows for test set', t)
                continue
            total_time_of_flight = (flight_data['time_microsec'][-1] - flight_data['time_microsec'][0]) / 1e6
            print('')
            print('Number of windows: ', max_num_windows)
            print('Number of measurements per window: ', window_size)
            print('Number of measurements per response: ', response_size)
            print('Number of measurements per step: ', step_size)
            print('Total time of flight', total_time_of_flight)
            print('Time interval between two consecutive flight measurements', dt, '\n')

            # initialize dictionary for losses
            loss_dict = {}
            if self.flag.test_simulated_data:
                loss_dict.update({'Test': 'Simulated data'})
            if self.flag.test_flight_data:
                loss_dict.update({'Test': 'Real flight data'})
            loss_dict.update({'Models used': self._model_args.params['name']})
            loss_dict.update({'Sample number': t})
            if self.flag.test_simulated_data:
                file_name = self.data_set_file_name
            if self.flag.test_flight_data:
                if self.flag.batch_test:
                    file_name = self._flight_args.params['files'][t]
                else:
                    file_name = self._flight_args.params['files'][self._flight_args.params['index']]
            loss_dict.update({'File name': file_name})
            loss_dict.update({'Number of windows': max_num_windows})
            # nn losses
            for m in range(len(self._model_args.params['name']) - 1):
                loss_dict.update({'NN wind loss mae ' + str(m): []})
                loss_dict.update({'NN wind loss mse ' + str(m): []})
            if self.flag.use_average_wind:
                loss_dict.update({'Average wind loss mae': []})
                loss_dict.update({'Average wind loss mse': []})
            if self.flag.use_optimized_corners:
                loss_dict.update({'Optimized corners wind loss mae': []})
                loss_dict.update({'Optimized corners wind loss mse': []})
            if self.flag.use_optimized_corners_polynomial:
                loss_dict.update({'Optimized corners polynomial wind loss mae': []})
                loss_dict.update({'Optimized corners polynomial wind loss mse': []})
            if self.flag.use_gpr_prediction:
                loss_dict.update({'GPR loss mae': []})
                loss_dict.update({'GPR loss mse': []})

            print('')
            print('Test set number: ', t)
            print('Test set file name: ', file_name)
            print('')

            for i in range(max_num_windows):
                # split data into input and label based on window variables
                input_flight_data, label_flight_data = \
                    self.sliding_window_split(flight_data, window_size, response_size, i*step_size)

                # wind field labels
                if self.flag.test_simulated_data and self.flag.rescale_prediction:
                    if i == 0:
                        self.rescale_prediction(label=self.labels)

                # trajectory labels
                # filter out points in the labels which are outside the terrain sample
                valid_labels = []
                for k in range(len(label_flight_data['x'])):
                    if ((label_flight_data['x'][k] > self.terrain.x_terr[0]) and
                            (label_flight_data['x'][k] < self.terrain.x_terr[-1]) and
                            (label_flight_data['y'][k] > self.terrain.y_terr[0]) and
                            (label_flight_data['y'][k] < self.terrain.y_terr[-1]) and
                            (label_flight_data['alt'][k] > self.terrain.z_terr[0]) and
                            (label_flight_data['alt'][k] < self.terrain.z_terr[-1])):
                        valid_labels.append([label_flight_data['wn'][k], label_flight_data['we'][k],
                                             label_flight_data['wd'][k]])
                if len(valid_labels) == 0:
                    print(i, 'Labels outside terrain dimensions')
                    print('')
                    continue
                trajectory_labels = torch.from_numpy(np.row_stack(valid_labels))

                # --- Zero wind prediction ---
                if self.flag.test_simulated_data:
                    zero_wind_output = torch.zeros_like(self.labels)
                    zero_wind_loss_mae = self._loss_fn_mae(zero_wind_output.to(self._device),
                                                           self.labels.to(self._device))
                    zero_wind_loss_mse = self._loss_fn_mse(zero_wind_output.to(self._device),
                                                           self.labels.to(self._device))
                    # check that error is not zero to avoid division by 0 when normalizing the other losses
                    not_zero_labels = not(zero_wind_loss_mae == 0 and zero_wind_loss_mse == 0)
                    # loss_dict.update({'zero wind loss mae': zero_wind_loss_mae.item()})
                    # loss_dict.update({'zero wind loss mse': zero_wind_loss_mse.item()})
                if self.flag.test_flight_data:
                    interpolated_zero_wind_output = torch.zeros_like(trajectory_labels)
                    zero_wind_loss_mae = self._loss_fn_mae(interpolated_zero_wind_output.to(self._device),
                                                           trajectory_labels.to(self._device))
                    zero_wind_loss_mse = self._loss_fn_mse(interpolated_zero_wind_output.to(self._device),
                                                           trajectory_labels.to(self._device))
                    # check that error is not zero to avoid division by 0 when normalizing the other losses
                    not_zero_labels = not(zero_wind_loss_mae == 0 and zero_wind_loss_mse == 0)
                    # loss_dict.update({'trajectory zero wind loss': zero_wind_loss_mse})

                # --- Networks prediction ---
                # scale real flight data if requested
                if self.flag.test_flight_data and self.flag.rescale_prediction:
                    input_flight_wind_vectors = np.array([input_flight_data['wn'], input_flight_data['we'], input_flight_data['wd']])
                    scale = np.linalg.norm(input_flight_wind_vectors, axis=0).mean()
                    # scale = 1
                    scale_x = 0.5
                    scale_y = 0.5
                    scale_z = 0.1
                    input_flight_data['wn'] /= (scale * scale_x)
                    input_flight_data['we'] /= (scale * scale_y)
                    input_flight_data['wd'] /= (scale * scale_z)

                # Create network input
                # sequential input
                if any(self.flag.use_hybrid_model):
                    sequence_length = self._model_args.params['hybrid_model']['sequence_length']
                    #sequence_length = i+1
                    trajectory_length = int((input_flight_data['x'].size / sequence_length) + 1)
                    trajectory_input = []
                    for k in range(sequence_length):
                        trajectory_sequence = {}
                        for keys, values in input_flight_data.items():
                            if k == 0:
                                trajectory_batch = values[:trajectory_length]
                                trajectory_sequence.update({keys: trajectory_batch})
                            elif k == sequence_length - 1:
                                trajectory_batch = values[k * trajectory_length:]
                                trajectory_sequence.update({keys: trajectory_batch})
                            else:
                                trajectory_batch = values[((k - 1) * trajectory_length):(k * trajectory_length)]
                                trajectory_sequence.update({keys: trajectory_batch})
                        wind_blocks, var_blocks, wind_zeros, wind_mask \
                            = self.get_flight_wind_blocks(trajectory_sequence)
                        traj_wind_input = self.add_mask_to_wind(wind_zeros, wind_mask)
                        trajectory_wind_input = torch.cat([self.terrain.network_terrain, traj_wind_input])
                        trajectory_input.append(trajectory_wind_input)
                    input_sequential = torch.stack(trajectory_input, dim=0)
                # normal input
                wind_blocks, var_blocks, wind_zeros, wind_mask \
                    = self.get_flight_wind_blocks(input_flight_data)
                # get prediction
                wind_input = self.add_mask_to_wind(wind_zeros, wind_mask)
                input = torch.cat([self.terrain.network_terrain, wind_input])

                # iterate over loaded networks
                for m in range(len(self._model_args.params['name']) - 1):  # last network is used for corner optimization
                    if self.flag.use_hybrid_model[m]:
                        nn_input = input_sequential
                    else:
                        nn_input = input
                    nn_output = self.run_prediction(nn_input, index=m)
                    if self.flag.test_simulated_data and self.flag.rescale_prediction:  # only for simulated trajectory
                        self.rescale_prediction(output=nn_output)
                    if self.flag.test_flight_data and self.flag.rescale_prediction:
                        if m == 0:
                            # rescale nn input and input flight data only once
                            input[:, 0, :] *= scale * scale_x
                            input[:, 1, :] *= scale * scale_y
                            input[:, 2, :] *= scale * scale_z
                            input_flight_data['wn'] *= scale * scale_x
                            input_flight_data['we'] *= scale * scale_y
                            input_flight_data['wd'] *= scale * scale_z
                        nn_output[0, :] *= scale * scale_x
                        nn_output[1, :] *= scale * scale_y
                        nn_output[2, :] *= scale * scale_z

                    # losses
                    if self.flag.test_simulated_data:
                        nn_loss_mae = self._loss_fn_mae(nn_output.to(self._device), self.labels.to(self._device))
                        nn_loss_mse = self._loss_fn_mse(nn_output.to(self._device), self.labels.to(self._device))
                        if self.flag.normalize_losses and not_zero_labels:
                            nn_loss_mae /= zero_wind_loss_mae
                            nn_loss_mse /= zero_wind_loss_mse
                        loss_dict['NN wind loss mae ' + str(m)].append(nn_loss_mae.item())
                        loss_dict['NN wind loss mse ' + str(m)].append(nn_loss_mse.item())
                        print(i, ' NN loss ' + str(m) + ' is: ', nn_loss_mse.item())
                    if self.flag.test_flight_data:
                        FlightInterpolator = FlightInterpolation(input_flight_data, terrain=self.terrain,
                                                               wind_data_for_prediction=label_flight_data)
                        interpolated_nn_output = FlightInterpolator.interpolate_flight_data_from_grid(nn_output)
                        nn_loss_mae = self._loss_fn_mae(interpolated_nn_output.to(self._device),
                                                        trajectory_labels.to(self._device))
                        nn_loss_mse = self._loss_fn_mse(interpolated_nn_output.to(self._device),
                                                        trajectory_labels.to(self._device))
                        loss_axis = False
                        if loss_axis:
                            nn_loss_mse_x = self._loss_fn_mse(interpolated_nn_output[:,0].to(self._device),
                                                            trajectory_labels[:,0].to(self._device))
                            nn_loss_mse_y = self._loss_fn_mse(interpolated_nn_output[:,1].to(self._device),
                                                            trajectory_labels[:,1].to(self._device))
                            nn_loss_mse_z = self._loss_fn_mse(interpolated_nn_output[:,2].to(self._device),
                                                            trajectory_labels[:,2].to(self._device))
                        if self.flag.normalize_losses and not_zero_labels:
                            nn_loss_mae /= zero_wind_loss_mae
                            nn_loss_mse /= zero_wind_loss_mse
                        loss_dict['NN wind loss mae ' + str(m)].append(nn_loss_mae.item())
                        loss_dict['NN wind loss mse ' + str(m)].append(nn_loss_mse.item())
                        print(i, ' Trajectory NN loss ' + str(m) + ' is: ', nn_loss_mse.item())
                        if loss_axis:
                            print(i, ' Trajectory NN loss x' + str(m) + ' is: ', nn_loss_mse_x.item())
                            print(i, ' Trajectory NN loss y' + str(m) + ' is: ', nn_loss_mse_y.item())
                            print(i, ' Trajectory NN loss z' + str(m) + ' is: ', nn_loss_mse_z.item())

                # --- Average wind prediction ---
                if self.flag.use_average_wind:
                    average_wind_input = np.array([input_flight_data['wn'], input_flight_data['we'], input_flight_data['wd']])
                    if self.flag.test_simulated_data:
                        average_wind_output = torch.ones_like(self.labels)
                        average_wind_output[0, :] *= average_wind_input[0].mean()
                        average_wind_output[1, :] *= average_wind_input[1].mean()
                        average_wind_output[2, :] *= average_wind_input[2].mean()
                        average_wind_loss_mae = self._loss_fn_mae(average_wind_output.to(self._device),
                                                                  self.labels.to(self._device))
                        average_wind_loss_mse = self._loss_fn_mse(average_wind_output.to(self._device),
                                                                  self.labels.to(self._device))
                        if self.flag.normalize_losses and not_zero_labels:
                            average_wind_loss_mae /= zero_wind_loss_mae
                            average_wind_loss_mse /= zero_wind_loss_mse
                        loss_dict['Average wind loss mae'].append(average_wind_loss_mae.item())
                        loss_dict['Average wind loss mse'].append(average_wind_loss_mse.item())
                        print(i, ' Average wind loss is: ', average_wind_loss_mse.item())
                    if self.flag.test_flight_data:
                        interpolated_average_wind_output = np.ones_like(trajectory_labels) * [average_wind_input[0].mean(),
                                                                                              average_wind_input[1].mean(),
                                                                                              average_wind_input[2].mean()]
                        interpolated_average_wind_output = torch.from_numpy(interpolated_average_wind_output)
                        average_wind_loss_mae = self._loss_fn_mae(interpolated_average_wind_output.to(self._device),
                                                                  trajectory_labels.to(self._device))
                        average_wind_loss_mse = self._loss_fn_mse(interpolated_average_wind_output.to(self._device),
                                                                  trajectory_labels.to(self._device))
                        loss_axis = False
                        if loss_axis:
                            average_wind_loss_mse_x = self._loss_fn_mse(interpolated_average_wind_output[:, 0].to(self._device),
                                                                      trajectory_labels[:, 0].to(self._device))
                            average_wind_loss_mse_y = self._loss_fn_mse(interpolated_average_wind_output[:, 1].to(self._device),
                                                                      trajectory_labels[:, 1].to(self._device))
                            average_wind_loss_mse_z = self._loss_fn_mse(interpolated_average_wind_output[:, 2].to(self._device),
                                                                      trajectory_labels[:, 2].to(self._device))
                        if self.flag.normalize_losses and not_zero_labels:
                            average_wind_loss_mae /= zero_wind_loss_mae
                            average_wind_loss_mse /= zero_wind_loss_mse
                        loss_dict['Average wind loss mae'].append(average_wind_loss_mae.item())
                        loss_dict['Average wind loss mse'].append(average_wind_loss_mse.item())
                        print(i, ' Trajectory average wind loss is: ', average_wind_loss_mse.item())
                        if loss_axis:
                            print(i, ' Trajectory average wind loss is x: ', average_wind_loss_mse_x.item())
                            print(i, ' Trajectory average wind loss is x: ', average_wind_loss_mse_y.item())
                            print(i, ' Trajectory average wind loss is x: ', average_wind_loss_mse_z.item())

                # --- Optimized corner prediction ---
                if self.flag.use_optimized_corners:
                    optimiser = OptTest(torch.optim.Adagrad, {'lr': 1.0, 'lr_decay': 0.1})
                    n_steps = self._corner_optimisation_args.params['num_of_optimisation_steps']
                    corners_output, _, _, _, _ \
                        = self.optimise_wind_corners(optimiser.opt, n=n_steps, opt_kwargs=optimiser.kwargs,
                                                     wind_zeros=wind_zeros, wind_mask=wind_mask)
                    if self.flag.test_simulated_data:
                        corners_loss_mae = self._loss_fn_mae(corners_output.to(self._device),
                                                             self.labels.to(self._device))
                        corners_loss_mse = self._loss_fn_mse(corners_output.to(self._device),
                                                             self.labels.to(self._device))
                        if self.flag.normalize_losses and not_zero_labels:
                            corners_loss_mae /= zero_wind_loss_mae
                            corners_loss_mse /= zero_wind_loss_mse
                        loss_dict['Optimized corners wind loss mae'].append(corners_loss_mae.item())
                        loss_dict['Optimized corners wind loss mse'].append(corners_loss_mse.item())
                        print(i, ' optimized corners loss is: ', corners_loss_mse.item())
                    if self.flag.test_flight_data:
                        # interpolate prediction to scattered flight data locations corresponding to the labels
                        FlightInterpolator = FlightInterpolation(input_flight_data, terrain=self.terrain,
                                                               wind_data_for_prediction=label_flight_data)
                        interpolated_corners_output = FlightInterpolator.interpolate_flight_data_from_grid(corners_output)
                        corners_loss_mae = self._loss_fn_mae(interpolated_corners_output.to(self._device),
                                                             trajectory_labels.to(self._device))
                        corners_loss_mse = self._loss_fn_mse(interpolated_corners_output.to(self._device),
                                                             trajectory_labels.to(self._device))
                        if self.flag.normalize_losses and not_zero_labels:
                            corners_loss_mae /= zero_wind_loss_mae
                            corners_loss_mse /= zero_wind_loss_mse
                        loss_dict['Optimized corners wind loss mae'].append(corners_loss_mae.item())
                        loss_dict['Optimized corners wind loss mse'].append(corners_loss_mse.item())
                        print(i, ' Trajectory optimized corners loss is: ', corners_loss_mse.item())

                # --- Optimized corner polynomial prediction ---
                if self.flag.use_optimized_corners_polynomial:
                    optimiser = OptTest(torch.optim.Adagrad, {'lr': 1.0, 'lr_decay': 0.1})
                    n_steps = self._corner_optimisation_args.params['num_of_optimisation_steps']
                    # value used to initalize the polynomial function at each point
                    average_wind_norm = np.linalg.norm(average_wind_input)
                    corners_output, _, _, _, _ \
                        = self.optimise_wind_corners(optimiser.opt, n=n_steps, opt_kwargs=optimiser.kwargs,
                                                     wind_zeros=wind_zeros, wind_mask=wind_mask,
                                                     corners_polynomial=True, use_average_wind_norm=True,
                                                     wind_norm=average_wind_norm)
                    if self.flag.test_simulated_data:
                        corners_loss_mae = self._loss_fn_mae(corners_output.to(self._device),
                                                             self.labels.to(self._device))
                        corners_loss_mse = self._loss_fn_mse(corners_output.to(self._device),
                                                             self.labels.to(self._device))
                        if self.flag.normalize_losses and not_zero_labels:
                            corners_loss_mae /= zero_wind_loss_mae
                            corners_loss_mse /= zero_wind_loss_mse
                        loss_dict['Optimized corners polynomial wind loss mae'].append(corners_loss_mae.item())
                        loss_dict['Optimized corners polynomial wind loss mse'].append(corners_loss_mse.item())
                        print(i, ' optimized corners polynomial loss is: ', corners_loss_mse.item())
                    if self.flag.test_flight_data:
                        # interpolate prediction to scattered flight data locations corresponding to the labels
                        FlightInterpolator = FlightInterpolation(input_flight_data, terrain=self.terrain,
                                                               wind_data_for_prediction=label_flight_data)
                        interpolated_corners_output = FlightInterpolator.interpolate_flight_data_from_grid(corners_output)
                        corners_loss_mae = self._loss_fn_mae(interpolated_corners_output.to(self._device),
                                                             trajectory_labels.to(self._device))
                        corners_loss_mse = self._loss_fn_mse(interpolated_corners_output.to(self._device),
                                                             trajectory_labels.to(self._device))
                        if self.flag.normalize_losses and not_zero_labels:
                            corners_loss_mae /= zero_wind_loss_mae
                            corners_loss_mse /= zero_wind_loss_mse
                        loss_dict['Optimized corners polynomial wind loss mae'].append(corners_loss_mae.item())
                        loss_dict['Optimized corners polynomial wind loss mse'].append(corners_loss_mse.item())
                        print(i, ' Trajectory optimized polynomial corners loss is: ', corners_loss_mse.item())

                # --- Interpolation (Krigging) prediction ---
                if self.flag.use_krigging_prediction:
                    # method too slow and should not be used
                    FlightInterpolator = FlightInterpolation(input_flight_data, terrain=self.terrain, predict=True,
                                                           wind_data_for_prediction=label_flight_data)
                    _, _, interpolated_krigging_output = FlightInterpolator.interpolate_flight_data_krigging()

                # --- Interpolation (Gaussian Process Regression) prediction ---
                if self.flag.use_gpr_prediction:
                    FlightInterpolator = FlightInterpolation(input_flight_data, terrain=self.terrain, predict=True,
                                                           wind_data_for_prediction=label_flight_data)
                    _, _, interpolated_gpr_output, gpr_output = FlightInterpolator.interpolate_flight_data_gpr()
                    if self.flag.test_flight_data:
                        gpr_loss_mae = self._loss_fn_mae(interpolated_gpr_output.to(self._device),
                                                     trajectory_labels.to(self._device))
                        gpr_loss_mse = self._loss_fn_mse(interpolated_gpr_output.to(self._device),
                                                     trajectory_labels.to(self._device))
                        if self.flag.normalize_losses and not_zero_labels:
                            gpr_loss_mae /= zero_wind_loss_mae
                            gpr_loss_mse /= zero_wind_loss_mse
                        loss_dict['GPR wind loss mae'].append(gpr_loss_mae.item())
                        loss_dict['GPR wind loss mse'].append(gpr_loss_mse.item())
                        print(i, ' Trajectoru GPR loss is: ', gpr_loss_mse.item())

                # outputs
                # if i == 0:
                #     outputs.append(nn_output)
                #     inputs.append(input)
            losses_dict.append(loss_dict)

        return outputs, losses_dict, inputs

