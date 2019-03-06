import numpy as np
import scipy.optimize
import nn_wind_prediction.utils as utils
import nn_wind_prediction.cosmo as cosmo
import nn_wind_prediction.models as models
from nn_wind_prediction.utils.interpolation import DataInterpolation
import datetime
from scipy import ndimage
import torch
import os
import time
import matplotlib.pyplot as plt


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


class WindOptimiser(object):
    def __init__(self, config_yaml, initial_rotation=0.0, initial_scale=1.0, resolution=64):
        self.config_yaml = config_yaml
        self.cosmo_args = utils.COSMOParameters(self.config_yaml)
        self.ulog_args = utils.UlogParameters(self.config_yaml)
        self.model_args = utils.BasicParameters(self.config_yaml, 'model')
        self.rotation = initial_rotation
        self.scale = initial_scale
        self.resolution = resolution
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ulog_data = self.load_ulog_data()
        self.cosmo_wind = self.load_wind()
        self.terrain = self.load_terrain()
        self.cosmo_corners = self.get_cosmo_corners()
        self.__interpolator = DataInterpolation(self.device, 3, *self.terrain.get_dimensions())
        self.net = self.load_network_model()
        self.wind_blocks, self.var_blocks = self.get_wind_blocks()

    def load_ulog_data(self):
        print('Loading ulog data...', end='', flush=True)
        t_start = time.time()
        self.ulog_args.print()
        ulog_data = utils.get_log_data(self.ulog_args.params['file'])

        if (self.ulog_args.params['use_ekf_wind']):
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
        self.cosmo_args.print()

        lat0, lon0 = self.ulog_data['lat'][0], self.ulog_data['lon'][0]

        # Get cosmo wind
        t0 = datetime.datetime.utcfromtimestamp(self.ulog_data['utc_microsec'][0] / 1e6)
        offset_cosmo_time = self.cosmo_args.get_cosmo_time(t0.hour)
        cosmo_wind = cosmo.extract_cosmo_data(self.cosmo_args.params['file'], lat0, lon0, offset_cosmo_time,
                                              terrain_file=self.cosmo_args.params['terrain_file'])
        print(' done [{:.2f} s]'.format(time.time() - t_start))
        return cosmo_wind

    def load_terrain(self):
        print('Loading terrain...', end='', flush=True)
        t_start = time.time()
        # Get corresponding terrain
        # min_height = min(ulog_data['alt'].min(), h_terr.min())
        block_height = [1100.0 / 95 * 63]
        # x_terr, y_terr and z_terr are the (regular, monotonic) index arrays for the h_terr and full_block arrays
        # h_terr is the terrain height
        boolean_terrain = self.model_args.params['boolean_terrain']

        terrain = TerrainBlock(
            *utils.get_terrain(self.cosmo_args.params['terrain_tiff'], self.cosmo_wind['x'][[0, 1], [0, 1]],
                               self.cosmo_wind['y'][[0, 1], [0, 1]],
                               block_height, (self.resolution, self.resolution, self.resolution)),
            device=self.device, boolean_terrain=boolean_terrain)
        print(' done [{:.2f} s]'.format(time.time() - t_start))
        return terrain

    def load_network_model(self):
        print('Loading network model...', end='', flush=True)
        t_start = time.time()
        yaml_loc = os.path.join(self.model_args.params['location'], self.model_args.params['name'], 'params.yaml')
        params = utils.EDNNParameters(yaml_loc)                                         # load the model config
        NetworkType = getattr(models, params.model['model_type'])                       # load the model
        net = NetworkType(**params.model_kwargs())                                 # load learnt parameters
        model_loc = os.path.join(self.model_args.params['location'], self.model_args.params['name'],
                                 self.model_args.params['version']+'.model')
        net.load_state_dict(torch.load(model_loc, map_location=lambda storage, loc: storage))
        net = net.to(self.device)
        print(' done [{:.2f} s]'.format(time.time() - t_start))
        return net

    def set_rotation_scale(self, rotation, scale):
        self.rotation = rotation
        self.scale = scale
        # Get corner winds for model inference, offset to actual terrain heights
        print('Rotation: {0:0.2f} deg, scale: {1:0.2f}'.format(self.rotation, scale))

    def get_cosmo_corners(self):
        cosmo_corners = cosmo.cosmo_corner_wind(self.cosmo_wind, self.terrain.z_terr, terrain_height=self.terrain.terrain_corners,
                                                rotate=self.rotation*np.pi / 180.0, scale=self.scale)
        return torch.from_numpy(cosmo_corners.astype(np.float32)).to(self.device)

    def get_wind_blocks(self):
        print('Getting binned wind blocks...', end='', flush=True)
        t_start = time.time()

        dx = self.terrain.x_terr[[0,-1]]; ddx = (self.terrain.x_terr[1]-self.terrain.x_terr[0])/2.0
        dy = self.terrain.y_terr[[0,-1]]; ddy = (self.terrain.y_terr[1]-self.terrain.y_terr[0])/2.0
        dz = self.terrain.z_terr[[0,-1]]; ddz = (self.terrain.z_terr[1]-self.terrain.z_terr[0])/2.0
        # determine the grid dimension
        corners = {'x_min': dx[0] - ddx, 'x_max': dx[1] + ddx, 'y_min': dy[0] - ddy, 'y_max': dy[1] + ddy,
                   'z_min': dz[0] - ddz, 'z_max': dz[1] + ddz, 'n_cells': self.terrain.get_dimensions()[0]}

        # bin the data into the regular grid
        wind, variance = utils.bin_log_data(self.ulog_data, corners)
        print(' done [{:.2f} s]'.format(time.time() - t_start))
        return wind, variance

    def generate_wind_input(self):
        # Currently can only use interpolated wind, since we only have the cosmo corners
        # interpolating the vertical edges
        input = torch.cat(
            [self.terrain.network_terrain, self.__interpolator.edge_interpolation(self.cosmo_corners)])
        return input

    def build_csv(self):
        try:
            csv_args = utils.BasicParameters(self.config_yaml, 'csv')
            print('Saving csv to {0}'.format(csv_args.params['file']))
            utils.build_csv(self.terrain.x_terr, self.terrain.y_terr, self.terrain.z_terr, self.terrain.full_block,
                            self.terrain.cosmo_corners, csv_args.params['file'])
        except:
            print('CSV filename parameter (csv:file) not found in {0}, csv not saved'.format(self.config_yaml))

    def run_prediction(self, input):
        return self.net(input.unsqueeze(0))

    def rms_error(self, input, output):
        return 0.0

    def evaluate_error(self, rotation, scale):
        self.set_rotation_scale(rotation, scale)        # Set new rotation
        input = self.generate_wind_input()
        output = self.run_prediction(input)                      # Run network prediction with input csv
        utils.plot_sample(output.squeeze(0).detach(), self.wind_blocks, self.terrain.network_terrain.squeeze(0))
        plt.show()


test = WindOptimiser('config/optim_config.yaml')
test.evaluate_error(0.0, 1.0)