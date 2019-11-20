import numpy as np
import nn_wind_prediction.utils as utils
import nn_wind_prediction.models as models
from analysis_utils import extract_cosmo_data as cosmo
from analysis_utils import ulog_utils, get_mapgeo_terrain
from analysis_utils.interpolate_log_data import UlogInterpolation
from nn_wind_prediction.utils.interpolation import DataInterpolation
from scipy.interpolate import RectBivariateSpline
from sklearn import metrics
from datetime import datetime
from scipy import ndimage
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


class WindOptimiser(object):
    _loss_fn = torch.nn.MSELoss()
    _optimisation_variables = None

    def __init__(self, config_yaml, resolution=64):
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._config_yaml = config_yaml
        self._cosmo_args = utils.COSMOParameters(self._config_yaml)
        self._ulog_args = utils.UlogParameters(self._config_yaml)
        self._traj_args = utils.TrajParameters(self._config_yaml)
        self._model_args = utils.BasicParameters(self._config_yaml, 'model')
        self._rotation0, self._scale0, self._directional_shear0, self._power_law_exponent0 \
            = self.get_optimisation_variables()
        self._resolution = resolution
        self._ulog_data = self.load_ulog_data()
        self._train_ulog_data, self._test_ulog_data = self.train_test_split()
        self._cosmo_wind = self.load_wind()
        self.terrain = self.load_terrain()
        self._trajectory = self.generate_trajectory()
        self._base_cosmo_corners = self.get_cosmo_corners()
        self._interpolator = DataInterpolation(self._device, 3, *self.terrain.get_dimensions())
        self.net = self.load_network_model()
        self.net.freeze_model()
        self._wind_blocks, self._var_blocks, self._wind_zeros, self._wind_mask = self.get_wind_blocks()
        self._loss_fn = self.get_loss_function()


    def load_ulog_data(self):
        print('Loading ulog data...', end='', flush=True)
        t_start = time.time()
        self._ulog_args.print()
        ulog_data = ulog_utils.get_log_data(self._ulog_args.params['file'])

        if (self._ulog_args.params['use_ekf_wind']):
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

    def load_terrain(self):
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

    def generate_trajectory(self):
        x = self._traj_args.params['x']
        y = self._traj_args.params['y']
        z = self._traj_args.params['z']
        h_above_terrain = 100
        x0 = np.zeros(len(x))
        y0 = np.zeros(len(y))
        z0 = np.zeros(len(z))
        for i in range(0, len(x)):
            x0[i] = x[i] + self.terrain.x_terr[0]
            y0[i] = y[i] + self.terrain.y_terr[0]
            interp_spline = RectBivariateSpline(self.terrain.x_terr, self.terrain.y_terr, self.terrain.h_terr)
            z0[i] = z[i] + interp_spline(x0[i], y0[i])

        # Generate points along the trajectory
        points_along_traj = self._traj_args.params['points_along_traj']
        x_pts = np.zeros((len(x)-1, points_along_traj))
        y_pts = np.zeros((len(y)-1, points_along_traj))
        z_pts = np.zeros((len(z)-1, points_along_traj))
        for i in range(0, len(x)-1):
            dist = np.sqrt((x0[i+1]-x0[i])**2 + (y0[i+1]-y0[i])**2 + (z0[i+1]-z0[i])**2)
            n = 1
            for j in range(0, points_along_traj):
                t = n/(points_along_traj+1)
                x_pts[i][j] = x0[i] + t*(x0[i+1]-x0[i])
                y_pts[i][j] = y0[i] + t*(y0[i+1]-y0[i])
                z_pts[i][j] = z0[i] + t*(z0[i+1]-z0[i])
                n += 1
        return x_pts, y_pts, z_pts

    def get_cosmo_corners(self):
        temp_cosmo = cosmo.cosmo_corner_wind(self._cosmo_wind, self.terrain.z_terr, rotate=0.0, scale=1.0,
                                             terrain_height=self.terrain.terrain_corners)
        cosmo_corners = torch.from_numpy(temp_cosmo.astype(np.float32)).to(self._device)
        return cosmo_corners

    def load_network_model(self):
        print('Loading network model...', end='', flush=True)
        t_start = time.time()
        yaml_loc = os.path.join(self._model_args.params['location'], self._model_args.params['name'], 'params.yaml')
        params = utils.EDNNParameters(yaml_loc)                                         # load the model config
        NetworkType = getattr(models, params.model['model_type'])                       # load the model
        net = NetworkType(**params.model_kwargs())                                 # load learnt parameters
        model_loc = os.path.join(self._model_args.params['location'], self._model_args.params['name'],
                                 self._model_args.params['version'] + '.model')
        net.load_state_dict(torch.load(model_loc, map_location=lambda storage, loc: storage))
        net = net.to(self._device)
        print(' done [{:.2f} s]'.format(time.time() - t_start))
        return net

    def get_optimisation_variables(self):
        rotation = self._cosmo_args.params['rotation']
        scale = self._cosmo_args.params['scale']
        directional_wind_shear = self._cosmo_args.params['directional_shear']
        power_law_exponent = self._cosmo_args.params['power_law_exponent']
        if self._cosmo_args.params['optimized_corners'] > 0:
            ones_ = np.ones(self._cosmo_args.params['optimized_corners'])
            rotation *= ones_
            scale *= ones_
            directional_wind_shear *= ones_
            power_law_exponent *= ones_

        return rotation, scale, directional_wind_shear, power_law_exponent

    def reset_optimisation_variables(self, rot=None, scale=None, shear=None, exponent=None):
        ones_ = np.ones(self._cosmo_args.params['optimized_corners'])
        if rot is None:
            rot = self._rotation0
        else:
            if self._cosmo_args.params['optimized_corners'] > 0:
                rot *= ones_
        if scale is None:
            scale = self._scale0
        else:
            if self._cosmo_args.params['optimized_corners'] > 0:
                scale *= ones_
        if shear is None:
            shear = self._directional_shear0
        else:
            if self._cosmo_args.params['optimized_corners'] > 0:
                shear *= ones_
        if exponent is None:
            exponent = self._power_law_exponent0
        else:
            if self._cosmo_args.params['optimized_corners'] > 0:
                exponent *= ones_

        self._optimisation_variables = torch.Tensor([rot, scale, shear, exponent]).to(self._device).requires_grad_()

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

    def get_wind_blocks(self):
        print('Getting binned wind blocks...', end='', flush=True)
        t_start = time.time()

        # Determine the grid dimensions
        dx = self.terrain.x_terr[[0, -1]]; ddx = (self.terrain.x_terr[1]-self.terrain.x_terr[0])/2.0
        dy = self.terrain.y_terr[[0, -1]]; ddy = (self.terrain.y_terr[1]-self.terrain.y_terr[0])/2.0
        dz = self.terrain.z_terr[[0, -1]]; ddz = (self.terrain.z_terr[1]-self.terrain.z_terr[0])/2.0
        grid_dimensions = {'x_min': dx[0] - ddx, 'x_max': dx[1] + ddx, 'y_min': dy[0] - ddy, 'y_max': dy[1] + ddy,
                           'z_min': dz[0] - ddz, 'z_max': dz[1] + ddz, 'n_cells': self.terrain.get_dimensions()[0]}
        if self._ulog_args.params['predict_ulog']:
            UlogInterpolator = UlogInterpolation(self._train_ulog_data, grid_dimensions, self.terrain)
        else:
            UlogInterpolator = UlogInterpolation(self._ulog_data, grid_dimensions, self.terrain)

        # bin the data into the regular grid
        try:
            if self._ulog_args.params['interpolation_method'].lower() == 'bin':
                wind, variance = UlogInterpolator.bin_log_data()
            elif self._ulog_args.params['interpolation_method'].lower() == 'krigging':
                wind, variance = UlogInterpolator.interpolate_log_data_krigging()
            elif self._ulog_args.params['interpolation_method'].lower() == 'idw':
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

    def get_rotated_wind(self):
        sr = torch.zeros((len(self.terrain.z_terr), self._cosmo_args.params['optimized_corners']))
        cr = torch.zeros((len(self.terrain.z_terr), self._cosmo_args.params['optimized_corners']))
        for z in range(len(self.terrain.z_terr)):
            for i in range(self._cosmo_args.params['optimized_corners']):
                angle = self._optimisation_variables[0, i] \
                        + self._optimisation_variables[3, i]*(self.terrain.z_terr[z]/self.terrain.z_terr[0])/10000
                sr[z, i] = torch.sin(angle)
                cr[z, i] = torch.cos(angle)
        # Get corner winds for model inference, offset to actual terrain heights
        cosmo_corners = self._base_cosmo_corners.clone()
        for z in range(len(self.terrain.z_terr)):
            for i in range(self._cosmo_args.params['optimized_corners']):
                cosmo_corners[0, :, i//2, (i+2)%2] = \
                    self._optimisation_variables[1, i]\
                    * (self.terrain.z_terr[z]/self.terrain.z_terr[0])**self._optimisation_variables[2, i]/10*(
                        self._base_cosmo_corners[0, :, i//2, (i+2)%2]*cr[z, i]
                        - self._base_cosmo_corners[1, :, i//2, (i+2)%2]*sr[z, i])
                cosmo_corners[1, :, i//2, (i+2)%2] = \
                    self._optimisation_variables[1, i] \
                    * (self.terrain.z_terr[z]/self.terrain.z_terr[0])**self._optimisation_variables[2, i]/10*(
                        self._base_cosmo_corners[0, :, i//2, (i+2)%2]*sr[z, i]
                        + self._base_cosmo_corners[1, :, i//2, (i+2)%2]*cr[z, i])
        return cosmo_corners

    def generate_wind_input(self):
        cosmo_corners = self.get_rotated_wind()
        # Currently can only use interpolated wind, since we only have the cosmo corners
        # interpolating the vertical edges
        input = torch.cat(
            [self.terrain.network_terrain, self._interpolator.edge_interpolation(cosmo_corners)])
        return input

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
        masked_out = output[0:3,:,:,:]
        masked_out[self._wind_mask] = 0.0
        return self._loss_fn(masked_out, self._wind_zeros)

    def get_prediction(self):
        input = self.generate_wind_input()       # Set new rotation
        output = self.run_prediction(input)      # Run network prediction with input csv
        return output

    def get_ulog_prediction(self):
        input = self.generate_ulog_input()
        output = self.run_prediction(input)
        return output

    def optimise_wind_variables(self, opt, n=1000, min_gradient=1e-5, opt_kwargs={'learning_rate':1e-5}, verbose=False):
        # Check which optimisation variables are passed to the optimiser
        # optimise_variables = self._cosmo_args.params['optimise_variables']
        # reduced_optimisation_variables = self._optimisation_variables.clone().detach().cpu().numpy()
        # reduced_optimisation_variables = reduced_optimisation_variables[optimise_variables]
        # reduced_optimisation_variables = \
        #     torch.Tensor([reduced_optimisation_variables]).to(self._device).requires_grad_()
        optimizer = opt([self._optimisation_variables], **opt_kwargs)
        print(optimizer)
        t0 = time.time()
        t = 0
        max_grad = min_gradient+1.0
        losses, grads, optimisation_variables_ = [], [], []
        while t < n and max_grad > min_gradient:
            if self._cosmo_args.params['optimized_corners'] > 0:
                print(t,
                      ' r: ', *self._optimisation_variables[0].detach().cpu().numpy() * 180.0 / np.pi,
                      ' s: ', *self._optimisation_variables[1].detach().cpu().numpy(),
                      ' ds: ', *self._optimisation_variables[2].detach().cpu().numpy() * 180.0 / (np.pi*10000),
                      ' e: ', *self._optimisation_variables[3].detach().cpu().numpy()/10,
                      '')
            else:
                print(t,
                      ' r: ', self._optimisation_variables[0].detach().cpu().numpy() * 180.0 / np.pi,
                      ' s: ', self._optimisation_variables[1].detach().cpu().numpy(),
                      ' ds: ', self._optimisation_variables[2].detach().cpu().numpy() * 180.0 / (np.pi*10000),
                      ' e: ', self._optimisation_variables[3].detach().cpu().numpy()/10,
                      '')
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