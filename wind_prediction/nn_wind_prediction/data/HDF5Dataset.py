from __future__ import print_function

from nn_wind_prediction.utils.interpolation import DataInterpolation
import nn_wind_prediction.utils.generate_turbulence as generate_turbulence

import numpy as np
import random
import threading
import torch
from torch.utils.data.dataset import Dataset
import h5py
import nn_wind_prediction.utils as utils
import sys

numpy_interpolation = False
if sys.version_info[0] > 2:
    from interpolation.splines import UCGrid, eval_linear
    numpy_interpolation = True


class HDF5Dataset(Dataset):
    '''
    Class to handle the dataset with the containing velocities, pressure, turbulent kinetic energy, turbulent dissipation
     and turbulent viscosity.

    The dataset is a single hdf5 file containing groups for all of the samples which contain 4D arrays for each of the
    channels.
    The four dimensions are: [channels, z, y, x].
    The channel ordering is: [terrain, u_x, u_y, u_z, turb, p, epsilon, nut]

    The raw data is split up to an input tensor and label tensor. The input and label tensor channels are specified and
    contain information in the following order: [terrain, u_x, u_y*, u_z, turb, p, epsilon, nut], where uy_in is only
    present in the 3D case.
    The number of label  channels is configurable.
    The grid sizes (dx, dy, dz) are contained in the output tensor.

    TODO:
    - Check if it is feasible also to store the filedescriptors or how much faster it will make the dataloading (using Lock when accessing the file descriptors
    - Reimplement the 2D data handling
    - If possible, generalize augmentation modes to be non dependant on provided channels
    '''

    __default_scaling_dict = {'terrain': 1.0, 'ux': 1.0, 'uy': 1.0, 'uz': 1.0, 'turb': 1.0, 'p': 1.0, 'epsilon': 1.0, 'nut': 1.0}
    __lock = threading.Lock()

    def __init__(self, filename, input_channels, label_channels, **kwargs):
        '''
        Params:
            device:
                Device (CPU or GPU) on which the tensor operations are executed, default 'cpu'
            filename (required):
                The name of the tar file containing the dataset
            input_channels (required):
                A list of the channels to be returned in the input tensor
            label_channels (required):
                A list of the channels to be returned in the label tensor
            nx:
                Number of grid points in x-direction of the output, default 64
            ny:
                Number of grid points in y-direction of the output, default 64
            nz:
                Number of grid points in z-direction of the output, default 64
            input_mode:
                Indicates how the input is constructed. The following modes are currently implemented (default 0):
                    0: The inflow condition is copied over the full domain
                    1: The vertical edges are interpolated over the full domain
                    2: The inputs have the same values as the labels (channel wise)
                    3: Randomly sample a sparse mask over the full domain and return the values of those sampled cells
                    4: Randomly sample a sparse mask over the sub domain and return the values of those sampled cells
                    5: Fake trajectories and return the values at those cells
                    6: Sequential trajectory input (Not implement yet)
            augmentation:
                If true the data is augmented according to the mode and augmentation_kwargs. The terrain and the velocities
                must be requested to use this mode for now
            augmentation_mode:
                Specifies the data augmentation mode
                    0: Rotating and subsampling the data without interpolation (rotation in 90deg steps, shift in integer steps)
                    1: Rotating and subsampling the data with interpolation (continuous rotation, continous shift)
            stride_hor:
                Horizontal stride, used to reduce the size of the output tensors, default 1
            stride_vert:
                Vertical stride, used to reduce the size of the output tensors, default 1
            scaling_uhor:
                Scaling factor for the horizontal velocity components, default 1.0
            scaling_uz:
                Scaling factor for the vertical velocity component, default 1.0
            scaling_turb:
                Scaling factor for the turbulent kinetic energy, default 1.0
            scaling_p:
                Scaling factor for the pressure, default 1.0
            scaling_epsilon:
                Scaling factor for dissipation, default 1.0
            scaling_nut:
                Scaling factor for viscosity, default 1.0
            scaling_terrain:
                Scaling factor for terrain channel, default 1.0
            return_grid_size:
                If true a tuple of the grid size is returned in addition to the input and output tensors, default False
            return_name:
                Return the filename of the sample, default False
            autoscale:
                Automatically scale the input and return the scale, default False
            loss_weighting_fn:
                Compose a matrix to weight the indivial cells differently
                    0: no weighting function, weights are ones
                    1: squared pressure fluctuations
                    2: l2 norm of the pressure gradient
                    3: l2 norm of the velocity gradient
            loss_weighting_clamp:
                Bool indicating if the weights should be clamped to a certain maximum value
            additive_gaussian_noise:
                Boolean indicating if the gaussian noise is additive or a randomly sampled factor to the actual velocities
            max_gaussian_noise_std:
                Maximum standard deviation of the gaussian noise added to the input. If set to 0 no noise is added.
                This is a relative value as the noise is expressed in percent.
            n_turb_fields:
                Number of turbulence fields generated in the initialization to add turbulent disturbances to the
                wind velocities. If either n_turb_fields or max_normalized_turb_scale are 0 no disturbances are
                added.
            max_normalized_turb_scale:
                Maximum magnitude of the turbulence added to the normalized wind.
                If either n_turb_fields or max_normalized_turb_scale are 0 no disturbances are
                added.
            max_normalized_bias_scale:
                Maximum magnitude of the bias added to the normalized wind.
                If set to 0 no bias is added.
            only_z_velocity_bias:
                Indicates if the bias is added only to the z-velocities. If false the bias is added to all wind axes.
            max_fraction_of_sparse_data:
                In case of a sparse input this indicates the maximum number of cells that should be sampled.
            min_fraction_of_sparse_data:
                In case of a sparse input this indicates the minimum number of cells that should be sampled.
            use_system_random:
                If true the true system random generator is used, else the standart pseudo number generated
                is used where setting the seed is feasible
        '''
        # ------------------------------------------- kwarg fetching ---------------------------------------------------
        parser = utils.KwargsParser(kwargs, 'HDF5Dataset')
        verbose = parser.get_safe('verbose', False, bool, False)
        self.__loss_weighting_fn = parser.get_safe('loss_weighting_fn', 0, int, verbose)
        self.__loss_weighting_clamp = parser.get_safe('loss_weighting_clamp', True, bool, verbose)
        self.__device = parser.get_safe('device', 'cpu', str, verbose)
        self.__nx = parser.get_safe('nx', 64, int, verbose)
        self.__ny = parser.get_safe('ny', 64, int, verbose)
        self.__nz = parser.get_safe('nz', 64, int, verbose)
        self.__input_mode = parser.get_safe('input_mode', 0, int, verbose)
        self.__augmentation = parser.get_safe('augmentation', False, bool, verbose)
        self.__stride_hor = parser.get_safe('stride_hor', 1, int, verbose)
        self.__stride_vert = parser.get_safe('stride_vert', 1, int, verbose)
        self.__return_grid_size = parser.get_safe('return_grid_size', False, bool, verbose)
        self.__return_name = parser.get_safe('return_name', False, bool, verbose)
        self.__autoscale = parser.get_safe('autoscale', False, bool, verbose)
        self.__max_gaussian_noise_std = parser.get_safe('max_gaussian_noise_std', 0.0, float, verbose)
        self.__additive_gaussian_noise = parser.get_safe('additive_gaussian_noise', True, bool, verbose)
        self.__n_turb_fields = parser.get_safe('n_turb_fields', 0, int, verbose)
        self.__max_normalized_turb_scale = parser.get_safe('max_normalized_turb_scale', 0.0, float, verbose)
        self.__max_normalized_bias_scale = parser.get_safe('max_normalized_bias_scale', 0.0, float, verbose)
        self.__only_z_velocity_bias = parser.get_safe('only_z_velocity_bias', False, bool, verbose)
        self.__use_system_random = parser.get_safe('use_system_random', False, bool, verbose)
        self.__max_gaussian_noise_std = parser.get_safe('max_gaussian_noise_std', 0.0, float, verbose)

        if self.__input_mode == 3 or self.__input_mode == 4:
            self.__max_fraction_of_sparse_data = parser.get_safe('max_fraction_of_sparse_data', 1.0, float, verbose)
            self.__min_fraction_of_sparse_data = parser.get_safe('min_fraction_of_sparse_data', 0.0, float, verbose)

        if self.__augmentation:
            self.__augmentation_mode = parser.get_safe('augmentation_mode', 0, int, verbose)

            if self.__augmentation_mode == 0:
                default_dict = {'subsampling': True, 'rotating': True,}
                augmentation_kwargs = parser.get_safe('augmentation_kwargs', default_dict, dict, verbose)

                self.__subsample = augmentation_kwargs['subsampling']
                self.__rotating = augmentation_kwargs['rotating']

        # --------------------------------------- initializing class params --------------------------------------------
        if len(input_channels) == 0 or len(label_channels) == 0:
            raise ValueError('HDF5Dataset: List of input or label channels cannot be empty')

        if self.__loss_weighting_fn == 1:
            weighting_channels = ['p']
        elif self.__loss_weighting_fn == 2:
            weighting_channels = ['terrain', 'p']
        elif self.__loss_weighting_fn == 3:
            weighting_channels = ['terrain', 'ux', 'uy', 'uz']
        else:
            weighting_channels = []

        # make sure that all requested channels are possible
        self.default_channels = ['terrain', 'ux', 'uy', 'uz', 'turb', 'p', 'epsilon', 'nut']
        for channel in input_channels:
            if channel not in self.default_channels:
                raise ValueError('HDF5Dataset: Incorrect input_channel detected: \'{}\', '
                                 'correct channels are {}'.format(channel, self.default_channels))

        for channel in label_channels:
            if channel not in self.default_channels:
                raise ValueError('HDF5Dataset: Incorrect label_channel detected: \'{}\', '
                                 'correct channels are {}'.format(channel, self.default_channels))

        self.__channels_to_load = []
        self.__input_channels = []
        self.__input_indices = []
        self.__label_indices = []
        self.__input_velocities_indices = []
        self.__data_velocity_indices = []

        # make sure that the channels_to_load list is correctly ordered, and save the input and label variable indices
        index = 0
        for channel in self.default_channels:
            if channel in input_channels or channel in label_channels or channel in weighting_channels:
                self.__channels_to_load += [channel]
                if channel in input_channels:
                    self.__input_channels += [channel]
                    self.__input_indices += [index]
                    if channel in ['ux', 'uy', 'uz']:
                        self.__data_velocity_indices += [index]
                        self.__input_velocities_indices += [len(self.__input_indices) - 1]
                if channel in label_channels:
                    self.__label_indices += [index]
                index += 1

        self.__input_indices = torch.LongTensor(self.__input_indices)
        self.__label_indices = torch.LongTensor(self.__label_indices)

        self.__filename = filename

        try:
            h5_file = h5py.File(filename, 'r')
        except IOError as e:
            print('I/O error({0}): {1}: {2}'.format(e.errno, e.strerror, filename))
            sys.exit()

        # extract info from the h5 file
        self.__num_files = len(h5_file.keys())
        self.__memberslist = list(h5_file.keys())
        h5_file.close()

        # check that all the required channels are present for autoscale for augmentation
        # this is due to the get_scale method which needs the the velocities to compute the scale for autoscaling
        # augmentation mode 1 and 0 use indexing, the first 4 channels are required
        if not all(elem in self.__channels_to_load for elem in ['terrain', 'ux', 'uy', 'uz']):
            print('HDF5Dataset: augmentation and autoscale will not be applied, not all of the required channels (terrain,ux, uy, uz) were requested')
            self.__autoscale = False
            self.__augmentation = False


        # create scaling dict for each channel
        self.__scaling_dict = dict()
        for channel in self.__channels_to_load:
            try:
                self.__scaling_dict[channel] = kwargs['scaling_' + channel]
            except KeyError:
                self.__scaling_dict[channel] = self.__default_scaling_dict[channel]
                if verbose:
                    print('HDF5Dataset: ', 'scaling_' + channel, 'not present in kwargs, using default value:',
                          self.__default_scaling_dict[channel])

        # create turbulent velocity fields for train set
        if self.__n_turb_fields > 0 and self.__max_normalized_turb_scale > 0.0:
            turbulent_velocity_fields = []
            for i in range(self.__n_turb_fields):
                # dx, dy, dz are hardcoded for the moment
                dx = dy = 16.6444
                dz = 11.5789
                turbulent_velocity_field, _ = generate_turbulence.generate_turbulence_spectral(int(self.__nx * 1.5), int(self.__ny * 1.5), int(self.__ny * 1.5), dx, dy, dz)
                turbulent_velocity_fields += [torch.from_numpy(turbulent_velocity_field.astype(np.float32)).unsqueeze(0)]
            self.__turbulent_velocity_fields = torch.cat(turbulent_velocity_fields, 0)

        # add the mask to the input channels for the respective input modes
        if self.__input_mode == 3 or self.__input_mode == 4 or self.__input_mode == 5:
            self.__input_channels += ['mask']

        # initialize random number generator used for the subsampling
        if self.__use_system_random:
            self.__rand = random.SystemRandom()
        else:
            self.__rand = random

        # interpolator for the three input velocities
        self.__interpolator = DataInterpolation(self.__device, len(self.__input_velocities_indices), self.__nx, self.__ny, self.__nz)

        # avoids printing a warning multiple times
        self.__augmentation_warning_printed = False

        self.__min_num_cells = self.__nx * self.__ny * self.__nz
        self.__max_num_cells = 0
        self.__num_samples = 0
        self.__average_num_cells = 0

        if verbose:
            print('HDF5Dataset: ' + filename + ' contains {} samples'.format(self.__num_files))

    def __getitem__(self, index):
        h5_file = h5py.File(self.__filename, 'r', swmr=True)
        sample = h5_file[self.__memberslist[index]]

        # load the data
        data_from_channels = []
        for i, channel in enumerate(self.__channels_to_load):
            # extract channel data and apply scaling
            data_from_channels += [torch.from_numpy(sample[channel][...]).float().unsqueeze(0) / self.__scaling_dict[channel]]

        data = torch.cat(data_from_channels, 0)

        # send full data to device
        data = data.to(self.__device)

        ds = torch.from_numpy(sample['ds'][...])

        data_shape = data[0, :].shape

        # 3D data transforms
        if (len(data_shape) == 3):
            # apply autoscale if requested
            if self.__autoscale:
                # determine the scales
                noise_scale = 1.0
                scale = self.get_scale(data[self.__data_velocity_indices, :, :, :])

                # applying the autoscale to the velocities
                data[self.__data_velocity_indices, :, :, :] /= scale

                # turb handling
                if 'turb' in self.__channels_to_load:
                    data[self.__channels_to_load.index('turb'), :, :, :] /= scale*scale

                # p handling
                if 'p' in self.__channels_to_load:
                    data[self.__channels_to_load.index('p'),:,:,:] /= scale*scale

                # epsilon handling
                if 'epsilon' in self.__channels_to_load:
                    data[self.__channels_to_load.index('epsilon'),:,:,:] /= scale*scale*scale

                # nut handling
                if 'nut' in self.__channels_to_load:
                    data[self.__channels_to_load.index('nut'), :, :, :] /= scale

            else:
                # determine the scale of the sample to properly scale the noise
                noise_scale = self.get_scale(data[self.__data_velocity_indices, :, :, :])

            # downscale if requested
            data = data[:, ::self.__stride_vert, ::self.__stride_hor, ::self.__stride_hor]

            # augment if requested according to the augmentation mode
            if self.__augmentation:
                if self.__augmentation_mode == 0:
                    # subsampling
                    if self.__subsample:
                        start_x = self.__rand.randint(0,data_shape[2]-self.__nx)
                        start_y = self.__rand.randint(0,data_shape[1]-self.__ny)
                        start_z = int(self.__rand.triangular(0,(data_shape[0]-self.__nz),0)) # triangle distribution

                        data = data[:, start_z:start_z+self.__nz,  start_y:start_y+self.__ny,  start_x:start_x+self.__nx]
                    else:
                        # select the first indices
                        data = data[:,:self.__nz, :self.__ny, :self.__nx]

                    # rotating and flipping
                    if self.__rotating:
                        # flip in x-direction
                        if (self.__rand.randint(0,1)):
                            data = data.flip(3)
                            data[1,:,:,:] *= -1.0

                        # flip in y-direction
                        if (self.__rand.randint(0,1)):
                            data = data.flip(2)
                            data[2,:,:,:] *= -1.0

                        # rotate 90 degrees
                        if (self.__rand.randint(0,1)):
                            data = data.transpose(2,3).flip(3)
                            data = torch.cat((data[[0,2,1]], data[3:]),0)
                            data[1,:,:,:] *= -1.0

                            ds = torch.tensor([ds[1], ds[0], ds[2]])

                elif self.__augmentation_mode == 1:
                    if numpy_interpolation:
                        # use the numpy implementation as it is more accurate and slightly faster
                        # and python 3 is used
                        data = self.__augmentation_mode2_numpy(data, self.__nx) # u_x: index 1, u_y: index 2
                    else:
                        # use the torch version for python 2
                        data = self.__augmentation_mode2_torch(data, self.__nx) # u_x: index 1, u_y: index 2

                    # flip in x-direction
                    if (self.__rand.randint(0,1)):
                        data = data.flip(3)
                        data[1,:,:,:] *= -1.0

                    # flip in y-direction
                    if (self.__rand.randint(0,1)):
                        data = data.flip(2)
                        data[2,:,:,:] *= -1.0

                else:
                    if not self.__augmentation_warning_printed:
                        print('WARNING: Unknown augmentation mode in HDF5Dataset ', self.__augmentation_mode,
                              ', not augmenting the data')

            else:
                # no data augmentation
                data = data[:, :self.__nz, :self.__ny, :self.__nx]

            # generate the input data
            input_data = torch.index_select(data, 0, self.__input_indices)
            boolean_terrain = input_data[0] > 0

            # add gaussian noise to wind velocities if requested
            if self.__max_gaussian_noise_std > 0.0:
                if self.__additive_gaussian_noise:
                    std = self.__rand.random() * self.__max_gaussian_noise_std * noise_scale
                    input_data[self.__input_velocities_indices] += torch.randn(input_data[self.__input_velocities_indices].shape) * std
                else:
                    std = self.__rand.random() * self.__max_gaussian_noise_std
                    input_data[self.__input_velocities_indices] *= torch.randn(input_data[self.__input_velocities_indices].shape) * std + 1.0

            # add turbulence to wind velocities if requested
            if self.__n_turb_fields > 0 and self.__max_normalized_turb_scale > 0.0:
                # randomly select one of the turbulent velocity fields
                rand_idx = self.__rand.randint(0, self.__n_turb_fields - 1)
                turb_scale = self.__rand.random() * self.__max_normalized_turb_scale
                turbulence = self.__turbulent_velocity_fields[rand_idx]
                _, turb_nx, turb_ny, turb_nz = turbulence.shape

                # subsample turbulent velocity field with the same shape as data
                _, nz, ny, nx = input_data.shape
                start_x = self.__rand.randint(0, turb_nx - nx)
                start_y = self.__rand.randint(0, turb_ny - ny)
                start_z = self.__rand.randint(0, turb_nz - nz)
                turbulence = turbulence[:, start_z:start_z+nz,  start_y:start_y+ny,  start_x:start_x+nx] * turb_scale * noise_scale

                # mask by terrain
                turbulence *= boolean_terrain

                if 'ux' in self.__input_channels:
                    input_data[self.__input_channels.index('ux')] += turbulence[0]
                if 'uy' in self.__input_channels:
                    input_data[self.__input_channels.index('uy')] += turbulence[1]
                if 'uz' in self.__input_channels:
                    input_data[self.__input_channels.index('uz')] += turbulence[2]

            # add bias to wind velocities if requested
            if self.__max_normalized_bias_scale > 0:
                if self.__only_z_velocity_bias:
                    if 'uz' in self.__input_channels:
                        bias_scale = self.__rand.random() * self.__max_normalized_bias_scale
                        bias = (torch.rand(1) * 2.0 - 1) * bias_scale * noise_scale
                        input_data[self.__input_channels.index('uz')] += bias
                    else:
                        print('Adding bias only for uz requested but uz not present, not adding any bias')
                else:
                    bias_scale = self.__rand.random() * self.__max_normalized_bias_scale
                    bias = (torch.rand(len(self.__input_velocities_indices), 1, 1, 1) * 2.0 - 1) * bias_scale * noise_scale
                    input_data[self.__input_velocities_indices] += bias

            # assemble the input according to the mode
            if self.__input_mode == 0:
                # copy the inflow condition across the full domain
                input = torch.cat((input_data[0,:,:,:].unsqueeze(0),
                                   input_data[1:,:,:,0].unsqueeze(-1).expand(-1,-1,-1,self.__nx)))

            elif self.__input_mode == 1:
                # This interpolation is slower (at least on a cpu)
                # input = torch.cat([data[0,:,:,:].unsqueeze(0),
                # self.__interpolator.edge_interpolation_batch(data[1:4,:,:,:].unsqueeze(0)).squeeze()])

                # interpolating the vertical edges
                input = torch.cat((input_data[0,:,:,:].unsqueeze(0),
                                    self.__interpolator.edge_interpolation(input_data[1:,:,:,:])))

            elif (self.__input_mode == 2):
                # Input the ground truth data
                input = input_data

            elif (self.__input_mode == 3):
                # random sparse input with an additional mask as an input
                mask =  self.__create_sparse_mask(boolean_terrain)
                self.__update_sparse_stats(mask.sum().item())

                # compose output
                input_data[1:] *= mask
                input = torch.cat([input_data, mask.float().unsqueeze(0)])

            elif (self.__input_mode == 4):
                # random sparse input from a subregion with an additional mask as an input
                mask = self.__create_sparse_mask_subregion(boolean_terrain)
                self.__update_sparse_stats(mask.sum().item())

                # compose output
                input_data[1:] *= mask
                input = torch.cat([input_data, mask.float().unsqueeze(0)])

            elif (self.__input_mode == 5):
                # faking input from a flight path
                mask = self.__create_sparse_mask_trajectory(boolean_terrain)
                self.__update_sparse_stats(mask.sum().item())

                # compose output
                input_data[1:] *= mask
                input = torch.cat([input_data, mask.float().unsqueeze(0)])

            elif (self.__input_mode == 6):
                # sequential input
                print('HDF5Dataset Error: Sequential input is not implemented anymore')
                sys.exit()

            else:
                print('HDF5Dataset Error: Input mode ', self.__input_mode, ' is not supported')
                sys.exit()

            # generate the input channels
            label = torch.index_select(data, 0, self.__label_indices)

            # generate the loss weighting matrix
            loss_weighting_matrix = self.__compute_loss_weighting(data, ds, self.__loss_weighting_fn)

            out = [input, label, loss_weighting_matrix]

            if self.__autoscale:
                out.append(scale)

            if self.__return_grid_size:
                out.append(ds)

            if self.__return_name:
                out.append(self.get_name(index))

            return out

        elif (len(data_shape) == 2):
            print('HDF5Dataset Error: 2D data handling is not implemented anymore')
            sys.exit()
        else:
            print('HDF5Dataset Error: Data dimension of ', len(data_shape), ' is not supported')
            sys.exit()

    def get_name(self, index):
        return self.__memberslist[index]

    def get_input_channels(self):
        return self.__input_channels

    def print_dataset_stats(self):
        if self.__input_mode == 3 or self.__input_mode == 4 or self.__input_mode == 5:
            print('-------------------------------------------------')
            print('HDF5 Dataset sparse mask statistics')
            print('\tminimum number of cells:', self.__min_num_cells)
            print('\tmaximum number of cells:', self.__max_num_cells)
            print('\taverage number of cells:', self.__average_num_cells)
            print('\tnumber of mask created: ', self.__num_samples)
            print('-------------------------------------------------')
        else:
            pass

    def __len__(self):
        return self.__num_files

    def __generate_interpolation_grid(self, data_shape, out_size):
        # generate the initial unrotated grid
        x_f = np.arange(out_size)
        y_f = np.arange(out_size)
        z_f = np.arange(out_size)
        Z, Y, X = np.meshgrid(z_f, y_f, x_f, indexing='ij')

        # sample the grid orientation
        ratio = data_shape[3] / out_size
        if ratio > np.sqrt(2):
            # fully rotation possible
            angle = self.__rand.uniform(0, 2.0*np.pi)
        elif (ratio <= 1.0):
            # no rotation possible
            angle = 0.0

        else:
            # some angles are infeasible
            angle_found = False
            lower_bound = 2 * np.arctan((1 - np.sqrt(2 - ratio*ratio)) / (1+ratio))
            upper_bound = 0.5 * np.pi - lower_bound

            while (not angle_found):
                angle = self.__rand.uniform(0, 0.5*np.pi)

                if not ((angle > lower_bound) and (angle < upper_bound)):
                    angle_found = True

            angle += 0.5*np.pi * self.__rand.randint(0, 3)

        # rotate the grid
        X_rot = np.cos(angle) * X - np.sin(angle) * Y
        Y_rot = np.sin(angle) * X + np.cos(angle) * Y

        # determine the shift limits
        x_minshift = -X_rot.min()
        x_maxshift = data_shape[3] - 1 - X_rot.max()
        y_minshift = -Y_rot.min()
        y_maxshift = data_shape[2] - 1 - Y_rot.max()
        z_minshift = 0.0
        z_maxshift = data_shape[1] - out_size

        # shift the grid
        X = X_rot + self.__rand.uniform(x_minshift, x_maxshift)
        Y = Y_rot + self.__rand.uniform(y_minshift, y_maxshift)
        Z = Z     + int(self.__rand.triangular(z_minshift,z_maxshift,z_minshift))

        # check if the shifted/rotated grid is fine
        if ((X.min() < 0.0) or
            (Y.min() < 0.0) or
            (Z.min() < 0.0) or
            (X.max() > data_shape[3] - 1) or
            (Y.max() > data_shape[2] - 1) or
            (Z.max() > data_shape[1] - 1)):
            raise RuntimeError("The rotated and shifted grid does not satisfy the data grid bounds")

        return X, Y, Z, angle

    def __augmentation_mode2_torch(self, data, out_size, vx_channel = 1, vy_channel = 2):
        '''
        Augment the data by random subsampling and rotation the data.
        This approach has errors in the order of 10^-6 even when resampling on the original grid.
        The output is a cubical grid with shape (N_channels, out_size, out_size, out_size)

        Assumptions:
            - The input is 4D (channels, z, y, x)
            - The number of cells is equal in x- and y-dimension
            - The number of cells in each dimension is at least as large as the out_size

        Inputs:
            - data: The input data
            - out_size: The size of the output grid
            - vx_channel: The channel of the x velocity
            - vy_channel: The channel of the y velocity
        '''
        if not len(data.shape) == 4:
            raise ValueError('The input dimension of the data array needs to be 4 (channels, z, y, x)')

        if data.shape[2] != data.shape[3]:
            raise ValueError('The number of cells in x- and y-direction must be the same, shape: ', data.shape)

        if out_size > data.shape[3]:
            raise ValueError('The number of output cells cannot be larger than the input')

        # generate the interpolation grid
        X, Y, Z, angle = self.__generate_interpolation_grid(data.shape, out_size)
        X, Y, Z = torch.from_numpy(X).float().unsqueeze(0).unsqueeze(-1), torch.from_numpy(Y).float().unsqueeze(0).unsqueeze(-1), torch.from_numpy(Z).float().unsqueeze(0).unsqueeze(-1)

        # convert the coordinates to a range of -1 to 1 as required by grid_sample
        X = 2.0 * X / (data.shape[3] - 1.0) - 1.0
        Y = 2.0 * Y / (data.shape[2] - 1.0) - 1.0
        Z = 2.0 * Z / (data.shape[1] - 1.0) - 1.0

        # assemble the final interpolation grid and interpolate
        grid = torch.cat((X, Y, Z), dim = 4)
        interpolated = torch.nn.functional.grid_sample(data.unsqueeze(0), grid).squeeze()

        # rotate also the horizontal velocities
        vel_x =  np.cos(angle) * interpolated[vx_channel] + np.sin(angle) * interpolated[vy_channel]
        vel_y = -np.sin(angle) * interpolated[vx_channel] + np.cos(angle) * interpolated[vy_channel]
        interpolated[vx_channel] = vel_x
        interpolated[vy_channel] = vel_y

        # fix the terrain
        terrain = interpolated[0]
        terrain[terrain<=0.5 / self.__scaling_terrain] = 0

        return interpolated

    def __augmentation_mode2_numpy(self, data, out_size, vx_channel = 1, vy_channel = 2):
        '''
        Augment the data by random subsampling and rotation the data.
        The output is a cubical grid with shape (N_channels, out_size, out_size, out_size)

        Assumptions:
            - The input is 4D (channels, z, y, x)
            - The number of cells is equal in x- and y-dimension
            - The number of cells in each dimension is at least as large as the out_size

        Inputs:
            - data: The input data
            - out_size: The size of the output grid
            - vx_channel: The channel of the x velocity
            - vy_channel: The channel of the y velocity
        '''
        if not len(data.shape) == 4:
            raise ValueError('The input dimension of the data array needs to be 4 (channels, z, y, x)')

        if data.shape[2] != data.shape[3]:
            raise ValueError('The number of cells in x- and y-direction must be the same, shape: ', data.shape)

        if out_size > data.shape[3]:
            raise ValueError('The number of output cells cannot be larger than the input')

        # generate the existing grid and data
        values = np.moveaxis(data.numpy(), 0, -1)
        grid = UCGrid((0,values.shape[0]-1,values.shape[0]),(0,values.shape[1]-1,values.shape[1]),(0,values.shape[2]-1,values.shape[2]))

        # generate the interpolation grid
        X, Y, Z, angle = self.__generate_interpolation_grid(data.shape, out_size)

        # interpolate
        points = np.stack((Z.ravel(), Y.ravel(), X.ravel()), axis=1)
        interpolated = eval_linear(grid, values, points)
        interpolated = interpolated.reshape((out_size,out_size,out_size, values.shape[3]))
        interpolated = np.moveaxis(interpolated, -1, 0)

        # rotate also the horizontal velocities
        vel_x =  np.cos(angle) * interpolated[vx_channel] + np.sin(angle) * interpolated[vy_channel]
        vel_y = -np.sin(angle) * interpolated[vx_channel] + np.cos(angle) * interpolated[vy_channel]
        interpolated[vx_channel] = vel_x
        interpolated[vy_channel] = vel_y

        return torch.from_numpy(interpolated).float()

    def get_scale(self, x):
        shape = x.shape

        # get data from corners of sample
        corners = torch.index_select(x, 2, torch.tensor([0, shape[2]-1]))
        corners = torch.index_select(corners, 3, torch.tensor([0, shape[3]-1]))

        # compute the scale
        scale = corners.norm(dim=0).mean(dim=0).max()

        # account for unfeasible scales (e.g. from zero samples)
        if scale > 0:
            return scale
        else:
            return torch.tensor(1.0)

    def __create_sparse_mask(self, boolean_terrain):
        '''
        Creating a randomly sampled sparse mask making sure only non-terrain cells are sampled.

        Input params:
            boolean_terrain: boolean representation of the terrain (False for terrain cells, True for wind cells)

        Output:
            mask: The sampled mask
        '''
        mask = torch.zeros_like(boolean_terrain)
        mask_shape = mask.shape

        # percentage of sparse data
        target_frac = self.__rand.random() * (self.__max_fraction_of_sparse_data - self.__min_fraction_of_sparse_data) + self.__min_fraction_of_sparse_data
        target_num_cell = int(mask.numel() * target_frac)

        # set the initial number of cells to a negative value to correct for the expected number of cells samples inside the terrain
        mask_num_cell = int(target_num_cell * (1.0 - 1.0 / boolean_terrain.float().mean().item()))

        # iterate multiple times to make sure the right amount of cells are set in the mask
        max_iter = 2
        iter = 0
        while iter < max_iter and mask_num_cell < target_num_cell:
            max_iter = max_iter + 1

            # sample cells
            indices = self.__rand.sample(range(mask.numel()), min(int(target_num_cell - mask_num_cell), mask.numel()))
            mask = mask.view(-1)
            mask[indices] = 1.0
            mask = mask.view(mask_shape)

            # make sure no cells are inside the terrain
            mask *= boolean_terrain
            mask_num_cell = mask.sum().item()

        return mask

    def __create_sparse_mask_subregion(self, boolean_terrain):
        '''
        Creating a randomly sampled sparse mask making sure only non-terrain cells are sampled and all samples
        are located in a smaller subregion of the full domain.

        Input params:
            boolean_terrain: boolean representation of the terrain (False for terrain cells, True for wind cells)

        Output:
            mask: The sampled mask
        '''
        mask = torch.zeros_like(boolean_terrain)
        mask_shape = mask.shape

        # allow for a maximum of 10 tries to get a mask containing non terrain cells
        iter = 0
        max_iter = 10
        inside_terrain = True
        while (iter < max_iter and inside_terrain):
            # create a subregion which is between 0.25 and 0.75 in each dimension
            sub_mask_shape = torch.Size((torch.tensor(mask_shape) * (torch.rand(3) * 0.5 + 0.25)).to(torch.long))
            sub_mask = torch.zeros(sub_mask_shape)
            sub_mask_start_idx = torch.zeros(3).to(torch.long)
            sub_mask_start_idx[0] = int(self.__rand.triangular(0, mask_shape[0] - sub_mask_shape[0], 0))
            sub_mask_start_idx[1] = self.__rand.randint(0, mask_shape[1] - sub_mask_shape[1])
            sub_mask_start_idx[2] = self.__rand.randint(0, mask_shape[2] - sub_mask_shape[2])

            mean_terrain = boolean_terrain[sub_mask_start_idx[0]:sub_mask_start_idx[0] + sub_mask_shape[0],
                                           sub_mask_start_idx[1]:sub_mask_start_idx[1] + sub_mask_shape[1],
                                           sub_mask_start_idx[2]:sub_mask_start_idx[2] + sub_mask_shape[2]].float().mean().item()

            inside_terrain = mean_terrain == 0
            iter += 1

        if iter == max_iter:
            print('Did not find a valid mask within ' + str(iter) + 'iterations')

        # determine the number of cells to sample and correct the number of cells by the subregion size and the terrain occlusion
        target_frac = self.__rand.random() * (self.__max_fraction_of_sparse_data - self.__min_fraction_of_sparse_data) + self.__min_fraction_of_sparse_data

        # limit it to a maximum 50 % of the cells to limit the cases where all the cells are sampled
        #target_num_cell = min(int(mask.numel() * target_frac), int(0.5 * sub_mask.numel()))
        target_num_cell = int(sub_mask.numel() * target_frac)

        # set the initial number of cells to a negative value to correct for the expected number of cells samples inside the terrain
        if inside_terrain:
            terrain_factor = 1.0
        else:
            terrain_factor = 1.0 / mean_terrain

        # limit it to 80 % of the cells to still ensure there are some free cells
        target_num_cell = min(int(target_num_cell * terrain_factor), int(0.8 * sub_mask.numel()))

        # sample cells
        indices = self.__rand.sample(range(sub_mask.numel()), target_num_cell)
        sub_mask = sub_mask.view(-1)
        sub_mask[indices] = 1.0
        sub_mask = sub_mask.view(sub_mask_shape)

        mask[sub_mask_start_idx[0]:sub_mask_start_idx[0] + sub_mask_shape[0],
             sub_mask_start_idx[1]:sub_mask_start_idx[1] + sub_mask_shape[1],
             sub_mask_start_idx[2]:sub_mask_start_idx[2] + sub_mask_shape[2]] = sub_mask

        # make sure no cells are inside the terrain
        mask *= boolean_terrain

        return mask

    def __create_sparse_mask_trajectory(self, boolean_terrain):
        '''
        Creating a sparse mask by faking a trajectory of a flying UAV.

        Input params:
            boolean_terrain: boolean representation of the terrain (False for terrain cells, True for wind cells)

        Output:
            mask: The sampled mask
        '''
        # TODO: Currently the params are hardcoded. Determine if they should be added to the yaml files.
        mask = torch.zeros_like(boolean_terrain)
        mask_shape = mask.shape

        # define some trajectory parameter
        min_trajectory_length = 30
        max_trajectory_length = 300
        min_segment_length = 5
        max_segment_length = 20
        step_size = 1.0
        max_iter = 50

        # initialize a random valid start position
        valid_start_positions = torch.nonzero(boolean_terrain, as_tuple=False)
        position = valid_start_positions[self.__rand.randint(0, valid_start_positions.shape[0]-1)]
        mask[position.split(1)] = 1.0

        # randomly determine a target trajectory length
        trajectory_length = self.__rand.randint(min_trajectory_length, max_trajectory_length)

        # loop through adding segments until target length is achieved
        iter = 0
        while (iter < max_iter) and mask.sum() < trajectory_length:
            iter += 1
            segment_length = self.__rand.randint(min_segment_length, max_segment_length)

            # sample random propagation direction, divide z component by 4 to make flatter trajectories more likely
            direction = torch.randn(3)
            direction[0] *= 0.25
            direction /= direction.norm()

            num_steps = int(segment_length / step_size)
            for i in range(num_steps):
                new_position = position + step_size * direction
                new_idx = torch.round(new_position).to(torch.long)

                # check if the new position is inside the domain, if not invert the respective direction
                if torch.prod(new_idx > 0) * torch.prod(new_idx + 1 < torch.tensor(mask_shape)):
                    # check if new position is not inside the terrain
                    if (boolean_terrain[new_idx.split(1)]):
                        mask[new_idx.split(1)] = 1.0
                        position = new_position
                    else:
                        break
                else:
                    direction *= (new_idx > 0) * 2 - 1
                    direction *= (new_idx + 1 < torch.tensor(mask_shape)) * 2 - 1

        return mask

    def __update_sparse_stats(self, num_cells):
        self.__lock.acquire()
        self.__min_num_cells = min(self.__min_num_cells, num_cells)
        self.__max_num_cells = max(self.__max_num_cells, num_cells)
        if (self.__num_samples < 1):
            self.__num_samples = 1
            self.__average_num_cells = num_cells
        else:
            self.__num_samples += 1
            self.__average_num_cells += (num_cells - self.__average_num_cells) / float(self.__num_samples)

        self.__lock.release()

    def __compute_loss_weighting(self, data, ds, weighting_fn=0):
        '''
        This function computes the matrix to be used for loss weighting. Different weighting functions can be used, but all
        are normalized, so that the mean of W for an individual sample is 1.

        Input params:
            data: 4D tensor [channels, Z, Y, X]
            grid_size: array of size 3 containing the sizes of the grid in X, Y and Z.
            weighting_fn: switches between the weighting functions:
                            0: no weighting function, weights are ones
                            1: squared pressure fluctuations
                            2: l2 norm of the pressure gradient
                            3: l2 norm of the velocity gradient

        Output:
            W: 4D tensor [weighting, Z, Y, X] containing the pixel-wise weights.
        '''
        # no weighting, return empty tensor
        if weighting_fn == 0:
            return torch.Tensor([])

        # squared pressure fluctuations weighting function
        if weighting_fn == 1:

            # get pressure and mean pressure per sample
            p_index = self.__channels_to_load.index('p')
            p = data[p_index].unsqueeze(0)
            p_mean = p.mean(-1).mean(-1).mean(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(p)

            # remove mean, square and remove outliers
            W = (p - p_mean) ** 2

            if self.__loss_weighting_clamp:
                # TODO: make the clamping value a parameter that can be set from the YAML config file
                W = W.clamp(0.0435)

            # normalize by its volume integral per sample
            W =  W/ (W.mean(-1).mean(-1).mean(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(p))

        # l2-norm pressure gradient weighting function
        elif weighting_fn == 2:
            # get pressure
            p = data[self.__channels_to_load.index('p')].unsqueeze(0)

            # get terrain
            terrain = data[self.__channels_to_load.index('terrain')].unsqueeze(0)

            # compute the spatial pressure gradient components and take the l2-norm of the gradient and remove outliers
            W = (utils.derive(p, 3, ds[0], terrain) ** 2
                 + utils.derive(p, 2, ds[1], terrain) ** 2 +
                 utils.derive(p, 1, ds[2], terrain) ** 2) ** 0.5

            if self.__loss_weighting_clamp:
                # TODO: make the clamping value a parameter that can be set from the YAML config file
                W = W.clamp(0.000814)

            # normalize by its volume integral per sample
            W = (W / ((W).mean(-1).mean(-1).mean(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(W)))

        # l2-norm velocity gradient weighting function
        elif weighting_fn == 3:

            vel_indices = torch.LongTensor([self.__channels_to_load.index(channel) for channel in ['ux', 'uy', 'uz']])

            # get the velocities
            U = data.index_select(0, vel_indices).unsqueeze(0)

            # get terrain
            terrain = data[self.__channels_to_load.index('terrain')].unsqueeze(0)

            # compute the spatial gradient tensor of the velocity gradient and take the l2-norm of the gradient per sample and remove outliers
            W = (utils.gradient(U, ds, terrain) ** 2).sum(1) ** 0.5

            if self.__loss_weighting_clamp:
                # TODO: make the clamping value a parameter that can be set from the YAML config file
                W = W.clamp(0.00175)

            # normalize by its volume integral per sample
            W = (W / ((W).mean(-1).mean(-1).mean(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(W)))

        else:
            raise ValueError('Unrecognized weighting function.')

        # handling for zero samples which create NaNs
        W[torch.isnan(W)] = 1.0
        return W