from __future__ import print_function

from nn_wind_prediction.utils.interpolation import DataInterpolation

from interpolation.splines import UCGrid, eval_linear
import numpy as np
import random
import torch
from torch.utils.data.dataset import Dataset
import h5py
import nn_wind_prediction.utils as utils
import sys
sys.path.append('../')
from wind_analysis.analysis_utils import generate_turbulence


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

    __default_device = 'cpu'
    __default_nx = 64
    __default_ny = 64
    __default_nz = 64
    __default_input_mode = 0
    __default_augmentation = False
    __default_augmentation_mode = 0
    __default_augmentation_kwargs = {'subsampling': True, 'rotating': True,}
    __default_stride_hor = 1
    __default_stride_vert = 1
    __default_scaling_dict = {'terrain': 1.0, 'ux': 1.0, 'uy': 1.0, 'uz': 1.0, 'turb': 1.0, 'p': 1.0, 'epsilon': 1.0, 'nut': 1.0}
    __default_return_grid_size = False
    __default_return_name = False
    __default_autoscale = False
    __default_loss_weighting_fn = 0
    __default_loss_weighting_clamp = True
    __default_create_sparse_mask = False
    __default_max_percentage_of_sparse_data = 1
    __default_terrain_percentage_correction = False
    __default_sample_terrain_region = False
    __default_add_gaussian_noise = False
    __default_add_turbulence = False

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
                Indicates how the input is constructed. The following modes are currently implemented:
                    0: The inflow condition is copied over the full domain
                    1: The vertical edges are interpolated over the full domain, default 0
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
        '''

        # ------------------------------------------- kwarg fetching ---------------------------------------------------
        try:
            verbose = kwargs['verbose']
        except KeyError:
            verbose = False

        try:
            self.__loss_weighting_fn = kwargs['loss_weighting_fn']
        except KeyError:
            self.__loss_weighting_fn = self.__default_loss_weighting_fn
            if verbose:
                print('HDF5Dataset: loss_weighting_fn not present in kwargs, using default value:', self.__default_loss_weighting_fn)

        try:
            self.__loss_weighting_clamp = kwargs['loss_weighting_clamp']
        except KeyError:
            self.__loss_weighting_clamp = self.__default_loss_weighting_clamp
            if verbose:
                print('HDF5Dataset: loss_weighting_clamp not present in kwargs, using default value:', self.__default_loss_weighting_clamp)

        try:
            self.__device = kwargs['device']
        except KeyError:
            self.__device = self.__default_device
            if verbose:
                print('HDF5Dataset: device not present in kwargs, using default value:', self.__default_device)

        try:
            self.__nx = kwargs['nx']
        except KeyError:
            self.__nx = self.__default_nx
            if verbose:
                print('HDF5Dataset: nx not present in kwargs, using default value:', self.__default_nx)

        try:
            self.__ny = kwargs['ny']
        except KeyError:
            self.__ny = self.__default_ny
            if verbose:
                print('HDF5Dataset: ny not present in kwargs, using default value:', self.__default_ny)

        try:
            self.__nz = kwargs['nz']
        except KeyError:
            self.__nz = self.__default_nz
            if verbose:
                print('HDF5Dataset: nz not present in kwargs, using default value:', self.__default_nz)

        try:
            self.__input_mode = kwargs['input_mode']
        except KeyError:
            self.__input_mode = self.__default_input_mode
            if verbose:
                print('HDF5Dataset: input_mode not present in kwargs, using default value:', self.__default_input_mode)

        try:
            self.__augmentation = kwargs['augmentation']
        except KeyError:
            self.__augmentation = self.__default_augmentation
            if verbose:
                print('HDF5Dataset: augmentation not present in kwargs, using default value:', self.__default_augmentation)

        try:
            self.__augmentation_mode = kwargs['augmentation_mode']
        except KeyError:
            self.__augmentation_mode = self.__default_augmentation_mode
            if verbose:
                print('HDF5Dataset: augmentation_mode not present in kwargs, using default value:', self.__default_augmentation_mode)

        try:
            self.__stride_hor = kwargs['stride_hor']
        except KeyError:
            self.__stride_hor = self.__default_stride_hor
            if verbose:
                print('HDF5Dataset: stride_hor not present in kwargs, using default value:', self.__default_stride_hor)

        try:
            self.__stride_vert = kwargs['stride_vert']
        except KeyError:
            self.__stride_vert = self.__default_stride_vert
            if verbose:
                print('HDF5Dataset: stride_vert not present in kwargs, using default value:', self.__default_stride_vert)

        try:
            self.__return_grid_size = kwargs['return_grid_size']
        except KeyError:
            self.__return_grid_size = self.__default_return_grid_size
            if verbose:
                print('HDF5Dataset: return_grid_size not present in kwargs, using default value:', self.__default_return_grid_size)

        try:
            self.__return_name = kwargs['return_name']
        except KeyError:
            self.__return_name = self.__default_return_name
            if verbose:
                print('HDF5Dataset: return_name not present in kwargs, using default value:', self.__default_return_name)

        try:
            self.__autoscale = kwargs['autoscale']
        except KeyError:
            self.__autoscale = self.__default_autoscale
            if verbose:
                print('HDF5Dataset: autoscale not present in kwargs, using default value:', self.__default_autoscale)

        try:
            self.__create_sparse_mask = kwargs['create_sparse_mask']
        except KeyError:
            self.__create_sparse_mask = self.__default_create_sparse_mask
            if verbose:
                print('HDF5Dataset: create_sparse_mask not present in kwargs, using default value:', self.__default_create_sparse_mask)

        try:
            self.__max_percentage_of_sparse_data = kwargs['max_percentage_of_sparse_data']
        except KeyError:
            self.__max_percentage_of_sparse_data = self.__default_max_percentage_of_sparse_data
            if verbose:
                print('HDF5Dataset: max_percentage_of_sparse_data not present in kwargs, using default value:', self.__default_max_percentage_of_sparse_data)

        try:
            self.__terrain_percentage_correction = kwargs['terrain_percentage_correction']
        except KeyError:
            self.__terrain_percentage_correction = self.__default_terrain_percentage_correction
            if verbose:
                print('HDF5Dataset: terrain_percentage_correction not present in kwargs, using default value:', self.__default_terrain_percentage_correction)

        try:
            self.__sample_terrain_region = kwargs['sample_terrain_region']
        except KeyError:
            self.__sample_terrain_region = self.__default_sample_terrain_region
            if verbose:
                print('HDF5Dataset: sample_terrain_region not present in kwargs, using default value:', self.__default_sample_terrain_region)

        try:
            self.__add_gaussian_noise = kwargs['add_gaussian_noise']
        except KeyError:
            self.__add_gaussian_noise = self.__default_add_gaussian_noise
            if verbose:
                print('HDF5Dataset: add_gaussian_noise not present in kwargs, using default value:', self.__default_add_gaussian_noise)

        try:
            self.__add_turbulence = kwargs['add_turbulence']
        except KeyError:
            self.__add_turbulence = self.__default_add_turbulence
            if verbose:
                print('HDF5Dataset: add_turbulence not present in kwargs, using default value:', self.__default_add_turbulence)

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
        default_channels = ['terrain', 'ux', 'uy', 'uz', 'turb', 'p', 'epsilon', 'nut']
        for channel in input_channels:
            if channel not in default_channels:
                raise ValueError('HDF5Dataset: Incorrect input_channel detected: \'{}\', '
                                 'correct channels are {}'.format(channel, default_channels))

        for channel in label_channels:
            if channel not in default_channels:
                raise ValueError('HDF5Dataset: Incorrect label_channel detected: \'{}\', '
                                 'correct channels are {}'.format(channel, default_channels))

        self.__channels_to_load = []
        self.__input_indices = []
        self.__label_indices = []

        # make sure that the channels_to_load list is correctly ordered, and save the input and label variable indices
        index = 0
        for channel in default_channels:
            if channel in input_channels or channel in label_channels or channel in weighting_channels:
                self.__channels_to_load += [channel]
                if channel in input_channels:
                    self.__input_indices += [index]
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

        # parse the augmentation_kwargs depending on the augmentation_mode
        if self.__augmentation:
            # mode 1 has no options
            if self.__augmentation_mode == 0:
                try:
                    self.__augmentation_kwargs = kwargs['augmentation_kwargs']
                except KeyError:
                    self.__augmentation_kwargs = self.__default_augmentation_kwargs
                    if verbose:
                        print('HDF5Dataset: augmentation_kwargs not present in kwargs, using default value:',
                              self.__default_augmentation_kwargs)
                try:
                    self.__subsample = self.__augmentation_kwargs['subsampling']
                except:
                    self.__subsample = True
                    if verbose:
                        print('HDF5Dataset: subsampling not present in augmentation_kwargs, using default value:', True)

                try:
                    self.__rotating = self.__augmentation_kwargs['rotating']
                except:
                    self.__rotating = True
                    if verbose:
                        print('HDF5Dataset: rotating not present in augmentation_kwargs, using default value:', True)

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
        is_train_set = 'train' in self.__filename
        if self.__add_turbulence and is_train_set:
            turbulent_velocity_fields = []
            # generate 100 fields
            for i in range(100):
                uvw, _ = generate_turbulence.generate_turbulence_spectral()
                turbulent_velocity_field = uvw[:, :-1, :-1, :-1]
                turbulent_velocity_fields += [torch.from_numpy(turbulent_velocity_field.astype(np.float32)).unsqueeze(0)]
            self.__turbulent_velocity_fields = torch.cat(turbulent_velocity_fields, 0)

        # initialize random number generator used for the subsampling
        self.__rand = random.SystemRandom()

        # interpolator for the three input velocities
        self.__interpolator = DataInterpolation(self.__device, 3, self.__nx, self.__ny, self.__nz)

        # avoids printing a warning multiple times
        self.__augmentation_warning_printed = False

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

        noise_scale = 1
        # 3D data transforms
        if (len(data_shape) == 3):
            # apply autoscale if requested
            if self.__autoscale:
                # the velocities are indices 1,2,3
                scale = self.__get_scale(data[1:4, :, :, :])
                noise_scale = scale

                # applying the autoscale to the velocities
                data[1:4, :, :, :] /= scale

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
                    # use the numpy implementation as it is more accurate and slightly faster
                    data = self.__augmentation_mode2_numpy(data, self.__nx) # u_x: index 1, u_y: index 2
                    #data = self.__augmentation_mode2_torch(data, self.__nx) # u_x: index 1, u_y: index 2

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

            sparse_input = data.clone()
            # add noise to wind velocities in train set if requested
            is_train_set = 'train' in self.__filename
            if self.__add_gaussian_noise and is_train_set:
                eps = 1
                noise = eps * torch.rand(sparse_input[1:4, :].shape)
                if self.__autoscale:  # apply scale to noise if necessary
                    noise /= noise_scale
                sparse_input[1:4, :] += noise

            # add turbulence to wind velocities in train set if requested
            is_train_set = 'train' in self.__filename
            if self.__add_turbulence and is_train_set:
                # randomly select one of the turbulent velocity fields
                rand_num = np.random.randint(0, 99)
                turbulence = self.__turbulent_velocity_fields[rand_num]
                _, turb_nx, turb_ny, turb_nz = turbulence.shape
                # subsample turbulent velocity field with the same shape as data
                _, nz, ny, nx = sparse_input[1:4, :].shape
                start_x = self.__rand.randint(0, turb_nx - nx)
                start_y = self.__rand.randint(0, turb_ny - ny)
                start_z = int(self.__rand.triangular(0, (turb_nz - nz), 0))  # triangle distribution
                turbulence = \
                    turbulence[:, start_z:start_z+nz,  start_y:start_y+ny,  start_x:start_x+nx]
                if self.__autoscale:  # apply scale to turbulence if necessary
                    turbulence /= noise_scale
                sparse_input[1:4, :] += turbulence

            # create and add sparse mask to input if requested
            if self.__create_sparse_mask:
                terrain = sparse_input[0, :]
                nz, ny, nx = terrain.shape
                boolean_terrain = terrain > 0
                # percentage of sparse data
                p = random.random() * self.__max_percentage_of_sparse_data
                if p < 1e-6:
                    p = 1e-6
                # terrain correction
                if self.__terrain_percentage_correction:
                    terrain_percentage = 1 - boolean_terrain.sum().item() / boolean_terrain.numel()
                    corrected_percentage = p / (1 - terrain_percentage)
                    percentage = corrected_percentage
                else:
                    percentage = p

                if self.__sample_terrain_region:
                    # sample terrain region
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
                    # percentage correction for the sampled terrain region
                    percentage = percentage / p_sample
                else:
                    uniform_dist = torch.FloatTensor(nz, ny, nx).uniform_()
                    terrain_uniform_mask = boolean_terrain.float() * uniform_dist
                # sparsity mask
                mask = terrain_uniform_mask > (1 - percentage)
                # add mask to data
                sparse_input = torch.cat([sparse_input, mask.float().unsqueeze(0)])

            # generate the input channels
            if self.__create_sparse_mask:
                # take modified labels with terrain and sparse mask as input data
                input = sparse_input
            else:
                input_data = torch.index_select(data, 0, self.__input_indices)
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
            print('HDF5Dataset Error: 2D data handling is not implemented yet')
            sys.exit()
        else:
            print('HDF5Dataset Error: Data dimension of ', len(data_shape), ' is not supported')
            sys.exit()

    def get_name(self, index):
        return self.__memberslist[index]

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

    def __get_scale(self, x):
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