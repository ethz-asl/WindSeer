from __future__ import print_function

from nn_wind_prediction.utils.interpolation import DataInterpolation

from interpolation.splines import UCGrid, eval_linear
import numpy as np
import random
import sys
import torch
from torch.utils.data.dataset import Dataset
import h5py

class HDF5Dataset(Dataset):
    '''
    Class to handle the dataset with containing velocities and the turbulent kinetic energy (k).

    The dataset is a single tar file containing all the samples stored as 4D-pytorch tensors.
    The four dimensions are: [channels, z, y, x].
    The channel ordering is: [terrain, u_x, u_y, u_z, turb, p, epsilon, nut]

    The raw data is split up to an input tensor and output tensor. The input tensor contains the velocities and the terrain
    information in the following order: [terrain, ux_in, *uy_in, uz_in], where uy_in is only present in the 3D case.
    The number of output channels is configurable but always contains at least the velocities:
    [ux_out, *uy_out, uz_out, *k, *dx, *dy, *dz].
    uy_out is only present in the 3D case and it can be chosen if also the turbulent kinetic energy (k) and the grid sizes
    (dx, dy, dz) are contained in the output tensor.

    TODO:
    - Check if it is feasible also to store the filedescriptors or how much faster it will make the dataloading (using Lock when accessing the file descriptors
    - Reimplement the 2D data handling
    '''

    __default_device = 'cpu'
    __default_nx = 64
    __default_ny = 64
    __default_nz = 64
    __default_input_mode = 0
    __default_augmentation = False
    __default_augmentation_mode = 0
    __default_augmentation_kwargs = None
    __default_stride_hor = 1
    __default_stride_vert = 1
    __default_turbulence_label = True
    __default_scaling_dict = {'terrain': 1.0, 'u_x': 1.0, 'u_y': 1.0, 'u_z': 1.0, 'turb': 1.0, 'p': 1.0, 'epsilon': 1.0, 'nut': 1.0}
    __default_return_grid_size = False
    __default_return_name = False
    __default_autoscale = False

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
                If true the data is augmented according to the mode and augmentation_kwargs
            augmentation_mode:
                Specifies the data augmentation mode
                    0: Rotating and subsampling the data without interpolation (rotation in 90deg steps, shift in integer steps)
                    1: Rotating and subsampling the data with interpolation (continuous rotation, continous shift)
            stride_hor:
                Horizontal stride, used to reduce the size of the output tensors, default 1
            stride_vert:
                Vertical stride, used to reduce the size of the output tensors, default 1
            turbulence_label:
                Specifies if the turbulent kinetic energy is contained in the output, default True
            scaling_uhor:
                Scaling factor for the horizontal velocity components, default 1.0
            scaling_uz:
                Scaling factor for the vertical velocity component, default 1.0
            scaling_turb:
                Scaling factor for the turbulent kinetic energy, default 1.0
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
        self.__filename = filename
        
        if len(input_channels) == 0:
            raise ValueError('HDF5Dataset: List of input channels cannot be empty')

        if len(label_channels) == 0:
            raise ValueError('HDF5Dataset: List of labels channels cannot be empty')

        self.__channels_to_load = []
        self.__input_indices = []
        self.__label_indices = []

        # this block makes sure that the input_channels and label_channels lists are correctly ordered
        default_channels = ['terrain', 'u_x', 'u_y', 'u_z', 'turb', 'p', 'epsilon', 'nut']
        for index, channel in enumerate(default_channels):
            if channel in input_channels or channel in label_channels:
                self.__channels_to_load += [channel]
                if channel in input_channels:
                    self.__input_indices += [index]
                if channel in label_channels:
                    self.__label_indices += [index]
        self.__input_indices = torch.LongTensor(self.__input_indices)
        self.__label_indices = torch.LongTensor(self.__label_indices)

        try:
            h5_file = h5py.File(filename, 'r')
        except IOError as e:
            print('I/O error({0}): {1}: {2}'.format(e.errno, e.strerror, filename))
            sys.exit()

        try:
            verbose = kwargs['verbose']
        except KeyError:
            verbose = False

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

        # parse the augmentation_kwargs depending on the augmentation_mode
        if self.__augmentation:
            # mode 1 has no options
            if self.__augmentation_mode == 0:
                try:
                    self.__subsample = kwargs['augmentation_kwargs']['subsampling']
                except KeyError:
                    self.__subsample = True
                    if verbose:
                        print('HDF5Dataset: subsampling not present in augmentation_kwargs, using default value:', True)

                try:
                    self.__rotating = kwargs['augmentation_kwargs']['rotating']
                except KeyError:
                    self.__rotating = True
                    if verbose:
                        print('HDF5Dataset: rotating not present in augmentation_kwargs, using default value:', True)

        # create scaling dict for each channel
        self.__scaling_dict = dict()
        for channel in input_channels:
            try:
                self.__scaling_dict[channel] = kwargs['scaling_' + channel]
            except KeyError:
                self.__scaling_dict[channel] = self.__default_scaling_dict[channel]
                if verbose:
                    print('HDF5Dataset: scaling_', channel, ' not present in kwargs, using default value:',
                          self.__default_scaling_dict[channel])

        # extract info from the h5 file
        self.__num_files = len(h5_file.keys())
        self.__memberslist = h5_file.keys()

        # initialize random number generator used for the subsampling
        self.__rand = random.SystemRandom()

        # interpolator for the three input velocities
        self.__interpolator = DataInterpolation(self.__device, 3, self.__nx, self.__ny, self.__nz)

        # avoids printing a warning multiple times
        self.__augmentation_warning_printed = False

        print('HDF5Dataset: ' + filename + ' contains {} samples'.format(self.__num_files))

    def __getitem__(self, index):
        h5_file = h5py.File(self.__filename, 'r')
        sample = h5_file[list(self.__memberslist)[index]]

        # load the data
        data = torch.Tensor()
        for channel in self.__channels_to_load:
            # extract channel data and apply scaling
            channel_data = torch.from_numpy(sample[channel][...]) / self.__scaling_dict[channel]
            data = torch.cat((data, channel_data) , 0)

        # send full data to device
        data = data.to(self.__device)

        ds = torch.from_numpy(sample['ds'][...])

        data_shape = data[0,:].shape

        # 3D data transforms
        if (len(data_shape) == 3):
            # apply autoscale if requested, needs all velocity channels to be loaded
            if self.__autoscale and all(elem in self.__channels_to_load for elem in ['u_x', 'u_y', 'u_z']):
                scale = self.__get_scale(data[1:4, :, :, :])
                data[1:4, :, :, :] /= scale
            elif self.__autoscale:
                print('HDF5Dataset: autoscale not applied, not all velocity channels were requested')

            # downscale if requested
            data = data[:,::self.__stride_vert,::self.__stride_hor, ::self.__stride_hor]

            # augment if requested according to the augmentation mode. Only works for now if all velocities and terrain are loaded
            if self.__augmentation and all(elem in self.__channels_to_load for elem in ['terrain', 'u_x', 'u_y', 'u_z']):
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
                            data = data[[0,2,1,3,4]]
                            data[1,:,:,:] *= -1.0

                            ds = (ds[1], ds[0], ds[2])

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
                    data = data[:,:self.__nz, :self.__ny, :self.__nx]
                    if self.__augmentation:
                        print('HDF5Dataset: augmentation not applied, not all of the needed channels '
                              '(terrain, u_x, u_y, u_z) were requested.')

            # generate the input channels
            input_data = torch.index_select(data, 0, self.__input_indices)
            if (self.__input_mode == 0):
                # copy the inflow condition across the full domain
                input = torch.cat((input_data[0,:,:,:].unsqueeze(0), input_data[1:,:,:,0].unsqueeze(-1).expand(-1,-1,-1,self.__nx)))

            elif (self.__input_mode == 1):
                # This interpolation is slower (at least on a cpu)
                # input = torch.cat([data[0,:,:,:].unsqueeze(0),
                # self.__interpolator.edge_interpolation_batch(data[1:4,:,:,:].unsqueeze(0)).squeeze()])

                # interpolating the vertical edges
                input = torch.cat((input_data[0,:,:,:].unsqueeze(0), self.__interpolator.edge_interpolation(input_data[1:,:,:,:])))

            else:
                print('HDF5Dataset Error: Input mode ', self.__input_mode, ' is not supported')
                sys.exit()

            output = torch.index_select(data, 0, self.__label_indices)

            out = [input, output]

            if self.__autoscale:
                out.append(scale)

            if self.__return_grid_size:
                out.append(ds)

            if self.__return_name:
                out.append(sample)

            return out

        elif (len(data_shape) == 2):
            print('HDF5Dataset Error: 2D data handling is not implemented yet')
            sys.exit()
        else:
            print('HDF5Dataset Error: Data dimension of ', len(data_shape), ' is not supported')
            sys.exit()

    def get_name(self, index):
        return self.__memberslist[index].name

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

        corners = torch.index_select(x, 2, torch.tensor([0,shape[2]-1]))
        corners = torch.index_select(corners, 3, torch.tensor([0,shape[3]-1]))

        return corners.norm(dim=0).mean(dim=0).max()
