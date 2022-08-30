from __future__ import print_function

from windseer.data.interpolation import DataInterpolation, interpolate_sparse_data
import windseer.data.generate_turbulence as generate_turbulence
import windseer.utils as windseer_utils

from enum import Enum
import numpy as np
import random
import threading
import torch
from torch.utils.data.dataset import Dataset
import h5py
import sys

numpy_interpolation = False
if sys.version_info[0] > 2:
    from interpolation.splines import UCGrid, eval_linear
    numpy_interpolation = True


class InputMode(Enum):
    INFLOW_CONDITION = 0
    CORNER_WINDS = 1
    LABEL_WINDS = 2
    SPARSE_FULL_DOMAIN = 3
    SPARSE_SUBDOMAIN = 4
    TRAJECTORIES = 5


class WeighingFunction(Enum):
    NONE = 0
    SQUARED_PRESSURE_FLUCTUATIONS = 1
    L2_PRESSURE_GRADIENTS = 2
    L2_VELOCITY_GRADIENTS = 3


class HDF5Dataset(Dataset):
    '''
    Dataset to handle the HDF5 file of cfd generated flows.

    The dataset is a single hdf5 file containing groups for all of the samples which contain 4D arrays for each of the
    channels.
    The four dimensions are: [channels, z, y, x].

    The raw data is split up to an input tensor and label tensor. The input and label tensor channels are specified and
    contain information in the following order: [terrain, u_x, u_y, u_z, turb, p, epsilon, nut].
    The number of input and label channels is configurable.
    The grid sizes (dx, dy, dz) are contained in the output tensor.
    '''

    _default_scaling_dict = {
        'terrain': 1.0,
        'ux': 1.0,
        'uy': 1.0,
        'uz': 1.0,
        'turb': 1.0,
        'p': 1.0,
        'epsilon': 1.0,
        'nut': 1.0
        }

    _lock = threading.Lock()
    _min_num_cells = np.inf
    _max_num_cells = 0
    _num_samples = 0
    _average_num_cells = 0

    def __init__(self, filename, input_channels, label_channels, **kwargs):
        '''
        Parameters
        ----------
        filename : string
            The name of the hdf5 file
        input_channels : list of str
            List of the channel names of the returned input tensor
        label_channels (required): list
            List of the channel names of the returned label tensor
        device : str, default: CPU
            Device name where tensor operations are executed (CPU or GPU)
        nx : int, default: 64
            Grid size in x-dimension
        ny : int, default: 64
            Grid size in y-dimension
        nz : int, default: 64
            Grid size in z-dimension
        input_mode : int, default: 0
            Indicates how the input is constructed. The following modes are currently implemented:
                0: The inflow condition is copied over the full domain
                1: The vertical edges are interpolated over the full domain
                2: Return the CFD ground truth solution (channel wise)
                3: Randomly sample a sparse mask over the full domain and return the values of those sampled cells
                4: Randomly sample a sparse mask over the sub domain and return the values of those sampled cells
                5: Generate trajectory and return the values along the trajectory
        augmentation : bool, default: False
            Choose if the data should be augmented according to the augmentation_mode and augmentation_kwargs.
        augmentation_mode : int, default: 0
            Specifies the data augmentation mode
                0: Rotating and subsampling the data without interpolation (rotation in 90deg steps, shift in integer steps)
                1: Rotating and subsampling the data with interpolation (continuous rotation, continuous shift)
        stride_hor : int, default: 1
            Horizontal stride, used to reduce the size of the output tensors
        stride_vert : int, default: 1
            Vertical stride, used to reduce the size of the output tensors
        scaling_ux : float, default: 1.0
            Scaling factor for the x velocity components
        scaling_uy : float, default: 1.0
            Scaling factor for the y velocity components
        scaling_uz : float, default: 1.0
            Scaling factor for the vertical velocity component
        scaling_turb : float, default: 1.0
            Scaling factor for the turbulent kinetic energy
        scaling_p : float, default: 1.0
            Scaling factor for the pressure
        scaling_epsilon : float, default: 1.0
            Scaling factor for dissipation
        scaling_nut : float, default: 1.0
            Scaling factor for viscosity
        scaling_terrain : float, default: 1.0
            Scaling factor for terrain channel
        return_grid_size : bool, default: False
            If true the grid size is returned in addition to the data tensors
        return_name : bool, default: False
            If true the sample name is returned in addition to the data tensors
        autoscale : bool, default: False
            Scale the input based on an automatically computed scale of the corner winds
        loss_weighting_fn : int, default: 0
            Return a matrix to weight the individual cells differently
                0: no weighting function, weights are ones
                1: squared pressure fluctuations
                2: l2 norm of the pressure gradient
                3: l2 norm of the velocity gradient
        loss_weighting_clamp : bool, default: False
            Indicating if the weights should be clamped to a certain maximum value
        additive_gaussian_noise : bool, default: False
            Indicating if the gaussian noise is additive or a relative factor on the input values
        max_gaussian_noise_std : float, default: 0.0
            Maximum standard deviation of the gaussian noise added to the input. If set to 0 no noise is added.
        n_turb_fields : int, default: 0
            Number of turbulence fields generated in the initialization to add turbulent disturbances to the
            wind velocities. If either n_turb_fields or max_normalized_turb_scale are 0 no disturbances are
            added.
        max_normalized_turb_scale : float, default: 0.0
            Maximum magnitude of the turbulence added to the normalized wind.
            If either n_turb_fields or max_normalized_turb_scale are 0 no disturbances are added.
        max_normalized_bias_scale : float, default: 0.0
            Maximum magnitude of the bias added to the normalized wind. If set to 0 no bias is added.
        only_z_velocity_bias : bool, default: False
            Indicates if the bias is added only to the z-velocities. If false the bias is added to all wind axes.
        max_fraction_of_sparse_data : float, default: 0.0
            In case of a sparse input this indicates the maximum number of cells that should be sampled.
        min_fraction_of_sparse_data : float, default: 0.0
            In case of a sparse input this indicates the minimum number of cells that should be sampled.
        use_system_random : bool, default: False
            If true the true system random generator is used, else the standard pseudo number generated
            is used where setting the seed is feasible
        input_smoothing : bool, default: False
            If false unknown cells for the sparse input modes are filled with zero else according to the settings
            input_smoothing_interpolation and input_smoothing_interpolation_linear.
        input_smoothing_interpolation : bool, default: False
            If true the unknown cell values for the sparse input modes are filled with an interpolated value
            according to the input_smoothing_interpolation_linear setting, else the average measurements value
            per direction is used. Only used if input_smoothing is true.
        input_smoothing_interpolation_linear : bool, default: False
            If true a linear interpolation is used, else the nearest value is taken to fill the unknown cells.
            Only used if input_smoothing and are true.
        trajectory_min_length : int, default: 30
            Minimum trajectory length in cells for the trajectory generation.
            Only used for input_mode 5.
        trajectory_max_length : int, default: 300
            Maximum trajectory length in cells for the trajectory generation.
            Only used for input_mode 5.
        trajectory_min_segment_length : float, default: 5.0
            Minimum trajectory segment length in cells for the trajectory generation.
            Only used for input_mode 5.
        trajectory_max_segment_length : float, default: 20.0
            Maximum trajectory segment length in cells for the trajectory generation.
            Only used for input_mode 5.
        trajectory_step_size : float, default: 1.0
            Step size in cells along the trajectory to determine the cells crossed along the path.
            Only used for input_mode 5.
        trajectory_max_iter : int, default: 50
            Maximum number of segments added during the trajectory generation.
            Only used for input_mode 5.
        trajectory_start_weighting_mode : int, default: 0
            Weighting mode the start altitudes for the trajectory generation.
            Only used for input_mode 5.
                0: Equal weighing for all heights
                1: Focusing on lower altitudes with a linear weighing function
                2: Focusing on lower altitudes with a squared weighing function
        trajectory_length_short_focus : bool, default: False
            Boolean indicating if shorter trajectories should be favored with a
            triangle distribution instead of a uniform distribution to determine the length.
            Only used for input_mode 5.
        '''

        parser = windseer_utils.KwargsParser(kwargs, 'HDF5Dataset')
        verbose = parser.get_safe('verbose', False, bool, False)
        self._device = parser.get_safe('device', 'cpu', str, verbose)
        self._nx = parser.get_safe('nx', 64, int, verbose)
        self._ny = parser.get_safe('ny', 64, int, verbose)
        self._nz = parser.get_safe('nz', 64, int, verbose)
        self._input_mode = InputMode(parser.get_safe('input_mode', 0, int, verbose))
        self._augmentation = parser.get_safe('augmentation', False, bool, verbose)
        self._stride_hor = parser.get_safe('stride_hor', 1, int, verbose)
        self._stride_vert = parser.get_safe('stride_vert', 1, int, verbose)
        self._return_grid_size = parser.get_safe(
            'return_grid_size', False, bool, verbose
            )
        self._return_name = parser.get_safe('return_name', False, bool, verbose)
        self._autoscale = parser.get_safe('autoscale', False, bool, verbose)
        self._loss_weighting_fn = WeighingFunction(
            parser.get_safe('loss_weighting_fn', 0, int, verbose)
            )
        self._loss_weighting_clamp = parser.get_safe(
            'loss_weighting_clamp', True, bool, verbose
            )
        self._additive_gaussian_noise = parser.get_safe(
            'additive_gaussian_noise', True, bool, verbose
            )
        self._max_gaussian_noise_std = parser.get_safe(
            'max_gaussian_noise_std', 0.0, float, verbose
            )
        self._n_turb_fields = parser.get_safe('n_turb_fields', 0, int, verbose)
        self._max_normalized_turb_scale = parser.get_safe(
            'max_normalized_turb_scale', 0.0, float, verbose
            )
        self._max_normalized_bias_scale = parser.get_safe(
            'max_normalized_bias_scale', 0.0, float, verbose
            )
        self._only_z_velocity_bias = parser.get_safe(
            'only_z_velocity_bias', False, bool, verbose
            )
        self._use_system_random = parser.get_safe(
            'use_system_random', False, bool, verbose
            )

        if self._input_mode == InputMode.SPARSE_FULL_DOMAIN or self._input_mode == InputMode.SPARSE_SUBDOMAIN:
            self._max_fraction_of_sparse_data = parser.get_safe(
                'max_fraction_of_sparse_data', 1.0, float, verbose
                )
            self._min_fraction_of_sparse_data = parser.get_safe(
                'min_fraction_of_sparse_data', 0.0, float, verbose
                )
            self._input_smoothing = parser.get_safe(
                'input_smoothing', False, bool, verbose
                )
            self._input_smoothing_interpolation = parser.get_safe(
                'input_smoothing_interpolation', False, bool, verbose
                )
            self._input_smoothing_interpolation_linear = parser.get_safe(
                'input_smoothing_interpolation_linear', False, bool, verbose
                )

        if self._input_mode == InputMode.TRAJECTORIES:
            self._trajectory_min_length = parser.get_safe(
                'trajectory_min_length', 30, int, verbose
                )
            self._trajectory_max_length = parser.get_safe(
                'trajectory_max_length', 300, int, verbose
                )
            self._trajectory_min_segment_length = parser.get_safe(
                'trajectory_min_segment_length', 5.0, float, verbose
                )
            self._trajectory_max_segment_length = parser.get_safe(
                'trajectory_max_segment_length', 20.0, float, verbose
                )
            self._trajectory_step_size = parser.get_safe(
                'trajectory_step_size', 1.0, float, verbose
                )
            self._trajectory_max_iter = parser.get_safe(
                'trajectory_max_iter', 50, int, verbose
                )
            self._trajectory_start_weighting_mode = parser.get_safe(
                'trajectory_start_weighting_mode', 0, int, verbose
                )
            self._trajectory_length_short_focus = parser.get_safe(
                'trajectory_length_short_focus', False, bool, verbose
                )
            self._input_smoothing = parser.get_safe(
                'input_smoothing', False, bool, verbose
                )
            self._input_smoothing_interpolation = parser.get_safe(
                'input_smoothing_interpolation', False, bool, verbose
                )
            self._input_smoothing_interpolation_linear = parser.get_safe(
                'input_smoothing_interpolation_linear', False, bool, verbose
                )

        if self._augmentation:
            self._augmentation_mode = parser.get_safe(
                'augmentation_mode', 0, int, verbose
                )

            if self._augmentation_mode == 0:
                default_dict = {'subsampling': True, 'rotating': True, }
                augmentation_kwargs = parser.get_safe(
                    'augmentation_kwargs', default_dict, dict, verbose
                    )

                self._subsample = augmentation_kwargs['subsampling']
                self._rotating = augmentation_kwargs['rotating']

        # --------------- initializing class params -------------------------
        if len(input_channels) == 0 or len(label_channels) == 0:
            raise ValueError(
                'HDF5Dataset: List of input or label channels cannot be empty'
                )

        if self._loss_weighting_fn == WeighingFunction.SQUARED_PRESSURE_FLUCTUATIONS:
            weighting_channels = ['p']
        elif self._loss_weighting_fn == WeighingFunction.L2_PRESSURE_GRADIENTS:
            weighting_channels = ['terrain', 'p']
        elif self._loss_weighting_fn == WeighingFunction.L2_VELOCITY_GRADIENTS:
            weighting_channels = ['terrain', 'ux', 'uy', 'uz']
        else:
            weighting_channels = []

        # make sure that all requested channels are possible
        self.default_channels = [
            'terrain', 'ux', 'uy', 'uz', 'turb', 'p', 'epsilon', 'nut'
            ]
        for channel in input_channels:
            if channel not in self.default_channels:
                raise ValueError(
                    'HDF5Dataset: Incorrect input_channel detected: \'{}\', '
                    'correct channels are {}'.format(channel, self.default_channels)
                    )

        for channel in label_channels:
            if channel not in self.default_channels:
                raise ValueError(
                    'HDF5Dataset: Incorrect label_channel detected: \'{}\', '
                    'correct channels are {}'.format(channel, self.default_channels)
                    )

        self._channels_to_load = []
        self._input_channels = []
        self._input_indices = []
        self._label_indices = []
        self._label_channels = []
        self._input_velocities_indices = []
        self._data_velocity_indices = []

        # make sure that the channels_to_load list is correctly ordered, and save the input and label variable indices
        index = 0
        for channel in self.default_channels:
            if channel in input_channels or channel in label_channels or channel in weighting_channels:
                self._channels_to_load += [channel]
                if channel in input_channels:
                    self._input_channels += [channel]
                    self._input_indices += [index]
                    if channel in ['ux', 'uy', 'uz']:
                        self._data_velocity_indices += [index]
                        self._input_velocities_indices += [len(self._input_indices) - 1]
                if channel in label_channels:
                    self._label_indices += [index]
                    self._label_channels += [channel]
                index += 1

        self._input_indices = torch.LongTensor(self._input_indices)
        self._label_indices = torch.LongTensor(self._label_indices)

        self._filename = filename

        try:
            h5_file = h5py.File(filename, 'r')
        except IOError as e:
            print('I/O error({0}): {1}: {2}'.format(e.errno, e.strerror, filename))
            sys.exit()

        # extract info from the h5 file
        self._num_files = len(h5_file.keys())
        self._memberslist = list(h5_file.keys())
        h5_file.close()

        # check that all the required channels are present for autoscale for augmentation
        # this is due to the get_scale method which needs the the velocities to compute the scale for autoscaling
        # augmentation mode 1 and 0 use indexing, the first 4 channels are required
        if not all(
            elem in self._channels_to_load for elem in ['terrain', 'ux', 'uy', 'uz']
            ):
            print(
                'HDF5Dataset: augmentation and autoscale will not be applied, not all of the required channels (terrain,ux, uy, uz) were requested'
                )
            self._autoscale = False
            self._augmentation = False

        # create scaling dict for each channel
        self._scaling_dict = dict()
        for channel in self._channels_to_load:
            try:
                self._scaling_dict[channel] = kwargs['scaling_' + channel]
            except KeyError:
                self._scaling_dict[channel] = self._default_scaling_dict[channel]
                if verbose:
                    print(
                        'HDF5Dataset:', 'scaling_' + channel,
                        'not present in kwargs, using default value:',
                        self._default_scaling_dict[channel]
                        )

        # create turbulent velocity fields for train set
        if self._n_turb_fields > 0 and self._max_normalized_turb_scale > 0.0:
            turbulent_velocity_fields = []
            for i in range(self._n_turb_fields):
                # dx, dy, dz are hardcoded for the moment
                dx = dy = 16.6444
                dz = 11.5789
                turbulent_velocity_field, _ = generate_turbulence.generate_turbulence_spectral(
                    int(self._nx * 1.5), int(self._ny * 1.5), int(self._ny * 1.5), dx,
                    dy, dz
                    )
                turbulent_velocity_field /= np.max(
                    np.abs(turbulent_velocity_field)
                    )  # scale the field to have a maximum value of 1
                turbulent_velocity_fields += [
                    torch.from_numpy(turbulent_velocity_field.astype(np.float32)
                                     ).unsqueeze(0)
                    ]
            self._turbulent_velocity_fields = torch.cat(turbulent_velocity_fields, 0)

        # add the mask to the input channels for the respective input modes
        if (
            self._input_mode == InputMode.SPARSE_FULL_DOMAIN or
            self._input_mode == InputMode.SPARSE_SUBDOMAIN or
            self._input_mode == InputMode.TRAJECTORIES
            ):
            self._input_channels += ['mask']

        # initialize random number generator used for the subsampling
        if self._use_system_random:
            self._rand = random.SystemRandom()
        else:
            self._rand = random

        # interpolator for the three input velocities
        self._interpolator = DataInterpolation(
            self._device, len(self._input_velocities_indices), self._nx, self._ny,
            self._nz
            )

        # avoids printing a warning multiple times
        self._augmentation_warning_printed = False

        if verbose:
            print(
                'HDF5Dataset: ' + filename +
                ' contains {} samples'.format(self._num_files)
                )

    def __getitem__(self, index):
        '''
        Get a sample from the dataset and return a list based on the configuration.
        
        Parameters
        ----------
        index : int
            Sample index

        Returns
        -------
        input : torch.Tensor
            Tensor containing the input data. The first channel is always the terrain.
        label : torch.Tensor
            Tensor with the label data.
        loss_weighting_matrix : torch.Tensor
            Tensor containing the weighing matrix constructed according to loss_weighting_fn.
        scale : torch.Tensor
            The scale of the sample. Only returned if autoscale is true.
        grid_size : list of int
            The size of a single cell in the grid. Only returned if return_grid_size is true.
        name : list of str
            The name of the sample. Only returned if return_name is true.
        '''
        h5_file = h5py.File(self._filename, 'r', swmr=True)
        sample = h5_file[self._memberslist[index]]

        # load the data
        data_from_channels = []
        for i, channel in enumerate(self._channels_to_load):
            # extract channel data and apply scaling
            data_from_channels += [
                torch.from_numpy(sample[channel][...]).float().unsqueeze(0) /
                self._scaling_dict[channel]
                ]

        data = torch.cat(data_from_channels, 0)

        # send full data to device
        data = data.to(self._device)

        ds = torch.from_numpy(sample['ds'][...])

        data_shape = data[0, :].shape

        # 3D data transforms
        if (len(data_shape) == 3):
            # apply autoscale if requested
            if self._autoscale:
                # determine the scales
                noise_scale = 1.0
                scale = self.get_scale(data[self._data_velocity_indices, :, :, :])

                # applying the autoscale to the velocities
                data[self._data_velocity_indices, :, :, :] /= scale

                # turb handling
                if 'turb' in self._channels_to_load:
                    data[self._channels_to_load.index('turb'), :, :, :] /= scale * scale

                # p handling
                if 'p' in self._channels_to_load:
                    data[self._channels_to_load.index('p'), :, :, :] /= scale * scale

                # epsilon handling
                if 'epsilon' in self._channels_to_load:
                    data[self._channels_to_load
                         .index('epsilon'), :, :, :] /= scale * scale * scale

                # nut handling
                if 'nut' in self._channels_to_load:
                    data[self._channels_to_load.index('nut'), :, :, :] /= scale

            else:
                # determine the scale of the sample to properly scale the noise
                noise_scale = self.get_scale(data[self._data_velocity_indices, :, :, :])

            # downscale if requested
            data = data[:, ::self._stride_vert, ::self._stride_hor, ::self._stride_hor]

            # augment if requested according to the augmentation mode
            if self._augmentation:
                if self._augmentation_mode == 0:
                    # subsampling
                    if self._subsample:
                        start_x = self._rand.randint(0, data_shape[2] - self._nx)
                        start_y = self._rand.randint(0, data_shape[1] - self._ny)
                        start_z = int(
                            self._rand.triangular(0, (data_shape[0] - self._nz), 0)
                            )  # triangle distribution

                        data = data[:, start_z:start_z + self._nz,
                                    start_y:start_y + self._ny,
                                    start_x:start_x + self._nx]
                    else:
                        # select the first indices
                        data = data[:, :self._nz, :self._ny, :self._nx]

                    # rotating and flipping
                    if self._rotating:
                        # flip in x-direction
                        if (self._rand.randint(0, 1)):
                            data = data.flip(3)
                            data[1, :, :, :] *= -1.0

                        # flip in y-direction
                        if (self._rand.randint(0, 1)):
                            data = data.flip(2)
                            data[2, :, :, :] *= -1.0

                        # rotate 90 degrees
                        if (self._rand.randint(0, 1)):
                            data = data.transpose(2, 3).flip(3)
                            data = torch.cat((data[[0, 2, 1]], data[3:]), 0)
                            data[1, :, :, :] *= -1.0

                            ds = torch.tensor([ds[1], ds[0], ds[2]])

                elif self._augmentation_mode == 1:
                    if numpy_interpolation:
                        # use the numpy implementation as it is more accurate and slightly faster
                        data = self._augmentation_mode2_numpy(
                            data, self._nx
                            )  # u_x: index 1, u_y: index 2
                    else:
                        # use the torch version for python 2
                        data = self._augmentation_mode2_torch(
                            data, self._nx
                            )  # u_x: index 1, u_y: index 2

                    # flip in x-direction
                    if (self._rand.randint(0, 1)):
                        data = data.flip(3)
                        data[1, :, :, :] *= -1.0

                    # flip in y-direction
                    if (self._rand.randint(0, 1)):
                        data = data.flip(2)
                        data[2, :, :, :] *= -1.0

                else:
                    if not self._augmentation_warning_printed:
                        print(
                            'WARNING: Unknown augmentation mode in HDF5Dataset ',
                            self._augmentation_mode, ', not augmenting the data'
                            )

            else:
                # no data augmentation
                data = data[:, :self._nz, :self._ny, :self._nx]

            # generate the input data
            input_data = torch.index_select(data, 0, self._input_indices)
            boolean_terrain = input_data[0] > 0

            # add gaussian noise to wind velocities if requested
            if self._max_gaussian_noise_std > 0.0:
                if self._additive_gaussian_noise:
                    std = self._rand.random(
                    ) * self._max_gaussian_noise_std * noise_scale
                    input_data[self._input_velocities_indices] += torch.randn(
                        input_data[self._input_velocities_indices].shape
                        ) * std
                else:
                    std = self._rand.random() * self._max_gaussian_noise_std
                    input_data[self._input_velocities_indices] *= torch.randn(
                        input_data[self._input_velocities_indices].shape
                        ) * std + 1.0

            # add turbulence to wind velocities if requested
            if self._n_turb_fields > 0 and self._max_normalized_turb_scale > 0.0:
                # randomly select one of the turbulent velocity fields
                rand_idx = self._rand.randint(0, self._n_turb_fields - 1)
                turb_scale = self._rand.random() * self._max_normalized_turb_scale
                turbulence = self._turbulent_velocity_fields[rand_idx]
                _, turb_nx, turb_ny, turb_nz = turbulence.shape

                # subsample turbulent velocity field with the same shape as data
                _, nz, ny, nx = input_data.shape
                start_x = self._rand.randint(0, turb_nx - nx)
                start_y = self._rand.randint(0, turb_ny - ny)
                start_z = self._rand.randint(0, turb_nz - nz)
                turbulence = turbulence[:, start_z:start_z + nz, start_y:start_y + ny,
                                        start_x:start_x + nx] * turb_scale * noise_scale

                # mask by terrain
                turbulence *= boolean_terrain

                if 'ux' in self._input_channels:
                    input_data[self._input_channels.index('ux')] += turbulence[0]
                if 'uy' in self._input_channels:
                    input_data[self._input_channels.index('uy')] += turbulence[1]
                if 'uz' in self._input_channels:
                    input_data[self._input_channels.index('uz')] += turbulence[2]

            # add bias to wind velocities if requested
            if self._max_normalized_bias_scale > 0:
                if self._only_z_velocity_bias:
                    if 'uz' in self._input_channels:
                        bias_scale = self._rand.random(
                        ) * self._max_normalized_bias_scale
                        bias = (torch.rand(1) * 2.0 - 1) * bias_scale * noise_scale
                        input_data[self._input_channels.index('uz')] += bias
                    else:
                        print(
                            'Adding bias only for uz requested but uz not present, not adding any bias'
                            )
                else:
                    bias_scale = self._rand.random() * self._max_normalized_bias_scale
                    bias = (
                        torch.rand(len(self._input_velocities_indices), 1, 1, 1) * 2.0 -
                        1
                        ) * bias_scale * noise_scale
                    input_data[self._input_velocities_indices] += bias

            # assemble the input according to the mode
            if self._input_mode == InputMode.INFLOW_CONDITION:
                # copy the inflow condition across the full domain
                input = torch.cat((
                    input_data[0, :, :, :].unsqueeze(0),
                    input_data[1:, :, :, 0].unsqueeze(-1).expand(-1, -1, -1, self._nx)
                    ))

            elif self._input_mode == InputMode.CORNER_WINDS:
                # This interpolation is slower (at least on a cpu)
                # input = torch.cat([data[0,:,:,:].unsqueeze(0),
                # self._interpolator.edge_interpolation_batch(data[1:4,:,:,:].unsqueeze(0)).squeeze()])

                # interpolating the vertical edges
                input = torch.cat((
                    input_data[0, :, :, :].unsqueeze(0),
                    self._interpolator.edge_interpolation(input_data[1:, :, :, :])
                    ))

            elif (self._input_mode == InputMode.LABEL_WINDS):
                # Input the ground truth data
                input = input_data

            elif (self._input_mode == InputMode.SPARSE_FULL_DOMAIN):
                # random sparse input with an additional mask as an input
                mask = self._create_sparse_mask(boolean_terrain)
                self._update_sparse_stats(mask.sum().item())

                # compose output
                input_data[1:] *= mask

                if self._input_smoothing:
                    input_data[1:] = self._get_smooth_data(input_data[1:], mask, ds)

                input = torch.cat([input_data, mask.float().unsqueeze(0)])

            elif (self._input_mode == InputMode.SPARSE_SUBDOMAIN):
                # random sparse input from a subregion with an additional mask as an input
                mask = self._create_sparse_mask_subregion(boolean_terrain)
                self._update_sparse_stats(mask.sum().item())

                # compose output
                input_data[1:] *= mask
                if self._input_smoothing:
                    input_data[1:] = self._get_smooth_data(input_data[1:], mask, ds)

                input = torch.cat([input_data, mask.float().unsqueeze(0)])

            elif (self._input_mode == InputMode.TRAJECTORIES):
                # faking input from a flight path
                mask = self._create_sparse_mask_trajectory(boolean_terrain)
                self._update_sparse_stats(mask.sum().item())

                # compose output
                input_data[1:] *= mask

                if self._input_smoothing:
                    input_data[1:] = self._get_smooth_data(input_data[1:], mask, ds)

                input = torch.cat([input_data, mask.float().unsqueeze(0)])

            else:
                print(
                    'HDF5Dataset Error: Input mode ', self._input_mode,
                    ' is not supported'
                    )
                sys.exit()

            # generate the input channels
            label = torch.index_select(data, 0, self._label_indices)

            # generate the loss weighting matrix
            loss_weighting_matrix = self._compute_loss_weighting(
                data, ds, self._loss_weighting_fn
                )

            out = [input, label, loss_weighting_matrix]

            if self._autoscale:
                out.append(scale)

            if self._return_grid_size:
                out.append(ds)

            if self._return_name:
                out.append(self.get_name(index))

            return out

        elif (len(data_shape) == 2):
            print('HDF5Dataset Error: 2D data handling is not implemented anymore')
            sys.exit()
        else:
            print(
                'HDF5Dataset Error: Data dimension of ', len(data_shape),
                ' is not supported'
                )
            sys.exit()

    def get_name(self, index):
        '''
        Get the name of the sample at index.
        
        Parameters
        ----------
        index : int
            Sample index

        Returns
        -------
        name : str
            The name of the sample.
        '''
        return self._memberslist[index]

    def get_input_channels(self):
        '''
        Get the list of input channels.

        Returns
        -------
        channels : list of str
            List of input channels
        '''
        return self._input_channels

    def get_label_channels(self):
        '''
        Get the list of label channels.

        Returns
        -------
        channels : list of str
            List of label channels
        '''
        return self._label_channels

    def get_ds(self):
        '''
        Get cell size in meter.

        Returns
        -------
        ds : list of float
            Cell size in each dimension
        '''
        h5_file = h5py.File(self._filename, 'r', swmr=True)
        sample = h5_file[self._memberslist[0]]
        ds = torch.from_numpy(sample['ds'][...])
        h5_file.close()
        return ds

    def print_dataset_stats(self):
        '''
        Print the statistics of the sparse mask generation if available.
        '''
        if (
            self._input_mode == InputMode.SPARSE_FULL_DOMAIN or
            self._input_mode == InputMode.SPARSE_SUBDOMAIN or
            self._input_mode == InputMode.TRAJECTORIES
            ):
            print('-------------------------------------------------')
            print('HDF5 Dataset sparse mask statistics')
            print('\tminimum number of cells:', self._min_num_cells)
            print('\tmaximum number of cells:', self._max_num_cells)
            print('\taverage number of cells:', self._average_num_cells)
            print('\tnumber of mask created: ', self._num_samples)
            print('-------------------------------------------------')
        else:
            pass

    def __len__(self):
        '''
        Get the size of the dataset.

        Returns
        -------
        length : int
            Dataset length
        '''
        return self._num_files

    def _generate_interpolation_grid(self, data_shape, out_size):
        '''
        Generate a regular grid with a size of out_size in each dimension inside the original grid
        with shape data_shape using a random yaw-rotation and shift in each direction.
        
        Currently only supports output grids with the same size in each dimension.
        
        Parameters
        ----------
        data_shape : list of int
            Extent of the original grid
        out_size : int
            Size of the output grid

        Returns
        -------
        X : torch.Tensor
            x-coordinates of the generated grid
        Y : torch.Tensor
            y-coordinates of the generated grid
        Z : torch.Tensor
            z-coordinates of the generated grid
        angle : float
            Angle of the yaw rotation [rad]
        '''
        # generate the initial unrotated grid
        x_f = np.arange(out_size)
        y_f = np.arange(out_size)
        z_f = np.arange(out_size)
        Z, Y, X = np.meshgrid(z_f, y_f, x_f, indexing='ij')

        # sample the grid orientation
        ratio = data_shape[3] / out_size
        if ratio > np.sqrt(2):
            # fully rotation possible
            angle = self._rand.uniform(0, 2.0 * np.pi)
        elif (ratio <= 1.0):
            # no rotation possible
            angle = 0.0

        else:
            # some angles are infeasible
            angle_found = False
            lower_bound = 2 * np.arctan((1 - np.sqrt(2 - ratio * ratio)) / (1 + ratio))
            upper_bound = 0.5 * np.pi - lower_bound

            while (not angle_found):
                angle = self._rand.uniform(0, 0.5 * np.pi)

                if not ((angle > lower_bound) and (angle < upper_bound)):
                    angle_found = True

            angle += 0.5 * np.pi * self._rand.randint(0, 3)

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
        X = X_rot + self._rand.uniform(x_minshift, x_maxshift)
        Y = Y_rot + self._rand.uniform(y_minshift, y_maxshift)
        Z = Z + int(self._rand.triangular(z_minshift, z_maxshift, z_minshift))

        # check if the shifted/rotated grid is fine
        if ((X.min() < 0.0) or (Y.min() < 0.0) or (Z.min() < 0.0) or
            (X.max() > data_shape[3] - 1) or (Y.max() > data_shape[2] - 1) or
            (Z.max() > data_shape[1] - 1)):
            raise RuntimeError(
                "The rotated and shifted grid does not satisfy the data grid bounds"
                )

        return X, Y, Z, angle

    def _augmentation_mode2_torch(self, data, out_size, vx_channel=1, vy_channel=2):
        '''
        Augment the data by random subsampling and rotation the data.
        This approach has errors in the order of 10^-6 even when resampling on the original grid.
        The output is a cubical grid with shape (N_channels, out_size, out_size, out_size)

        Implementation using torch functions.

        Assumptions:
            - The input is 4D (channels, z, y, x)
            - The number of cells is equal in x- and y-dimension
            - The number of cells in each dimension is at least as large as the out_size

        Parameters
        ----------
        data : torch.Tensor
            Input data tensor
        out_size : int
            Size of the output grid
        vx_channel : int, default: 1
            Channel number of the x-velocity
        vy_channel : int, default: 1
            Channel number of the y-velocity

        Returns
        -------
        interpolated : torch.Tensor
            Output tensor with the data interpolated at the generated output grid
        '''
        if not len(data.shape) == 4:
            raise ValueError(
                'The input dimension of the data array needs to be 4 (channels, z, y, x)'
                )

        if data.shape[2] != data.shape[3]:
            raise ValueError(
                'The number of cells in x- and y-direction must be the same, shape: ',
                data.shape
                )

        if out_size > data.shape[3]:
            raise ValueError(
                'The number of output cells cannot be larger than the input'
                )

        # generate the interpolation grid
        X, Y, Z, angle = self._generate_interpolation_grid(data.shape, out_size)
        X, Y, Z = torch.from_numpy(X).float(
        ).unsqueeze(0).unsqueeze(-1), torch.from_numpy(Y).float().unsqueeze(
            0
            ).unsqueeze(-1), torch.from_numpy(Z).float().unsqueeze(0).unsqueeze(-1)

        # convert the coordinates to a range of -1 to 1 as required by grid_sample
        X = 2.0 * X / (data.shape[3] - 1.0) - 1.0
        Y = 2.0 * Y / (data.shape[2] - 1.0) - 1.0
        Z = 2.0 * Z / (data.shape[1] - 1.0) - 1.0

        # assemble the final interpolation grid and interpolate
        grid = torch.cat((X, Y, Z), dim=4)
        interpolated = torch.nn.functional.grid_sample(data.unsqueeze(0),
                                                       grid).squeeze()

        # rotate also the horizontal velocities
        vel_x = np.cos(angle) * interpolated[vx_channel] + np.sin(angle) * interpolated[
            vy_channel]
        vel_y = -np.sin(angle) * interpolated[
            vx_channel] + np.cos(angle) * interpolated[vy_channel]
        interpolated[vx_channel] = vel_x
        interpolated[vy_channel] = vel_y

        # fix the terrain
        terrain = interpolated[0]
        terrain[terrain <= 0.5 / self._scaling_terrain] = 0

        return interpolated

    def _augmentation_mode2_numpy(self, data, out_size, vx_channel=1, vy_channel=2):
        '''
        Augment the data by random subsampling and rotation the data.
        The output is a cubical grid with shape (N_channels, out_size, out_size, out_size)

        Implementation using numpy functions.

        Assumptions:
            - The input is 4D (channels, z, y, x)
            - The number of cells is equal in x- and y-dimension
            - The number of cells in each dimension is at least as large as the out_size

        Parameters
        ----------
        data : torch.Tensor
            Input data tensor
        out_size : int
            Size of the output grid
        vx_channel : int, default: 1
            Channel number of the x-velocity
        vy_channel : int, default: 1
            Channel number of the y-velocity

        Returns
        -------
        interpolated : torch.Tensor
            Output tensor with the data interpolated at the generated output grid
        '''
        if not len(data.shape) == 4:
            raise ValueError(
                'The input dimension of the data array needs to be 4 (channels, z, y, x)'
                )

        if data.shape[2] != data.shape[3]:
            raise ValueError(
                'The number of cells in x- and y-direction must be the same, shape: ',
                data.shape
                )

        if out_size > data.shape[3]:
            raise ValueError(
                'The number of output cells cannot be larger than the input'
                )

        # generate the existing grid and data
        values = np.moveaxis(data.numpy(), 0, -1)
        grid = UCGrid((0.0, values.shape[0] - 1.0, values.shape[0]),
                      (0.0, values.shape[1] - 1.0, values.shape[1]),
                      (0.0, values.shape[2] - 1.0, values.shape[2]))

        # generate the interpolation grid
        X, Y, Z, angle = self._generate_interpolation_grid(data.shape, out_size)

        # interpolate
        points = np.stack((Z.ravel(), Y.ravel(), X.ravel()), axis=1)
        interpolated = eval_linear(grid, values, points)
        interpolated = interpolated.reshape(
            (out_size, out_size, out_size, values.shape[3])
            )
        interpolated = np.moveaxis(interpolated, -1, 0)

        # rotate also the horizontal velocities
        vel_x = np.cos(angle) * interpolated[vx_channel] + np.sin(angle) * interpolated[
            vy_channel]
        vel_y = -np.sin(angle) * interpolated[
            vx_channel] + np.cos(angle) * interpolated[vy_channel]
        interpolated[vx_channel] = vel_x
        interpolated[vy_channel] = vel_y

        return torch.from_numpy(interpolated).float()

    def get_scale(self, x):
        '''
        Compute the scale of the sample as the average velocity norm of the corner velocities.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor

        Returns
        -------
        scale : torch.Tensor
            Scale of the sample
        '''
        shape = x.shape

        # get data from corners of sample
        corners = torch.index_select(x, 2, torch.tensor([0, shape[2] - 1]))
        corners = torch.index_select(corners, 3, torch.tensor([0, shape[3] - 1]))

        # compute the scale
        scale = corners.norm(dim=0).mean(dim=0).max()

        # account for unfeasible scales (e.g. from zero samples)
        if scale > 0:
            return scale
        else:
            return torch.tensor(1.0)

    def _create_sparse_mask(self, boolean_terrain):
        '''
        Creating a randomly sampled sparse mask making sure only non-terrain cells are sampled.
        The number of sampled cells is uniformly sampled between
        (min_fraction_of_sparse_data, max_fraction_of_sparse_data)        

        Parameters
        ----------
        boolean_terrain : torch.Tensor
            Boolean representation of the terrain (true for wind cells and false for terrain cells)

        Returns
        -------
        mask : torch.Tensor
            Sampled output mask (true for sampled location and false everywhere else)
        '''
        mask = torch.zeros_like(boolean_terrain)
        mask_shape = mask.shape

        # percentage of sparse data
        target_frac = self._rand.random() * (
            self._max_fraction_of_sparse_data - self._min_fraction_of_sparse_data
            ) + self._min_fraction_of_sparse_data
        target_num_cell = int(mask.numel() * target_frac)

        # set the initial number of cells to a negative value to correct for the expected number of cells samples inside the terrain
        mask_num_cell = int(
            target_num_cell * (1.0 - 1.0 / boolean_terrain.float().mean().item())
            )

        # iterate multiple times to make sure the right amount of cells are set in the mask, currently hardcoded to 2 iterations
        max_iter = 2
        iter = 0
        while iter < max_iter and mask_num_cell < target_num_cell:
            max_iter = max_iter + 1

            # sample cells
            indices = self._rand.sample(
                range(mask.numel()),
                min(int(target_num_cell - mask_num_cell), mask.numel())
                )
            mask = mask.view(-1)
            mask[indices] = 1.0
            mask = mask.view(mask_shape)

            # make sure no cells are inside the terrain
            mask *= boolean_terrain
            mask_num_cell = mask.sum().item()

        return mask

    def _create_sparse_mask_subregion(self, boolean_terrain):
        '''
        Creating a randomly sampled sparse mask making sure only non-terrain cells are sampled and all samples
        are located in a smaller subregion of the full domain.
        The number of sampled cells is uniformly sampled between
        (min_fraction_of_sparse_data, max_fraction_of_sparse_data)

        Parameters
        ----------
        boolean_terrain : torch.Tensor
            Boolean representation of the terrain (true for wind cells and false for terrain cells)

        Returns
        -------
        mask : torch.Tensor
            Sampled output mask (true for sampled location and false everywhere else)
        '''
        mask = torch.zeros_like(boolean_terrain)
        mask_shape = mask.shape

        # allow for a maximum of 10 tries to get a mask containing non terrain cells
        iter = 0
        max_iter = 10
        inside_terrain = True
        while (iter < max_iter and inside_terrain):
            # create a subregion which is between 0.25 and 0.75 in each dimension
            sub_mask_shape = torch.Size(
                (torch.tensor(mask_shape) *
                 (torch.rand(3) * 0.5 + 0.25)).to(torch.long)
                )
            sub_mask = torch.zeros(sub_mask_shape)
            sub_mask_start_idx = torch.zeros(3).to(torch.long)
            sub_mask_start_idx[0] = int(
                self._rand.triangular(0, mask_shape[0] - sub_mask_shape[0], 0)
                )
            sub_mask_start_idx[1] = self._rand.randint(
                0, mask_shape[1] - sub_mask_shape[1]
                )
            sub_mask_start_idx[2] = self._rand.randint(
                0, mask_shape[2] - sub_mask_shape[2]
                )

            mean_terrain = boolean_terrain[sub_mask_start_idx[0]:sub_mask_start_idx[0] +
                                           sub_mask_shape[0],
                                           sub_mask_start_idx[1]:sub_mask_start_idx[1] +
                                           sub_mask_shape[1],
                                           sub_mask_start_idx[2]:sub_mask_start_idx[2] +
                                           sub_mask_shape[2]].float().mean().item()

            inside_terrain = mean_terrain == 0
            iter += 1

        if iter == max_iter:
            print('Did not find a valid mask within ' + str(iter) + 'iterations')

        # determine the number of cells to sample and correct the number of cells by the subregion size and the terrain occlusion
        target_frac = self._rand.random() * (
            self._max_fraction_of_sparse_data - self._min_fraction_of_sparse_data
            ) + self._min_fraction_of_sparse_data

        # limit it to a maximum 50 % of the cells to limit the cases where all the cells are sampled
        #target_num_cell = min(int(mask.numel() * target_frac), int(0.5 * sub_mask.numel()))
        target_num_cell = int(sub_mask.numel() * target_frac)

        # set the initial number of cells to a negative value to correct for the expected number of cells samples inside the terrain
        if inside_terrain:
            terrain_factor = 1.0
        else:
            terrain_factor = 1.0 / mean_terrain

        # limit it to 80 % of the cells to still ensure there are some free cells
        target_num_cell = min(
            int(target_num_cell * terrain_factor), int(0.8 * sub_mask.numel())
            )

        # sample cells
        indices = self._rand.sample(range(sub_mask.numel()), target_num_cell)
        sub_mask = sub_mask.view(-1)
        sub_mask[indices] = 1.0
        sub_mask = sub_mask.view(sub_mask_shape)

        mask[sub_mask_start_idx[0]:sub_mask_start_idx[0] + sub_mask_shape[0],
             sub_mask_start_idx[1]:sub_mask_start_idx[1] + sub_mask_shape[1],
             sub_mask_start_idx[2]:sub_mask_start_idx[2] + sub_mask_shape[2]] = sub_mask

        # make sure no cells are inside the terrain
        mask *= boolean_terrain

        return mask

    def _create_sparse_mask_trajectory(self, boolean_terrain):
        '''
        Creating a sparse mask by simulating a flight path.
        The trajectory length is randomly sampled between
        (trajectory_min_length, trajectory_max_length).
        It consists of straight line segments with a length between
        (trajectory_min_segment_length, trajectory_max_segment_length).

        Parameters
        ----------
        boolean_terrain : torch.Tensor
            Boolean representation of the terrain (true for wind cells and false for terrain cells)

        Returns
        -------
        mask : torch.Tensor
            Sampled output mask (true for sampled location and false everywhere else)
        '''
        mask = torch.zeros_like(boolean_terrain)
        mask_shape = mask.shape

        # initialize a random valid start position
        valid_start_positions = torch.nonzero(boolean_terrain, as_tuple=False)

        if self._trajectory_start_weighting_mode == 0:
            position = random.choices(valid_start_positions, k=1)[0]

        elif self._trajectory_start_weighting_mode == 1:
            cum_weights = torch.cumsum(mask.shape[0] - valid_start_positions[:, 0], 0)
            position = random.choices(
                valid_start_positions, cum_weights=cum_weights.numpy(), k=1
                )[0]

        elif self._trajectory_start_weighting_mode == 2:
            cum_weights = torch.cumsum((mask.shape[0] - valid_start_positions[:, 0])**2,
                                       0)
            position = random.choices(
                valid_start_positions, cum_weights=cum_weights.numpy(), k=1
                )[0]

        else:
            raise ValueError('Unsupported trajectory_start_weighting_mode')

        mask[position.split(1)] = 1.0

        # randomly determine a target trajectory length
        if self._trajectory_length_short_focus:
            trajectory_length = int(
                self._rand.triangular(
                    self._trajectory_min_length, self._trajectory_max_length,
                    self._trajectory_min_length
                    )
                )
        else:
            trajectory_length = self._rand.randint(
                self._trajectory_min_length, self._trajectory_max_length
                )

        # loop through adding segments until target length is achieved
        iter = 0
        while (iter < self._trajectory_max_iter) and mask.sum() < trajectory_length:
            iter += 1
            segment_length = self._rand.randint(
                self._trajectory_min_segment_length, self._trajectory_max_segment_length
                )

            # sample random propagation direction, divide z component by 4 to make flatter trajectories more likely
            direction = torch.randn(3)
            direction[0] *= 0.25
            direction /= direction.norm()

            num_steps = int(segment_length / self._trajectory_step_size)
            for i in range(num_steps):
                new_position = position + self._trajectory_step_size * direction
                new_idx = torch.round(new_position).to(torch.long)

                # check if the new position is inside the domain, if not invert the respective direction
                if torch.prod(new_idx > 0
                              ) * torch.prod(new_idx + 1 < torch.tensor(mask_shape)):
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

    def _update_sparse_stats(self, num_cells):
        '''
        Update the statistics of the sparse mask generation.
        
        Parameters
        ----------
        num_cells : int
            Number of sampled cells in the generated sparse mask.

        '''
        self._lock.acquire()
        self._min_num_cells = min(self._min_num_cells, num_cells)
        self._max_num_cells = max(self._max_num_cells, num_cells)
        if (self._num_samples < 1):
            self._num_samples = 1
            self._average_num_cells = num_cells
        else:
            self._num_samples += 1
            self._average_num_cells += (num_cells - self._average_num_cells) / float(
                self._num_samples
                )

        self._lock.release()

    def _compute_loss_weighting(self, data, ds, weighting_fn=0):
        '''
        This function computes the matrix to be used for loss weighting. Different weighting functions can be used, but all
        are normalized, so that the mean of W for an individual sample is 1.

        The following weighing function modes are supported:
            0: no weighting function, weights are ones
            1: squared pressure fluctuations
            2: l2 norm of the pressure gradient
            3: l2 norm of the velocity gradient

        Parameters
        ----------
        data : torch.Tensor
            4D input tensor [channels, Z, Y, X]
        grid_size : list of int
            Size of the input data in X, Y, and Z dimension
        ds : list of float
            Cell size of the data grid
        weighting_fn : int
            Specifies the weighing function that is used.

        Returns
        -------
        mask : torch.Tensor
            Sampled output mask (true for sampled location and false everywhere else)
        '''
        # no weighting, return empty tensor
        if weighting_fn == WeighingFunction.NONE:
            return torch.Tensor([])

        # squared pressure fluctuations weighting function
        elif weighting_fn == WeighingFunction.SQUARED_PRESSURE_FLUCTUATIONS:

            # get pressure and mean pressure per sample
            p_index = self._channels_to_load.index('p')
            p = data[p_index].unsqueeze(0)
            p_mean = p.mean(-1).mean(-1).mean(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(
                -1
                ).expand_as(p)

            # remove mean, square and remove outliers
            W = (p - p_mean)**2

            if self._loss_weighting_clamp:
                # TODO: make the clamping value a parameter that can be set from the YAML config file
                W = W.clamp(0.0435)

            # normalize by its volume integral per sample
            W = W / (
                W.mean(-1).mean(-1).mean(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                .expand_as(p)
                )

        # l2-norm pressure gradient weighting function
        elif weighting_fn == WeighingFunction.L2_PRESSURE_GRADIENTS:
            # get pressure
            p = data[self._channels_to_load.index('p')].unsqueeze(0)

            # get terrain
            terrain = data[self._channels_to_load.index('terrain')].unsqueeze(0)

            # compute the spatial pressure gradient components and take the l2-norm of the gradient and remove outliers
            W = (
                windseer_utils.derive(p, 3, ds[0], terrain)**2 +
                windseer_utils.derive(p, 2, ds[1], terrain)**2 +
                windseer_utils.derive(p, 1, ds[2], terrain)**2
                )**0.5

            if self._loss_weighting_clamp:
                # TODO: make the clamping value a parameter that can be set from the YAML config file
                W = W.clamp(0.000814)

            # normalize by its volume integral per sample
            W = (
                W / ((W).mean(-1).mean(-1).mean(-1).unsqueeze(-1).unsqueeze(-1)
                     .unsqueeze(-1).expand_as(W))
                )

        # l2-norm velocity gradient weighting function
        elif weighting_fn == WeighingFunction.L2_VELOCITY_GRADIENTS:

            vel_indices = torch.LongTensor([
                self._channels_to_load.index(channel) for channel in ['ux', 'uy', 'uz']
                ])

            # get the velocities
            U = data.index_select(0, vel_indices).unsqueeze(0)

            # get terrain
            terrain = data[self._channels_to_load.index('terrain')].unsqueeze(0)

            # compute the spatial gradient tensor of the velocity gradient and take the l2-norm of the gradient per sample and remove outliers
            W = (windseer_utils.gradient(U, ds, terrain)**2).sum(1)**0.5

            if self._loss_weighting_clamp:
                # TODO: make the clamping value a parameter that can be set from the YAML config file
                W = W.clamp(0.00175)

            # normalize by its volume integral per sample
            W = (
                W / ((W).mean(-1).mean(-1).mean(-1).unsqueeze(-1).unsqueeze(-1)
                     .unsqueeze(-1).expand_as(W))
                )

        else:
            raise ValueError('Unrecognized weighting function.')

        # handling for zero samples which create NaNs
        W[torch.isnan(W)] = 1.0
        return W

    def _get_smooth_data(self, data, mask, grid_size):
        '''
        Set the values of the unknown cells in the input tensor according to the settings
        (input_smoothing_interpolation, input_smoothing_interpolation_linear)

        Parameters
        ----------
        data : torch.Tensor
            Input data tensor
        mask : torch.Tensor
            Mask indicating the cells containing valid measurements
        grid_size : list of int
            Dimensions of the data tensor

        Returns
        -------
        data_smoothed : torch.Tensor
            Processed data tensor
        '''
        if self._input_smoothing_interpolation:
            return interpolate_sparse_data(
                data, mask, grid_size, self._input_smoothing_interpolation_linear
                )
        else:
            data_smoothed = torch.ones_like(data)
            scale = data.sum(-1).sum(-1).sum(-1) / mask.sum()

            data_smoothed *= scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            for i in range(data.shape[0]):
                data_smoothed[i, mask] = data[i, mask]

            return data_smoothed
