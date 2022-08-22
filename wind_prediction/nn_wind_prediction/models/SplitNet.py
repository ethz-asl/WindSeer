import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import nn_wind_prediction.utils as utils
import nn_wind_prediction.nn as nn_custom

from .ModelBase import ModelBase

'''
Encoder/Decoder Neural Network that has a separate encoding for the terrain and the rest of the input

The first input layer is assumed to be terrain information. It should be zero in the terrain and nonzero elsewhere.
'''
class SplitNet(ModelBase):
    __default_activation = nn.LeakyReLU
    __default_activation_kwargs = {'negative_slope': 0.1}
    __default_filter_kernel_size = 3
    __default_n_first_conv_channels = 8
    __default_channel_multiplier = 2
    __default_n_downsample_layers = 4
    __default_use_terrain_mask = True
    __default_use_mapping_layer = False
    __default_use_fc_layers = True
    __default_fc_scaling = 8
    __default_potential_flow = False
    __default_n_x = 64
    __default_n_y = 64
    __default_n_z = 64
    __default_interpolation_mode = 'nearest'
    __default_skipping = True
    __default_align_corners = None
    __default_pooling_method = 'striding'
    __default_use_uz_in = True
    __default_use_turbulence = True
    __default_use_pressure = False
    __default_use_epsilon = False
    __default_use_nut = False
    __default_grid_size = [1, 1, 1]
    __default_vae = False
    __default_logvar_scaling = 10
    __default_predict_uncertainty = False
    __default_use_sparse_mask = False
    __default_use_sparse_convolution = False

    def __init__(self, **kwargs):
        super(SplitNet, self).__init__()

        try:
            verbose = kwargs['verbose']
        except KeyError:
            verbose = False

        try:
            self.__use_terrain_mask = kwargs['use_terrain_mask']
        except KeyError:
            self.__use_terrain_mask = self.__default_use_terrain_mask
            if verbose:
                print('SplitNet: use_terrain_mask not present in kwargs, using default value:', self.__default_use_terrain_mask)

        try:
            self.__n_downsample_layers = kwargs['n_downsample_layers']
        except KeyError:
            self.__n_downsample_layers = self.__default_n_downsample_layers
            if verbose:
                print('SplitNet: n_downsample_layers not present in kwargs, using default value:', self.__default_n_downsample_layers)

        try:
            self.__filter_kernel_size = int(kwargs['filter_kernel_size']) # needs to be an integer
        except KeyError:
            self.__filter_kernel_size = self.__default_filter_kernel_size
            if verbose:
                print('SplitNet: filter_kernel_size not present in kwargs, using default value:', self.__default_filter_kernel_size)

        try:
            self.__n_first_conv_channels = int(kwargs['n_first_conv_channels']) # needs to be an integer
        except KeyError:
            self.__n_first_conv_channels = self.__default_n_first_conv_channels
            if verbose:
                print('SplitNet: n_first_conv_channels not present in kwargs, using default value:', self.__default_n_first_conv_channels)

        try:
            self.__channel_multiplier = kwargs['channel_multiplier']
        except KeyError:
            self.__channel_multiplier = self.__default_channel_multiplier
            if verbose:
                print('SplitNet: channel_multiplier not present in kwargs, using default value:', self.__default_channel_multiplier)

        try:
            self.__use_mapping_layer = kwargs['use_mapping_layer']
        except KeyError:
            self.__use_mapping_layer = self.__default_use_mapping_layer
            if verbose:
                print('SplitNet: use_mapping_layer not present in kwargs, using default value:', self.__default_use_mapping_layer)

        try:
            self.__use_fc_layers = kwargs['use_fc_layers']
        except KeyError:
            self.__use_fc_layers = self.__default_use_fc_layers
            if verbose:
                print('SplitNet: use_fc_layers not present in kwargs, using default value:', self.__default_use_fc_layers)

        try:
            self.__fc_scaling = kwargs['fc_scaling']
        except KeyError:
            self.__fc_scaling = self.__default_fc_scaling
            if verbose:
                print('SplitNet: fc_scaling not present in kwargs, using default value:', self.__default_fc_scaling)

        try:
            self.__potential_flow = kwargs['potential_flow']
        except KeyError:
            self.__potential_flow = self.__default_potential_flow
            if verbose:
                print('SplitNet: potential_flow not present in kwargs, using default value:', self.__default_potential_flow)

        try:
            self.__n_x = kwargs['n_x']
        except KeyError:
            self.__n_x = self.__default_n_x
            if verbose:
                print('SplitNet: n_x not present in kwargs, using default value:', self.__default_n_x)

        try:
            self.__n_y = kwargs['n_y']
        except KeyError:
            self.__n_y = self.__default_n_y
            if verbose:
                print('SplitNet: n_y not present in kwargs, using default value:', self.__default_n_y)

        try:
            self.__n_z = kwargs['n_z']
        except KeyError:
            self.__n_z = self.__default_n_z
            if verbose:
                print('SplitNet: n_z not present in kwargs, using default value:', self.__default_n_z)

        try:
            self.__interpolation_mode = kwargs['interpolation_mode']
        except KeyError:
            self.__interpolation_mode = self.__default_interpolation_mode
            if verbose:
                print('SplitNet: interpolation_mode not present in kwargs, using default value:', self.__default_interpolation_mode)

        try:
            self.__skipping = kwargs['skipping']
        except KeyError:
            self.__skipping = self.__default_skipping
            if verbose:
                print('SplitNet: skipping not present in kwargs, using default value:', self.__default_skipping)

        try:
            self.__align_corners = kwargs['align_corners']
            if self.__align_corners == False:
                self.__align_corners = None
        except KeyError:
            self.__align_corners = self.__default_align_corners
            if verbose:
                print('SplitNet: align_corners not present in kwargs, using default value:', self.__default_align_corners)

        try:
            self.__pooling_method = kwargs['pooling_method']
        except KeyError:
            self.__pooling_method = self.__default_pooling_method
            if verbose:
                print('SplitNet: pooling_method not present in kwargs, using default value:', self.__default_pooling_method)

        try:
            self.__grid_size = kwargs['grid_size']
        except KeyError:
            self.__grid_size = self.__default_grid_size
            if verbose:
                print('SplitNet: grid_size is not present in kwargs, using default value:', self.__default_grid_size)

        try:
            self.__use_uz_in = kwargs['use_uz_in']
        except KeyError:
            self.__use_uz_in = self.__default_use_uz_in
            if verbose:
                print('SplitNet: use_uz_in not present in kwargs, using default value:', self.__default_use_uz_in)

        try:
            self.__use_turbulence = kwargs['use_turbulence']
        except KeyError:
            self.__use_turbulence = self.__default_use_turbulence
            if verbose:
                print('SplitNet: use_turbulence not present in kwargs, using default value:', self.__default_use_turbulence)

        try:
            self.__use_pressure = kwargs['use_pressure']
        except KeyError:
            self.__use_pressure = self.__default_use_pressure
            if verbose:
                print('SplitNet: use_pressure not present in kwargs, using default value:', self.__default_use_pressure)

        try:
            self.__use_epsilon = kwargs['use_epsilon']
        except KeyError:
            self.__use_epsilon = self.__default_use_epsilon
            if verbose:
                print('SplitNet: use_epsilon not present in kwargs, using default value:', self.__default_use_epsilon)

        try:
            self.__use_nut = kwargs['use_nut']
        except KeyError:
            self.__use_nut = self.__default_use_nut
            if verbose:
                print('SplitNet: use_nut not present in kwargs, using default value:', self.__default_use_nut)

        try:
            self.__vae = kwargs['vae']
        except KeyError:
            self.__vae = self.__default_vae
            if verbose:
                print('SplitNet: vae not present in kwargs, using default value:', self.__default_vae)

        try:
            self.__logvar_scaling = kwargs['logvar_scaling']
        except KeyError:
            self.__logvar_scaling = self.__default_logvar_scaling
            if verbose:
                print('SplitNet: logvar_scaling not present in kwargs, using default value:', self.__default_logvar_scaling)

        try:
            self.__predict_uncertainty = kwargs['predict_uncertainty']
        except KeyError:
            self.__predict_uncertainty = self.__default_predict_uncertainty
            if verbose:
                print('SplitNet: predict_uncertainty not present in kwargs, using default value:', self.__default_predict_uncertainty)

        try:
            self.__use_sparse_mask = kwargs['use_sparse_mask']
        except KeyError:
            self.__use_sparse_mask = self.__default_use_sparse_mask
            if verbose:
                print('SplitNet: use_sparse_mask not present in kwargs, using default value:', self.__default_use_sparse_mask)

        try:
            self.__use_sparse_convolution = kwargs['use_sparse_convolution']
        except KeyError:
            self.__use_sparse_convolution = self.__default_use_sparse_convolution
            if verbose:
                print('SplitNet: use_sparse_convolution not present in kwargs, using default value:', self.__default_use_sparse_convolution)

        if self.__vae and not self.__use_fc_layers:
            print('SplitNet: Error, to use the vae mode the fc layers need to be enabled.')
            sys.exit()

        if self.__n_downsample_layers <= 0:
            print('SplitNet: Error, n_downsample_layers must be larger than 0')
            sys.exit()

        # input variable check
        if (self.__filter_kernel_size % 2 == 0) or (self.__filter_kernel_size < 1):
            raise ValueError('The filter kernel size needs to be odd and larger than 0.')

        # construct the number of input and output channels based on the parameters
        self.num_inputs = 3  # (terrain, u_x_in, u_y_in)

        if self.__use_uz_in:
            self.num_inputs += 1 # uz

        if self.__use_sparse_mask:
            self.num_inputs += 1  # sparse_mask

        self.num_outputs = 3  # (u_x_out, u_y_out, u_z_out)

        if self.__use_turbulence:
            self.num_outputs += 1  # turb. kin. en.

        if self.__use_pressure:
            self.num_outputs += 1  # pressure

        if self.__use_epsilon:
            self.num_outputs += 1  # dissipation

        if self.__use_nut:
            self.num_outputs += 1  # viscosity

        try:
            self.num_inputs = kwargs['force_num_inputs']
        except KeyError:
            pass

        try:
            self.num_outputs = kwargs['force_num_outputs']
            self.__predict_uncertainty = False
        except KeyError:
            pass

        # down-convolution layers
        self.__conv_terrain = nn.ModuleList()
        self.__conv_input = nn.ModuleList()

        if self.__use_sparse_convolution:
            Conv = nn_custom.SparseConv
            kwargs_conv = {'conv_type': nn.Conv3d,
                           'mask_exclude_first_dim': True}

        else:
            Conv = nn.Conv3d
            kwargs_conv = {}

        num_channels = []
        for i in range(self.__n_downsample_layers):
            if i == 0:
                self.__conv_terrain += [nn.Conv3d(1, self.__n_first_conv_channels, self.__filter_kernel_size)]

                kwargs_conv['in_channels'] = self.num_inputs - 1
                kwargs_conv['out_channels'] = self.__n_first_conv_channels
                kwargs_conv['kernel_size'] = self.__filter_kernel_size
                self.__conv_input += [Conv(**kwargs_conv)]
                if 'mask_exclude_first_dim' in kwargs_conv:
                    kwargs_conv['mask_exclude_first_dim'] = False

                num_channels += [2 * self.__n_first_conv_channels]

            else:
                kwargs_conv['in_channels'] = int(self.__n_first_conv_channels*(self.__channel_multiplier**(i-1)))
                kwargs_conv['out_channels'] = int(self.__n_first_conv_channels*(self.__channel_multiplier**i))
                kwargs_conv['kernel_size'] = self.__filter_kernel_size

                self.__conv_terrain += [nn.Conv3d(kwargs_conv['in_channels'], kwargs_conv['out_channels'],
                                        self.__filter_kernel_size)]

                self.__conv_input += [Conv(**kwargs_conv)]

                num_channels += [2 * kwargs_conv['out_channels']]

        # fully connected layers
        if self.__use_fc_layers:
            if self.__n_downsample_layers == 0:
                n_features = int(self.num_inputs * self.__n_x * self.__n_y * self.__n_z)
            else:
                n_features = int(num_channels[-1] *  # number of channels
                                 self.__n_x * self.__n_y * self.__n_z / ((2**self.__n_downsample_layers)**3)) # number of pixels

            self.__num_flat_features = n_features

            if self.__vae:
                self.__vae_dim = int(n_features/self.__fc_scaling)
                self.__fc1 = nn.Linear(n_features, 2 * int(n_features/self.__fc_scaling))
                print("SplitNet: VAE state space is {} dimensional".format(self.__vae_dim))
            else:
                self.__fc1 = nn.Linear(n_features, int(n_features/self.__fc_scaling))

            self.__fc2 = nn.Linear(int(n_features/self.__fc_scaling), n_features)
        else:
            kwargs_conv['in_channels'] = int(self.__n_first_conv_channels*(self.__channel_multiplier**(self.__n_downsample_layers-1)))
            kwargs_conv['out_channels'] = int(self.__n_first_conv_channels*(self.__channel_multiplier**self.__n_downsample_layers))
            kwargs_conv['kernel_size'] = self.__filter_kernel_size
            
            self.__c1_terrain = nn.Conv3d(kwargs_conv['in_channels'], kwargs_conv['out_channels'],
                                          self.__filter_kernel_size)


            self.__c1_input = Conv(**kwargs_conv)

            num_channels += [2 * kwargs_conv['out_channels']]

            self.__c2 = nn.Conv3d(num_channels[-1], num_channels[-2], self.__filter_kernel_size)

        # up-convolution layers
        self.__deconv1 = nn.ModuleList()
        self.__deconv2 = nn.ModuleList()

        num_out = self.num_outputs
        if self.__predict_uncertainty:
            num_out *= 2

        out_channels = [num_out] + num_channels
        if (self.__skipping):
            for i in range(self.__n_downsample_layers):
                self.__deconv1 += [nn.Conv3d(2 * out_channels[i + 1], out_channels[i + 1], self.__filter_kernel_size+1)]
                self.__deconv2 += [nn.Conv3d(out_channels[i + 1], out_channels[i], self.__filter_kernel_size+1)]

        else:
            for i in range(self.__n_downsample_layers):
                self.__deconv1 += [nn.Conv3d(out_channels[i + 1], out_channels[i], self.__filter_kernel_size+1)]


        # mapping layer
        if self.__use_mapping_layer:
            self.__mapping_layer = nn.Conv3d(num_out, num_out,1, groups=num_out)  # for each channel a separate filter

        # padding modules
        padding_required = int((self.__filter_kernel_size - 1) / 2)
        self.__pad_conv = nn.ReplicationPad3d(padding_required)
        self.__pad_deconv = nn.ReplicationPad3d((padding_required, padding_required+1, padding_required, padding_required+1, padding_required, padding_required+1))

        # Check if we have defined a specific activation layer
        try:
            activation = getattr(nn, kwargs['activation_type'])
            self.__activation = activation(**kwargs['activation_args'])
        except KeyError as e:
            print('Activation function not specified or not found, using default: {0}'.format(self.__default_activation))
            self.__activation = self.__default_activation(**self.__default_activation_kwargs)

        # pooling module
        if (self.__pooling_method == 'averagepool'):
            self.__pooling = nn.AvgPool3d(2)
        elif (self.__pooling_method == 'maxpool'):
            self.__pooling = nn.MaxPool3d(2)
        elif (self.__pooling_method == 'striding'):
            self.__pooling = nn.MaxPool3d(1, stride=2)
        else:
            raise ValueError('The pooling method value is invalid: ' + self.__pooling_method)

    def forward(self, x):
        if self.__use_terrain_mask:
            # store the terrain data
            is_wind = x[:, 0].unsqueeze(1).clone()
            is_wind.sign_()

        output = {}
        x_skip = []

        # down-convolution
        x_terrain = x[:, 0].unsqueeze(1)
        x = x[:, 1:]
        if (self.__skipping):
            for i in range(self.__n_downsample_layers):
                x = self.__activation(self.__conv_input[i](self.__pad_conv(x)))
                x_terrain = self.__activation(self.__conv_terrain[i](self.__pad_conv(x_terrain)))
                x_skip.append(torch.cat((x, x_terrain), 1).clone())
                x = self.__pooling(x)
                x_terrain = self.__pooling(x_terrain)
        else:
            for i in range(self.__n_downsample_layers):
                x = self.__pooling(self.__activation(self.__conv_input[i](self.__pad_conv(x))))
                x_terrain = self.__pooling(self.__activation(self.__conv_terrain[i](self.__pad_conv(x_terrain))))

        # fully connected layers
        if self.__use_fc_layers:
            x = torch.cat((x, x_terrain), 1)
            shape = x.size()
            del x_terrain
            x = x.view(-1, self.num_flat_features(x))
            x = self.__activation(self.__fc1(x))
            if self.__vae:
                x_mean = x[:,:self.__vae_dim]
                x_logvar = x[:,self.__vae_dim:]
                output['distribution_mean'] = x_mean.clone()
                output['distribution_logvar'] = x_logvar.clone()

                # during training sample from the distribution, during inference take values with the maximum probability
                if self.training:
                    std = torch.torch.exp(0.5 * x_logvar)
                    x = x_mean + std * torch.randn_like(std)
                else:
                    x = x_mean

            if self.__vae:
                output["encoding"] = x.clone()

            x = self.__activation(self.__fc2(x))
            x = x.view(shape)
        else:
            x = self.__activation(self.__c1_input(self.__pad_conv(x)))
            x_terrain = self.__activation(self.__c1_terrain(self.__pad_conv(x_terrain)))

            x = torch.cat((x, x_terrain), 1)

            if self.__vae:
                output["encoding"] = x.view(-1, self.num_flat_features(x)).clone()

            x = self.__activation(self.__c2(self.__pad_conv(x)))

        # up-convolution
        if (self.__skipping):
            for i in range(self.__n_downsample_layers-1, -1, -1):
                if (i == 0):
                    # no nonlinearity in the output layer
                    x = self.__deconv2[i](self.__pad_deconv(self.__activation(self.__deconv1[i](self.__pad_deconv(
                        torch.cat([F.interpolate(x, scale_factor=2, mode=self.__interpolation_mode, align_corners=self.__align_corners),
                                   x_skip[i]], 1))))))
                else:
                    x = self.__activation(self.__deconv2[i](self.__pad_deconv(self.__activation(self.__deconv1[i](self.__pad_deconv(
                        torch.cat([F.interpolate(x, scale_factor=2, mode=self.__interpolation_mode, align_corners=self.__align_corners),
                                   x_skip[i]], 1)))))))
        else:
            for i in range(self.__n_downsample_layers-1, -1, -1):
                if (i == 0):
                    # no nonlinearity in the output layer
                    x = self.__deconv1[i](self.__pad_deconv(
                        F.interpolate(x, scale_factor=2, mode=self.__interpolation_mode, align_corners=self.__align_corners)))
                else:
                    x = self.__activation(self.__deconv1[i](self.__pad_deconv(
                        F.interpolate(x, scale_factor=2, mode=self.__interpolation_mode, align_corners=self.__align_corners))))

        if self.__use_mapping_layer:
            x = self.__mapping_layer(x)

        if self.__potential_flow:
            x = torch.cat([utils.curl(x, self.__grid_size, x[:, 0, :]), x[:, 3:, :]], 1)

        if self.__use_terrain_mask:
            x = is_wind.repeat(1, x.shape[1], 1, 1, 1) * x

        if self.__predict_uncertainty:
            output['pred'] = x[:,:self.num_outputs]
            output['logvar'] = self.__logvar_scaling * torch.nn.functional.softsign(x[:,self.num_outputs:])
        else:
            output['pred'] = x

        return output

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features