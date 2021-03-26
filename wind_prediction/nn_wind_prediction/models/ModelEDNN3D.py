import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import nn_wind_prediction.utils as utils
import nn_wind_prediction.nn as nn_custom

from .ModelBase import ModelBase

'''
Encoder/Decoder Neural Network

The first input layer is assumed to be terrain information. It should be zero in the terrain and nonzero elsewhere.
'''
class ModelEDNN3D(ModelBase):
    __default_activation = nn.LeakyReLU
    __default_activation_kwargs = {'negative_slope': 0.1}

    def __init__(self, **kwargs):
        super(ModelEDNN3D, self).__init__()

        parser = utils.KwargsParser(kwargs, 'EDNN3D')
        verbose = parser.get_safe('verbose', False, bool, False)
        self.__use_terrain_mask = parser.get_safe('use_terrain_mask', True, bool, verbose)
        self.__n_downsample_layers = parser.get_safe('n_downsample_layers', 4, int, verbose)
        self.__filter_kernel_size = parser.get_safe('filter_kernel_size', 3, int, verbose)
        self.__n_first_conv_channels = parser.get_safe('n_first_conv_channels', 8, int, verbose)
        self.__channel_multiplier = parser.get_safe('channel_multiplier', 2, float, verbose)
        self.__use_mapping_layer = parser.get_safe('use_mapping_layer', False, bool, verbose)
        self.__use_fc_layers = parser.get_safe('use_fc_layers', True, bool, verbose)
        self.__fc_scaling = parser.get_safe('fc_scaling', 8.0, float, verbose)
        self.__potential_flow = parser.get_safe('potential_flow', False, bool, verbose)
        self.__n_x = parser.get_safe('n_x', 64, int, verbose)
        self.__n_y = parser.get_safe('n_y', 64, int, verbose)
        self.__n_z = parser.get_safe('n_z', 64, int, verbose)
        self.__interpolation_mode = parser.get_safe('interpolation_mode',  'nearest', str, verbose)
        self.__skipping = parser.get_safe('skipping', True, bool, verbose)
        self.__align_corners = parser.get_safe('align_corners', False, bool, verbose)
        self.__use_terrain_mask = parser.get_safe('align_corners', True, bool, verbose)
        self.__pooling_method = parser.get_safe('pooling_method',  'striding', str, verbose)
        self.__grid_size = parser.get_safe('grid_size',  [1, 1, 1], list, verbose)
        self.__use_uz_in = parser.get_safe('use_uz_in', True, bool, verbose)
        self.__use_turbulence = parser.get_safe('use_turbulence', True, bool, verbose)
        self.__use_pressure = parser.get_safe('use_pressure', False, bool, verbose)
        self.__use_epsilon = parser.get_safe('use_epsilon', False, bool, verbose)
        self.__use_nut = parser.get_safe('use_nut', False, bool, verbose)
        self.__vae = parser.get_safe('vae', False, bool, verbose)
        self.__logvar_scaling = parser.get_safe('logvar_scaling', 10.0, float, verbose)
        self.__predict_uncertainty = parser.get_safe('predict_uncertainty', False, bool, verbose)
        self.__use_sparse_mask = parser.get_safe('use_sparse_mask', False, bool, verbose)
        self.__use_sparse_convolution = parser.get_safe('use_sparse_convolution', False, bool, verbose)

        if self.__vae and not self.__use_fc_layers:
            print('EDNN3D: Error, to use the vae mode the fc layers need to be enabled.')
            sys.exit()

        if self.__n_downsample_layers <= 0:
            print('EDNN3D: Error, n_downsample_layers must be larger than 0')
            sys.exit()

        if self.__align_corners == False:
            self.__align_corners = None

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
        self.__conv = nn.ModuleList()

        if self.__use_sparse_convolution:
            Conv = nn_custom.SparseConv
            kwargs_conv = {'conv_type': nn.Conv3d,
                           'mask_exclude_first_dim': True}

        else:
            Conv = nn.Conv3d
            kwargs_conv = {}

        for i in range(self.__n_downsample_layers):
            if i == 0:
                kwargs_conv['in_channels'] = self.num_inputs
                kwargs_conv['out_channels'] = self.__n_first_conv_channels
                kwargs_conv['kernel_size'] = self.__filter_kernel_size
                self.__conv += [Conv(**kwargs_conv)]
                if 'mask_exclude_first_dim' in kwargs_conv:
                    kwargs_conv['mask_exclude_first_dim'] = False

            else:
                kwargs_conv['in_channels'] = int(self.__n_first_conv_channels*(self.__channel_multiplier**(i-1)))
                kwargs_conv['out_channels'] = int(self.__n_first_conv_channels*(self.__channel_multiplier**i))
                kwargs_conv['kernel_size'] = self.__filter_kernel_size
                self.__conv += [Conv(**kwargs_conv)]

        # fully connected layers
        if self.__use_fc_layers:
            if self.__n_downsample_layers == 0:
                n_features = int(self.num_inputs * self.__n_x * self.__n_y * self.__n_z)
            else:
                n_features = int(int(self.__n_first_conv_channels*(self.__channel_multiplier**(self.__n_downsample_layers-1))) *  # number of channels
                                 self.__n_x * self.__n_y * self.__n_z / ((2**self.__n_downsample_layers)**3)) # number of pixels

            if self.__vae:
                self.__vae_dim = int(n_features/self.__fc_scaling)
                self.__fc1 = nn.Linear(n_features, 2 * int(n_features/self.__fc_scaling))
                print("EDNN3D: VAE state space is {} dimensional".format(self.__vae_dim))
            else:
                self.__fc1 = nn.Linear(n_features, int(n_features/self.__fc_scaling))

            self.__fc2 = nn.Linear(int(n_features/self.__fc_scaling), n_features)
        else:
            kwargs_conv['in_channels'] = int(self.__n_first_conv_channels*(self.__channel_multiplier**(self.__n_downsample_layers-1)))
            kwargs_conv['out_channels'] = int(self.__n_first_conv_channels*(self.__channel_multiplier**self.__n_downsample_layers))
            kwargs_conv['kernel_size'] = self.__filter_kernel_size
            self.__c1 = Conv(**kwargs_conv)

            self.__c2 = nn.Conv3d(int(self.__n_first_conv_channels*(self.__channel_multiplier**self.__n_downsample_layers)),
                                  int(self.__n_first_conv_channels*(self.__channel_multiplier**(self.__n_downsample_layers-1))),
                                  self.__filter_kernel_size)

        # up-convolution layers
        self.__deconv1 = nn.ModuleList()
        self.__deconv2 = nn.ModuleList()

        num_out = self.num_outputs
        if self.__predict_uncertainty:
            num_out *= 2

        if (self.__skipping):
            for i in range(self.__n_downsample_layers):
                if i == 0:
                    self.__deconv1 += [nn.Conv3d(2*self.__n_first_conv_channels, self.__n_first_conv_channels, self.__filter_kernel_size+1)]
                    self.__deconv2 += [nn.Conv3d(self.__n_first_conv_channels, num_out, self.__filter_kernel_size+1)]
                else:
                    self.__deconv1 += [nn.Conv3d(2*int(self.__n_first_conv_channels*(self.__channel_multiplier**i)),
                                                 int(self.__n_first_conv_channels*(self.__channel_multiplier**i)),
                                                 self.__filter_kernel_size+1)]
                    self.__deconv2 += [nn.Conv3d(int(self.__n_first_conv_channels*(self.__channel_multiplier**i)),
                                                 int(self.__n_first_conv_channels*(self.__channel_multiplier**(i-1))),
                                                 self.__filter_kernel_size+1)]

        else:
            for i in range(self.__n_downsample_layers):
                if i == 0:
                    self.__deconv1 += [nn.Conv3d(self.__n_first_conv_channels, num_out, self.__filter_kernel_size+1)]
                else:
                    self.__deconv1 += [nn.Conv3d(int(self.__n_first_conv_channels*(self.__channel_multiplier**i)),
                                                 int(self.__n_first_conv_channels*(self.__channel_multiplier**(i-1))),
                                                 self.__filter_kernel_size+1)]

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
            is_wind = x[:, 0, :].unsqueeze(1).clone()
            is_wind.sign_()

        output = {}
        x_skip = []

        # down-convolution
        if (self.__skipping):
            for i in range(self.__n_downsample_layers):
                x = self.__activation(self.__conv[i](self.__pad_conv(x)))
                x_skip.append(x.clone())
                x = self.__pooling(x)
        else:
            for i in range(self.__n_downsample_layers):
                x = self.__pooling(self.__activation(self.__conv[i](self.__pad_conv(x))))

        # fully connected layers
        if self.__use_fc_layers:
            shape = x.size()
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
            x = self.__activation(self.__c1(self.__pad_conv(x)))
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

