import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import nn_wind_prediction.utils as utils
from nn_wind_prediction.models.ConvLSTMLayer import ConvLSTM3d

from .ModelBase import ModelBase

'''
Encoder/Decoder Neural Network

1. Five layers of convolution and max pooling up to 128 channels
2. Two fully connected layers
3. Five times upsampling followed by convolution
4. Mapping layer with a separate filter for each output channel

The first input layer is assumed to be terrain information. It should be zero in the terrain and nonzero elsewhere.
'''
class ModelEDNN3D(ModelBase):
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
    __default_use_hybrid_model = False
    __default_use_hybrid_model_ext = False

    def __init__(self, **kwargs):
        super(ModelEDNN3D, self).__init__()

        try:
            verbose = kwargs['verbose']
        except KeyError:
            verbose = False

        try:
            self.__use_terrain_mask = kwargs['use_terrain_mask']
        except KeyError:
            self.__use_terrain_mask = self.__default_use_terrain_mask
            if verbose:
                print('EDNN3D: use_terrain_mask not present in kwargs, using default value:', self.__default_use_terrain_mask)

        try:
            self.__n_downsample_layers = kwargs['n_downsample_layers']
        except KeyError:
            self.__n_downsample_layers = self.__default_n_downsample_layers
            if verbose:
                print('EDNN3D: n_downsample_layers not present in kwargs, using default value:', self.__default_n_downsample_layers)

        try:
            self.__filter_kernel_size = int(kwargs['filter_kernel_size']) # needs to be an integer
        except KeyError:
            self.__filter_kernel_size = self.__default_filter_kernel_size
            if verbose:
                print('EDNN3D: filter_kernel_size not present in kwargs, using default value:', self.__default_filter_kernel_size)

        try:
            self.__n_first_conv_channels = int(kwargs['n_first_conv_channels']) # needs to be an integer
        except KeyError:
            self.__n_first_conv_channels = self.__default_n_first_conv_channels
            if verbose:
                print('EDNN3D: n_first_conv_channels not present in kwargs, using default value:', self.__default_n_first_conv_channels)

        try:
            self.__channel_multiplier = kwargs['channel_multiplier']
        except KeyError:
            self.__channel_multiplier = self.__default_channel_multiplier
            if verbose:
                print('EDNN3D: channel_multiplier not present in kwargs, using default value:', self.__default_channel_multiplier)

        try:
            self.__use_mapping_layer = kwargs['use_mapping_layer']
        except KeyError:
            self.__use_mapping_layer = self.__default_use_mapping_layer
            if verbose:
                print('EDNN3D: use_mapping_layer not present in kwargs, using default value:', self.__default_use_mapping_layer)

        try:
            self.__use_fc_layers = kwargs['use_fc_layers']
        except KeyError:
            self.__use_fc_layers = self.__default_use_fc_layers
            if verbose:
                print('EDNN3D: use_fc_layers not present in kwargs, using default value:', self.__default_use_fc_layers)

        try:
            self.__fc_scaling = kwargs['fc_scaling']
        except KeyError:
            self.__fc_scaling = self.__default_fc_scaling
            if verbose:
                print('EDNN3D: fc_scaling not present in kwargs, using default value:', self.__default_fc_scaling)

        try:
            self.__potential_flow = kwargs['potential_flow']
        except KeyError:
            self.__potential_flow = self.__default_potential_flow
            if verbose:
                print('EDNN3D: potential_flow not present in kwargs, using default value:', self.__default_potential_flow)

        try:
            self.__n_x = kwargs['n_x']
        except KeyError:
            self.__n_x = self.__default_n_x
            if verbose:
                print('EDNN3D: n_x not present in kwargs, using default value:', self.__default_n_x)

        try:
            self.__n_y = kwargs['n_y']
        except KeyError:
            self.__n_y = self.__default_n_y
            if verbose:
                print('EDNN3D: n_y not present in kwargs, using default value:', self.__default_n_y)

        try:
            self.__n_z = kwargs['n_z']
        except KeyError:
            self.__n_z = self.__default_n_z
            if verbose:
                print('EDNN3D: n_z not present in kwargs, using default value:', self.__default_n_z)

        try:
            self.__interpolation_mode = kwargs['interpolation_mode']
        except KeyError:
            self.__interpolation_mode = self.__default_interpolation_mode
            if verbose:
                print('EDNN3D: interpolation_mode not present in kwargs, using default value:', self.__default_interpolation_mode)

        try:
            self.__skipping = kwargs['skipping']
        except KeyError:
            self.__skipping = self.__default_skipping
            if verbose:
                print('EDNN3D: skipping not present in kwargs, using default value:', self.__default_skipping)

        try:
            self.__align_corners = kwargs['align_corners']
            if self.__align_corners == False:
                self.__align_corners = None
        except KeyError:
            self.__align_corners = self.__default_align_corners
            if verbose:
                print('EDNN3D: align_corners not present in kwargs, using default value:', self.__default_align_corners)

        try:
            self.__pooling_method = kwargs['pooling_method']
        except KeyError:
            self.__pooling_method = self.__default_pooling_method
            if verbose:
                print('EDNN3D: pooling_method not present in kwargs, using default value:', self.__default_pooling_method)

        try:
            self.__grid_size = kwargs['grid_size']
        except KeyError:
            self.__grid_size = self.__default_grid_size
            if verbose:
                print('EDNN3D: grid_size is not present in kwargs, using default value:', self.__default_grid_size)

        try:
            self.__use_turbulence = kwargs['use_turbulence']
        except KeyError:
            self.__use_turbulence = self.__default_use_turbulence
            if verbose:
                print('EDNN3D: use_turbulence not present in kwargs, using default value:', self.__default_use_turbulence)

        try:
            self.__use_pressure = kwargs['use_pressure']
        except KeyError:
            self.__use_pressure = self.__default_use_pressure
            if verbose:
                print('EDNN3D: use_pressure not present in kwargs, using default value:', self.__default_use_pressure)

        try:
            self.__use_epsilon = kwargs['use_epsilon']
        except KeyError:
            self.__use_epsilon = self.__default_use_epsilon
            if verbose:
                print('EDNN3D: use_epsilon not present in kwargs, using default value:', self.__default_use_epsilon)

        try:
            self.__use_nut = kwargs['use_nut']
        except KeyError:
            self.__use_nut = self.__default_use_nut
            if verbose:
                print('EDNN3D: use_nut not present in kwargs, using default value:', self.__default_use_nut)

        try:
            self.__vae = kwargs['vae']
        except KeyError:
            self.__vae = self.__default_vae
            if verbose:
                print('EDNN3D: vae not present in kwargs, using default value:', self.__default_vae)

        try:
            self.__logvar_scaling = kwargs['logvar_scaling']
        except KeyError:
            self.__logvar_scaling = self.__default_logvar_scaling
            if verbose:
                print('EDNN3D: logvar_scaling not present in kwargs, using default value:', self.__default_logvar_scaling)

        try:
            self.__predict_uncertainty = kwargs['predict_uncertainty']
        except KeyError:
            self.__predict_uncertainty = self.__default_predict_uncertainty
            if verbose:
                print('EDNN3D: predict_uncertainty not present in kwargs, using default value:', self.__default_predict_uncertainty)

        try:
            self.__use_sparse_mask = kwargs['use_sparse_mask']
        except KeyError:
            self.__use_sparse_mask = self.__default_use_sparse_mask
            if verbose:
                print('EDNN3D: use_sparse_mask not present in kwargs, using default value:', self.__default_use_sparse_mask)

        try:
            self.__use_sparse_convolution = kwargs['use_sparse_convolution']
        except KeyError:
            self.__use_sparse_convolution = self.__default_use_sparse_convolution
            if verbose:
                print('EDNN3D: use_sparse_convolution not present in kwargs, using default value:', self.__default_use_sparse_convolution)

        try:
            self.__use_hybrid_model = kwargs['use_hybrid_model']
        except KeyError:
            self.__use_hybrid_model = self.__default_use_hybrid_model
            if verbose:
                print('EDNN3D: use_hybrid_model not present in kwargs, using default value:', self.__default_use_hybrid_model)

        try:
            self.__use_hybrid_model_ext = kwargs['use_hybrid_model_ext']
        except KeyError:
            self.__use_hybrid_model_ext = self.__default_use_hybrid_model_ext
            if verbose:
                print('EDNN3D: use_hybrid_model_ext not present in kwargs, using default value:', self.__default_use_hybrid_model_ext)

        if self.__vae and not self.__use_fc_layers:
            print('EDNN3D: Error, to use the vae mode the fc layers need to be enabled.')
            sys.exit()

        if self.__n_downsample_layers <= 0:
            print('EDNN3D: Error, n_downsample_layers must be larger than 0')
            sys.exit()

        # input variable check
        if (self.__filter_kernel_size % 2 == 0) or (self.__filter_kernel_size < 1):
            raise ValueError('The filter kernel size needs to be odd and larger than 0.')

        # construct the number of input and output channels based on the parameters
        self.__num_inputs = 4  # (terrain, u_x_in, u_y_in, u_z_in)
        if self.__use_sparse_mask:
            self.__num_inputs = 5  # (terrain, u_x_in, u_y_in, u_z_in, sparse_mask)
        self.__num_outputs = 3  # (u_x_out, u_y_out, u_z_out)

        if self.__use_turbulence:
            self.__num_outputs += 1  # turb. kin. en.

        if self.__use_pressure:
            self.__num_outputs += 1  # pressure

        if self.__use_epsilon:
            self.__num_outputs += 1  # dissipation

        if self.__use_nut:
            self.__num_outputs += 1  # viscosity

        try:
            self.__num_inputs = kwargs['force_num_inputs']
        except KeyError:
            pass

        try:
            self.__num_outputs = kwargs['force_num_outputs']
            self.__predict_uncertainty = False
        except KeyError:
            pass

        if self.__use_sparse_convolution:
            # Bias needs to be defined separately in case of sparse convolution
            use_bias = False
        else:
            use_bias = True
        # down-convolution layers
        self.__conv = nn.ModuleList()
        if self.__use_sparse_convolution:
            self.__bias_conv = nn.ParameterList()
        for i in range(self.__n_downsample_layers):
            if i == 0:
                self.__conv += [nn.Conv3d(self.__num_inputs,
                                          self.__n_first_conv_channels,
                                          self.__filter_kernel_size, bias=use_bias)]
                if self.__use_sparse_convolution:
                    self.__bias_conv += [nn.Parameter(torch.zeros(1, self.__n_first_conv_channels, 1, 1, 1).float().cuda(), requires_grad=True)]
            else:
                self.__conv += [nn.Conv3d(int(self.__n_first_conv_channels*(self.__channel_multiplier**(i-1))),
                                          int(self.__n_first_conv_channels*(self.__channel_multiplier**i)),
                                          self.__filter_kernel_size, bias=use_bias)]
                if self.__use_sparse_convolution:
                    self.__bias_conv += [nn.Parameter(torch.zeros(1, int(self.__n_first_conv_channels*(self.__channel_multiplier**i)), 1, 1, 1).float().cuda(), requires_grad=True)]

        # fully connected layers
        if self.__use_fc_layers:
            if self.__n_downsample_layers == 0:
                n_features = int(self.__num_inputs * self.__n_x * self.__n_y * self.__n_z)
            else:
                n_features = int(int(self.__n_first_conv_channels*(self.__channel_multiplier**(self.__n_downsample_layers-1))) *  # number of channels
                                 self.__n_x * self.__n_y * self.__n_z / ((2**self.__n_downsample_layers)**3)) # number of pixels

            if self.__vae:
                self.__vae_dim = int(n_features/self.__fc_scaling)
                if self.__use_hybrid_model:
                    self.__lstm1 = nn.LSTM(n_features, 2 * int(n_features/self.__fc_scaling), num_layers=1, batch_first=True)
                else:
                    self.__fc1 = nn.Linear(n_features, 2 * int(n_features/self.__fc_scaling))
                print("EDNN3D: VAE state space is {} dimensional".format(self.__vae_dim))
            else:
                self.__fc1 = nn.Linear(n_features, int(n_features/self.__fc_scaling))
            if self.__use_hybrid_model:
                self.__lstm2 = nn.LSTM(int(n_features/self.__fc_scaling), n_features, num_layers=1, batch_first=True)
            else:
                self.__fc2 = nn.Linear(int(n_features/self.__fc_scaling), n_features)
        else:
            if self.__use_hybrid_model:
                self.__convLSTM1 = ConvLSTM3d(int(self.__n_first_conv_channels*(self.__channel_multiplier**(self.__n_downsample_layers-1))),
                                             int(self.__n_first_conv_channels*(self.__channel_multiplier**self.__n_downsample_layers)),
                                             self.__filter_kernel_size, num_layers=1, batch_first=True, bias=True, return_all_layers=False)
                self.__convLSTM2 = ConvLSTM3d(int(self.__n_first_conv_channels*(self.__channel_multiplier**self.__n_downsample_layers)),
                                             int(self.__n_first_conv_channels*(self.__channel_multiplier**(self.__n_downsample_layers-1))),
                                             self.__filter_kernel_size, num_layers=1, batch_first=True, bias=True, return_all_layers=False)
            else:
                self.__c1 = nn.Conv3d(int(self.__n_first_conv_channels*(self.__channel_multiplier**(self.__n_downsample_layers-1))),
                                      int(self.__n_first_conv_channels*(self.__channel_multiplier**self.__n_downsample_layers)),
                                      self.__filter_kernel_size, bias=use_bias)

                self.__c2 = nn.Conv3d(int(self.__n_first_conv_channels*(self.__channel_multiplier**self.__n_downsample_layers)),
                                      int(self.__n_first_conv_channels*(self.__channel_multiplier**(self.__n_downsample_layers-1))),
                                      self.__filter_kernel_size, bias=use_bias)
                if self.__use_sparse_convolution:
                    self.__bias_c1 = nn.Parameter(torch.zeros(1, int(self.__n_first_conv_channels * (self.__channel_multiplier ** self.__n_downsample_layers)), 1, 1, 1).float().cuda(), requires_grad=True)
                    self.__bias_c2 = nn.Parameter(torch.zeros(1, int(self.__n_first_conv_channels * (self.__channel_multiplier**(self.__n_downsample_layers-1))), 1, 1, 1).float().cuda(), requires_grad=True)

        # up-convolution layers
        self.__deconv1 = nn.ModuleList()
        self.__deconv2 = nn.ModuleList()

        num_out = self.__num_outputs
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

        if self.__use_hybrid_model_ext:
            self.__ConvLSTM_ext = ConvLSTM3d(self.__num_outputs, self.__num_outputs,
                                             self.__filter_kernel_size, num_layers=1,
                                             batch_first=True, bias=True, return_all_layers=False)

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
        # mask pooling
        if self.__use_sparse_convolution:
            self.__mask_pooling = nn.MaxPool3d(1, stride=2)

    def forward(self, x):
        if self.__use_sparse_mask:
            if self.__use_hybrid_model or self.__use_hybrid_model_ext:
                batch, timesteps, channels, nz, ny, nx = x.shape
                for i in range(timesteps):
                    sparse_mask = x[:, i, -1, :].unsqueeze(1).clone()
                    x[:, i, 1:-1, :] = sparse_mask.repeat(1, self.__num_outputs, 1, 1, 1) * x[:, i, 1:-1, :].clone()
            else:
                sparse_mask = x[:, -1, :].unsqueeze(1).clone()
                x[:, 1:-1, :] = sparse_mask.repeat(1, self.__num_outputs, 1, 1, 1) * x[:, 1:-1, :].clone()

        if self.__use_sparse_convolution:
            # separate mask from inputs and treat it separately
            sparse_mask = x[:, -1, :].unsqueeze(1).clone()
            x = x[:, :-1, :].clone()

        if self.__use_terrain_mask:
            # store the terrain data
            if self.__use_hybrid_model or self.__use_hybrid_model_ext:
                is_wind = x[:, 0, 0, :].unsqueeze(1).clone()
                is_wind.sign_()
            else:
                is_wind = x[:, 0, :].unsqueeze(1).clone()
                is_wind.sign_()

        output = {}
        x_skip = []
        sparse_mask_skip = []
        x_time_steps = []
        if self.__use_hybrid_model_ext:
            time_steps = x.shape[1]
            x_original = x.clone()
        else:
            time_steps = 1
        for t in range(time_steps):
            if self.__use_hybrid_model_ext:
                x = x_original[:, t, :].clone()
            # down-convolution
            if self.__use_hybrid_model:
                batch, timesteps, channels, nz, ny, nx = x.shape
                x_timesteps = []
                x_original = x.clone()
                for i in range(timesteps):
                    for j in range(self.__n_downsample_layers):
                        if j == 0:
                            x = self.__pooling(self.__activation(self.__conv[j](self.__pad_conv(x_original[:, i, :]))))
                        else:
                            x = self.__pooling(self.__activation(self.__conv[j](self.__pad_conv(x))))
                    x_timesteps.append(x)
                x = torch.stack(x_timesteps, dim=1)
            else:
                if (self.__skipping):
                    for i in range(self.__n_downsample_layers):
                        if self.__use_sparse_convolution:  # sparse convolution operation
                            # elementwise multiplication
                            sparse_mask_expanded = sparse_mask.expand_as(x)
                            x = sparse_mask_expanded * x

                            # convolution
                            x = self.__conv[i](self.__pad_conv(x))

                            # normalization
                            weights = torch.ones_like(self.__conv[i].weight)
                            norm = F.conv3d(self.__pad_conv(sparse_mask_expanded), weights, bias=None,
                                            stride=self.__conv[i].stride, padding=self.__conv[i].padding,
                                            dilation=self.__conv[i].dilation)
                            norm = torch.clamp(norm, min=1e-5)
                            norm = 1. / norm
                            x = norm * x

                            # add bias
                            bias = self.__bias_conv[i].expand_as(x)
                            x = x + bias

                            # activation
                            x = self.__activation(x)
                            x_skip.append(x.clone())
                            sparse_mask_skip.append(sparse_mask.expand_as(x).clone())

                            # pooling
                            x = self.__pooling(x)
                            sparse_mask = self.__mask_pooling(sparse_mask)
                        else:
                            x = self.__activation(self.__conv[i](self.__pad_conv(x)))
                            x_skip.append(x.clone())
                            x = self.__pooling(x)

                else:
                    for i in range(self.__n_downsample_layers):
                        if self.__use_sparse_convolution:  # sparse convolution operation
                            # elementwise multiplication
                            sparse_mask_expanded = sparse_mask.expand_as(x)
                            x = sparse_mask_expanded * x

                            # convolution
                            x = self.__conv[i](self.__pad_conv(x))

                            # normalization
                            weights = torch.ones_like(self.__conv[i].weight)
                            norm = F.conv3d(self.__pad_conv(sparse_mask_expanded), weights, bias=None,
                                            stride=self.__conv[i].stride, padding=self.__conv[i].padding,
                                            dilation=self.__conv[i].dilation)
                            norm = torch.clamp(norm, min=1e-5)
                            norm = 1. / norm
                            x = norm * x

                            # add bias
                            bias = self.__bias_conv[i].expand_as(x)
                            x = x + bias

                            # activation
                            x = self.__activation(x)

                            # pooling
                            x = self.__pooling(x)
                            sparse_mask = self.__mask_pooling(sparse_mask)
                        else:
                            x = self.__pooling(self.__activation(self.__conv[i](self.__pad_conv(x))))

            # fully connected layers
            if self.__use_fc_layers:
                if self.__use_hybrid_model:
                    shape = x[:, 0, :].size()
                    x = x.view(x.size()[0], x.size()[1], self.num_flat_features(x))
                    # hidden = (torch.randn(1, x.size()[0], int(x.size()[-1] / 2)).cuda(),
                    #           torch.randn(1, x.size()[0], int(x.size()[-1] / 2)).cuda())
                    out, hidden = self.__lstm1(x)
                    x = self.__activation(hidden[0].squeeze(0))
                    if self.__vae:
                        x_mean = x[:, :self.__vae_dim]
                        x_logvar = x[:, self.__vae_dim:]
                        output['distribution_mean'] = x_mean.clone()
                        output['distribution_logvar'] = x_logvar.clone()

                        # during training sample from the distribution, during inference take values with the maximum probability
                        if self.training:
                            std = torch.torch.exp(0.5 * x_logvar)
                            x = x_mean + std * torch.randn_like(std)
                        else:
                            x = x_mean
                    out, hidden = self.__lstm2(x.unsqueeze(1))
                    output["encoding"] = out.clone()
                    x = self.__activation(out.squeeze(1))
                    x = x.view(shape)
                else:
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

                    output["encoding"] = x.clone()

                    x = self.__activation(self.__fc2(x))
                    x = x.view(shape)
            else:
                if self.__use_sparse_convolution:  # sparse convolution operation
                    # Convolution 1
                    # elementwise multiplication
                    sparse_mask_expanded = sparse_mask.expand_as(x)
                    x = sparse_mask_expanded * x

                    # convolution
                    x = self.__c1(self.__pad_conv(x))

                    # normalization
                    weights = torch.ones_like(self.__c1.weight)
                    norm = F.conv3d(self.__pad_conv(sparse_mask_expanded), weights, bias=None,
                                    stride=self.__c1.stride, padding=self.__c1.padding,
                                    dilation=self.__c1.dilation)
                    norm = torch.clamp(norm, min=1e-5)
                    norm = 1. / norm
                    x = norm * x

                    # add bias
                    bias = self.__bias_c1.expand_as(x)
                    x = x + bias

                    # activation
                    x = self.__activation(x)

                    output["encoding"] = x.view(-1, self.num_flat_features(x)).clone()

                    # Convolution 2
                    # elementwise multiplication
                    sparse_mask_expanded = sparse_mask.expand_as(x)
                    x = sparse_mask_expanded * x

                    # convolution
                    x = self.__c2(self.__pad_conv(x))

                    # normalization
                    weights = torch.ones_like(self.__c2.weight)
                    norm = F.conv3d(self.__pad_conv(sparse_mask_expanded), weights, bias=None,
                                    stride=self.__c2.stride, padding=self.__c2.padding,
                                    dilation=self.__c2.dilation)
                    norm = torch.clamp(norm, min=1e-5)
                    norm = 1. / norm
                    x = norm * x

                    # add bias
                    bias = self.__bias_c2.expand_as(x)
                    x = x + bias

                    # activation
                    x = self.__activation(x)
                elif self.__use_hybrid_model:
                    # ConvLSTM1
                    out, _ = self.__convLSTM1(x)
                    x = out[0]  # output
                    x = self.__activation(x)

                    output["encoding"] = x.view(-1, self.num_flat_features(x)).clone()

                    # ConvLSTM2
                    _, last_states = self.__convLSTM2(x)
                    x = last_states[0][0]  # last hidden state
                    x = self.__activation(x)
                else:
                    x = self.__activation(self.__c1(self.__pad_conv(x)))

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

            if self.__use_hybrid_model_ext:
                # stack output from each timesptes
                x_time_steps.append(x)
        if self.__use_hybrid_model_ext:
            x = torch.stack(x_time_steps, dim=1)
            _, last_states = self.__ConvLSTM_ext(x)
            x = self.__activation(last_states[0][0])

        if self.__use_mapping_layer:
            x = self.__mapping_layer(x)

        if self.__potential_flow:
            x = torch.cat([utils.curl(x, self.__grid_size, x[:, 0, :]), x[:, 3:, :]], 1)

        if self.__use_terrain_mask:
            x = is_wind.repeat(1, x.shape[1], 1, 1, 1) * x

        if self.__predict_uncertainty:
            output['pred'] = x[:,:self.__num_outputs]
            output['logvar'] = self.__logvar_scaling * torch.nn.functional.softsign(x[:,self.__num_outputs:])
        else:
            output['pred'] = x

        return output

    def num_flat_features(self, x):
        if self.__use_hybrid_model:
            size = x.size()[2:]  # all dimensions except the batch dimension and timesteps dimension
        else:
            size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

