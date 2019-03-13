import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import nn_wind_prediction.utils as utils

'''
Encoder/Decoder Neural Network

1. Five layers of convolution and max pooling up to 128 channels
2. Two fully connected layers
3. Five times upsampling followed by convolution
4. Mapping layer with a separate filter for each output channel

The first input layer is assumed to be terrain information. It should be zero in the terrain and nonzero elsewhere.
'''
class ModelEDNN3D(nn.Module):
    __default_activation = nn.LeakyReLU
    __default_activation_kwargs = {'negative_slope': 0.1}
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
    __default_use_grid_size = False
    __default_use_turbulence = True

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
            self.__use_grid_size = kwargs['use_grid_size']
        except KeyError:
            self.__use_grid_size = self.__default_use_grid_size
            if verbose:
                print('EDNN3D: use_grid_size not present in kwargs, using default value:', self.__default_use_grid_size)

        try:
            self.__use_turbulence = kwargs['use_turbulence']
        except KeyError:
            self.__use_turbulence = self.__default_use_turbulence
            if verbose:
                print('EDNN3D: use_turbulence not present in kwargs, using default value:', self.__default_use_turbulence)

        if self.__n_downsample_layers <= 0:
            print('EDNN3D: Error, n_downsample_layers must be larger than 0')
            sys.exit()

        # construct the number of input and output channels based on the parameters
        self.__num_inputs = 4 # (terrain, u_x_in, u_y_in, u_z_in)
        self.__num_outputs = 3 # (u_x_out, u_y_out, u_z_out)

        if self.__use_grid_size:
            self.__num_inputs += 3 # x, y, z

        if self.__use_turbulence:
            self.__num_outputs += 1 # turbulence

        try:
            self.__num_inputs = kwargs['force_num_inputs']
        except KeyError:
            pass

        try:
            self.__num_outputs = kwargs['force_num_outputs']
        except KeyError:
            pass

        # convolution layers
        self.__conv = nn.ModuleList()
        for i in range(self.__n_downsample_layers):
            if i == 0:
                self.__conv += [nn.Conv3d(self.__num_inputs, 8, 3)]
            else:
                self.__conv += [nn.Conv3d(4*2**i, 8*2**i, 3)]

        if self.__use_fc_layers:
            # fully connected layers
            if self.__n_downsample_layers == 0:
                n_features = int(self.__num_inputs * self.__n_x * self.__n_y * self.__n_z)
            else:
                n_features = int(8*2**(self.__n_downsample_layers-1) * self.__n_x * self.__n_y * self.__n_z / ((2**self.__n_downsample_layers)**3))
            self.__fc1 = nn.Linear(n_features, int(n_features/self.__fc_scaling))
            self.__fc2 = nn.Linear(int(n_features/self.__fc_scaling), n_features)

        # modules
        self.__pad_conv = nn.ReplicationPad3d(1)
        self.__pad_deconv = nn.ReplicationPad3d((1, 2, 1, 2, 1, 2))

        # Check if we have defined a specific activation layer
        try:
            activation = getattr(nn, kwargs['activation_type'])
            self.__activation = activation(**kwargs['activation_args'])
        except KeyError as e:
            print('Activation function not specified or not found, using default: {0}'.format(self.__default_activation))
            self.__activation = self.__default_activation(**self.__default_activation_kwargs)

        if (self.__pooling_method == 'averagepool'):
            self.__pooling = nn.AvgPool3d(2)
        elif (self.__pooling_method == 'maxpool'):
            self.__pooling = nn.MaxPool3d(2)
        elif (self.__pooling_method == 'striding'):
            self.__pooling = nn.MaxPool3d(1, stride=2)
        else:
            raise ValueError('The pooling method value is invalid: ' + pooling_method)

        # upconvolution layers
        self.__deconv1 = nn.ModuleList()
        self.__deconv2 = nn.ModuleList()

        if (self.__skipping):
            for i in range(self.__n_downsample_layers):
                if i == 0:
                    self.__deconv1 += [nn.Conv3d(16, 8, 4)]
                    self.__deconv2 += [nn.Conv3d(8, self.__num_outputs, 4)]
                else:
                    self.__deconv1 += [nn.Conv3d(16*2**i, 8*2**i, 4)]
                    self.__deconv2 += [nn.Conv3d(8*2**i, 4*2**i, 4)]

        else:
            for i in range(self.__n_downsample_layers):
                if i == 0:
                    self.__deconv1 += [nn.Conv3d(8, self.__num_outputs, 4)]
                else:
                    self.__deconv1 += [nn.Conv3d(8*2**i, 4*2**i, 4)]

        if self.__use_mapping_layer:
            # mapping layer
            self.__mapping_layer = nn.Conv3d(self.__num_outputs,self.__num_outputs,1,groups=self.__num_outputs) # for each channel a separate filter

        #if self.__potential_flow:
        #   self.__pf_convolution = nn.Conv3d(3,1,1)
        #  self.__pf_pad = nn.ReplicationPad3d((0, 1, 0, 1, 0, 1))

    def new_epoch_callback(self, epoch):
        # nothing to do here
        return

    def freeze_model(self):
        def freeze_weights(m):
            for params in m.parameters():
                params.requires_grad = False

        self.apply(freeze_weights)

    def unfreeze_model(self):
        def unfreeze_weights(m):
            for params in m.parameters():
                params.requires_grad = True

        self.apply(unfreeze_weights)

    def num_inputs(self):
        return self.__num_inputs

    def num_outputs(self):
        return self.__num_outputs

    def init_params(self):
        def init_weights(m):
            if (type(m) != type(self)):
                try:
                    torch.nn.init.xavier_normal_(m.weight.data)
                except:
                    pass
                try:
                    torch.nn.init.normal_(m.bias, mean = 0.0, std = 0.02)
                except:
                    pass

        self.apply(init_weights)

    def forward(self, x):
        if self.__use_terrain_mask:
            # store the terrain data
            is_wind = x[:,0, :].unsqueeze(1).clone()
            is_wind.sign_()

        x_skip = []
        if (self.__skipping):
            for i in range(self.__n_downsample_layers):
                x = self.__activation(self.__conv[i](self.__pad_conv(x)))
                x_skip.append(x.clone())
                x = self.__pooling(x)

        else:
            for i in range(self.__n_downsample_layers):
                x = self.__pooling(self.__activation(self.__conv[i](self.__pad_conv(x))))

        if self.__use_fc_layers:
            shape = x.size()
            x = x.view(-1, self.num_flat_features(x))
            x = self.__activation(self.__fc1(x))
            x = self.__activation(self.__fc2(x))
            x = x.view(shape)

        if (self.__skipping):
            for i in range(self.__n_downsample_layers-1, -1, -1):
                x = self.__deconv2[i](self.__pad_deconv(self.__deconv1[i](self.__pad_deconv(
                    torch.cat([F.interpolate(x, scale_factor=2, mode=self.__interpolation_mode, align_corners=self.__align_corners),
                               x_skip[i]], 1)))))
        else:
            for i in range(self.__n_downsample_layers-1, -1, -1):
                x = self.__deconv1[i](self.__pad_deconv(
                    F.interpolate(x, scale_factor=2, mode=self.__interpolation_mode, align_corners=self.__align_corners)))

        if self.__use_mapping_layer:
            x = self.__mapping_layer(x)

        if self.__potential_flow:
            # potential = self.__pf_convolution(self.__pf_pad(x[:,:3,:]))
            # x = torch.cat([(potential[:,:,:-1,:-1,1: ]-potential[:,:,:-1,:-1,:-1]), # U_x
            #                (potential[:,:,:-1,1: ,:-1]-potential[:,:,:-1,:-1,:-1]), # U_y
            #                (potential[:,:,1: ,:-1,:-1]-potential[:,:,:-1,:-1,:-1]), # U_z
            #                 x[:,3:,:]], 1)
            x = torch.cat([utils.curl(x, ds=1), x[:, 3:, :]], 1)

        if self.__use_terrain_mask:
            x = is_wind.repeat(1, self.__num_outputs, 1, 1, 1) * x

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

