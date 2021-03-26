import torch
import torch.nn as nn
import torch.nn.functional as F
from .ModelBase import ModelBase

import nn_wind_prediction.utils as utils

'''
Encoder/Decoder Neural Network

1. Five layers of convolution and max pooling up to 128 channels
2. Two fully connected layers
3. Five times upsampling followed by convolution
4. Mapping layer with a separate filter for each output channel

The first input layer is assumed to be is_wind. This value is true for all cells except the terrain.
'''
class ModelEDNN2D(ModelBase):
    def __init__(self, **kwargs):
        super(ModelEDNN2D, self).__init__()

        parser = utils.KwargsParser(kwargs, 'EDNN2D')
        verbose = parser.get_safe('verbose', False, bool, False)
        self.__use_terrain_mask = parser.get_safe('use_terrain_mask', True, bool, verbose)
        self.__n_downsample_layers = parser.get_safe('n_downsample_layers', 4, int, verbose)
        self.__use_mapping_layer = parser.get_safe('use_mapping_layer', False, bool, verbose)
        self.__use_fc_layers = parser.get_safe('use_fc_layers', True, bool, verbose)
        self.__fc_scaling = parser.get_safe('fc_scaling', 8.0, float, verbose)
        self.__potential_flow = parser.get_safe('potential_flow', False, bool, verbose)
        self.__n_x = parser.get_safe('n_x', 64, int, verbose)
        self.__n_z = parser.get_safe('n_z', 64, int, verbose)
        self.__interpolation_mode = parser.get_safe('interpolation_mode', 'nearest', str, verbose)
        self.__skipping = parser.get_safe('skipping', True, bool, verbose)
        self.__align_corners = parser.get_safe('align_corners', False, bool, verbose)
        self.__pooling_method = parser.get_safe('pooling_method', 'striding', str, verbose)
        self.__use_turbulence = parser.get_safe('use_turbulence', True, bool, verbose)

        if self.__align_corners == False:
            self.__align_corners = None

        self.num_inputs = 3
        self.num_outputs = 2

        if self.__use_turbulence:
            self.num_outputs += 1

        # convolution layers
        self.__conv = nn.ModuleList()
        for i in range(self.__n_downsample_layers):
            if i == 0:
                self.__conv += [nn.Conv2d(self.num_inputs, 8, 3)]
            else:
                self.__conv += [nn.Conv2d(4*2**i, 8*2**i, 3)]

        if self.__use_fc_layers:
            # fully connected layers
            if self.__n_downsample_layers == 0:
                n_features = int(self.num_inputs * self.__n_x * self.__n_z)
            else:
                n_features = int(8*2**(self.__n_downsample_layers-1) * self.__n_x * self.__n_z / ((2**self.__n_downsample_layers)**2))
            self.__fc1 = nn.Linear(n_features, int(n_features/self.__fc_scaling))
            self.__fc2 = nn.Linear(int(n_features/self.__fc_scaling), n_features)

        # modules
        self.__pad_conv = nn.ReplicationPad2d(1)
        self.__pad_deconv = nn.ReplicationPad2d((1, 2, 1, 2))

        # Check if we have defined a specific activation layer
        try:
            activation = getattr(nn, kwargs['activation_type'])
            self.__activation = activation(**kwargs['activation_args'])
        except KeyError as e:
            print('Activation function not specified or not found, using default: {0}'.format(self.__default_activation))
            self.__activation = self.__default_activation(**self.__default_activation_kwargs)

        if (self.__pooling_method == 'averagepool'):
            self.__pooling = nn.AvgPool2d(2)
        elif (self.__pooling_method == 'maxpool'):
            self.__pooling = nn.MaxPool2d(2)
        elif (self.__pooling_method == 'striding'):
            self.__pooling = nn.MaxPool2d(1, stride=2)
        else:
            raise ValueError('The pooling method value is invalid: ' + pooling_method)

        # upconvolution layers
        self.__deconv1 = nn.ModuleList()
        self.__deconv2 = nn.ModuleList()

        if (self.__skipping):
            for i in range(self.__n_downsample_layers):
                if i == 0:
                    self.__deconv1 += [nn.Conv2d(16, 8, 4)]
                    self.__deconv2 += [nn.Conv2d(8, self.num_outputs, 4)]
                else:
                    self.__deconv1 += [nn.Conv2d(16*2**i, 8*2**i, 4)]
                    self.__deconv2 += [nn.Conv2d(8*2**i, 4*2**i, 4)]

        else:
            for i in range(self.__n_downsample_layers):
                if i == 0:
                    self.__deconv1 += [nn.Conv2d(8, self.num_outputs, 4)]
                else:
                    self.__deconv1 += [nn.Conv2d(8*2**i, 4*2**i, 4)]

        if self.__use_mapping_layer:
            # mapping layer
            self.__mapping_layer = nn.Conv2d(self.num_outputs,self.num_outputs,1,groups=self.num_outputs) # for each channel a separate filter

        if self.__potential_flow:
            self.__pf_convolution = nn.Conv2d(2,1,1)
            self.__pf_pad = nn.ReplicationPad2d((0, 1, 0, 1))

    def forward(self, x):
        output = {}

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

            output["encoding"] = x.clone()

            x = self.__activation(self.__fc2(x))
            x = x.view(shape)

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
            potential = self.__pf_convolution(self.__pf_pad(x[:,:2,:]))
            x = torch.cat([(potential[:,:,:-1,1: ]-potential[:,:,:-1,:-1]), # U_x
                           (potential[:,:,1: ,:-1]-potential[:,:,:-1,:-1]), # U_z
                            x[:,2:,:]], 1)

        if self.__use_terrain_mask:
            x = is_wind.repeat(1, self.num_outputs, 1, 1) * x

        output["pred"] = x

        return output

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
