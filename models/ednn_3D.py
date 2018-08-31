import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Encoder/Decoder Neural Network

1. Five layers of convolution and max pooling up to 128 channels
2. Two fully connected layers
3. Five times upsampling followed by convolution
4. Mapping layer with a separate filter for each output channel

The first input layer is assumed to be terrain information. It should be zero in the terrain and nonzero elsewhere.
'''
class ModelEDNN3D(nn.Module):
    def __init__(self, n_input_layers, n_output_layers, n_x, n_y, n_z, n_downsample_layers,
                 interpolation_mode, align_corners, skipping, use_terrain_mask, pooling_method,
                 use_mapping_layer, use_fc_layers, fc_scaling):
        super(ModelEDNN3D, self).__init__()

        self.__num_outputs = n_output_layers
        self.__num_inputs = n_input_layers
        self.__use_terrain_mask = use_terrain_mask
        self.__n_downsample_layers = n_downsample_layers
        self.__use_mapping_layer = use_mapping_layer
        self.__use_fc_layers = use_fc_layers
        self.__fc_scaling = fc_scaling

        self.__n_x = n_x
        self.__n_y = n_y
        self.__n_z = n_z

        # convolution layers
        self.__conv = nn.ModuleList()
        for i in range(n_downsample_layers):
            if i == 0:
                self.__conv += [nn.Conv3d(self.__num_inputs, 8, 3)]
            else:
                self.__conv += [nn.Conv3d(4*2**i, 8*2**i, 3)]

        if self.__use_fc_layers:
            # fully connected layers
            n_features = int(8*2**(n_downsample_layers-1) * n_x * n_y * n_z / ((2**n_downsample_layers)**3))
            self.__fc1 = nn.Linear(n_features, int(n_features/self.__fc_scaling))
            self.__fc2 = nn.Linear(int(n_features/self.__fc_scaling), n_features)

        # modules
        self.__pad = nn.ReplicationPad3d(1)

        self.__leakyrelu = nn.LeakyReLU(0.1)

        if (pooling_method == 'averagepool'):
            self.__pooling = nn.MaxPool3d(2)
        elif (pooling_method == 'maxpool'):
            self.__pooling = nn.MaxPool3d(2)
        elif (pooling_method == 'striding'):
            self.__pooling = nn.MaxPool3d(1, stride=2)
        else:
            raise ValueError('The pooling method value is invalid: ' + pooling_method)

        if (align_corners):
            self.__upsampling = nn.Upsample(scale_factor=2, mode=interpolation_mode, align_corners=True)
        else:
            self.__upsampling = nn.Upsample(scale_factor=2, mode=interpolation_mode)

        # upconvolution layers
        self.__skipping = skipping
        self.__deconv1 = nn.ModuleList()
        self.__deconv2 = nn.ModuleList()

        if (skipping):
            for i in range(n_downsample_layers):
                if i == 0:
                    self.__deconv1 += [nn.Conv3d(16, 8, 3)]
                    self.__deconv2 += [nn.Conv3d(8, self.__num_outputs, 3)]
                else:
                    self.__deconv1 += [nn.Conv3d(16*2**i, 8*2**i, 3)]
                    self.__deconv2 += [nn.Conv3d(8*2**i, 4*2**i, 3)]

        else:
            for i in range(n_downsample_layers):
                if i == 0:
                    self.__deconv1 += [nn.Conv3d(8, self.__num_outputs, 3)]
                else:
                    self.__deconv1 += [nn.Conv3d(8*2**i, 4*2**i, 3)]

        if self.__use_mapping_layer:
            # mapping layer
            self.__mapping_layer = nn.Conv3d(self.__num_outputs,self.__num_outputs,1,groups=self.__num_outputs) # for each channel a separate filter

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
            is_wind = x[:,0, :, :].unsqueeze(1).clone()
            is_wind.sign_()

        x_skip = []
        if (self.__skipping):
            for i in range(self.__n_downsample_layers):
                x = self.__leakyrelu(self.__conv[i](self.__pad(x)))
                x_skip.append(x.clone())
                x = self.__pooling(x)

        else:
            for i in range(self.__n_downsample_layers):
                x = self.__pooling(self.__leakyrelu(self.__conv[i](self.__pad(x))))

        if self.__use_fc_layers:
            shape = x.size()
            x = x.view(-1, self.num_flat_features(x))
            x = self.__leakyrelu(self.__fc1(x))
            x = self.__leakyrelu(self.__fc2(x))
            x = x.view(shape)

        if (self.__skipping):
            for i in range(self.__n_downsample_layers-1, -1, -1):
                x = self.__deconv2[i](self.__pad(self.__deconv1[i](self.__pad(torch.cat([self.__upsampling(x), x_skip[i]], 1)))))
        else:
            for i in range(self.__n_downsample_layers-1, -1, -1):
                x = self.__deconv1[i](self.__pad(self.__upsampling(x)))

        if self.__use_mapping_layer:
            x = self.__mapping_layer(x)

        if self.__use_terrain_mask:
            x = is_wind.repeat(1, self.__num_outputs, 1, 1, 1) * x

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

