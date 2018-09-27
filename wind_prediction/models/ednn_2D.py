import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Encoder/Decoder Neural Network

1. Five layers of convolution and max pooling up to 128 channels
2. Two fully connected layers
3. Five times upsampling followed by convolution
4. Mapping layer with a separate filter for each output channel

The first input layer is assumed to be is_wind. This value is true for all cells except the terrain.
'''
class ModelEDNN2D(nn.Module):
    def __init__(self, n_input_layers, interpolation_mode, align_corners, skipping, predict_turbulence):
        super(ModelEDNN2D, self).__init__()

        if predict_turbulence:
            self.__num_outputs = 3
        else:
            self.__num_outputs = 2

        self.conv1 = nn.Conv2d(n_input_layers, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.conv5 = nn.Conv2d(64, 128, 3)

        self.pad_conv = nn.ReplicationPad2d(1)
        self.pad_deconv = nn.ReplicationPad2d((1,2,1,2))

        self.leakyrelu = nn.LeakyReLU(0.1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 1024)

        if (align_corners):
            self.upsampling = nn.Upsample(scale_factor=2, mode=interpolation_mode, align_corners=True)
        else:
            self.upsampling = nn.Upsample(scale_factor=2, mode=interpolation_mode)

        self.skipping = skipping
        if (skipping):
            self.deconv51 = nn.Conv2d(256, 128, 4)
            self.deconv52 = nn.Conv2d(128, 64, 4)
            self.deconv41 = nn.Conv2d(128, 64, 4)
            self.deconv42 = nn.Conv2d(64, 32, 4)
            self.deconv31 = nn.Conv2d(64, 32, 4)
            self.deconv32 = nn.Conv2d(32, 16, 4)
            self.deconv21 = nn.Conv2d(32, 16, 4)
            self.deconv22 = nn.Conv2d(16, 8, 4)
            self.deconv11 = nn.Conv2d(16, 8, 4)
            self.deconv12 = nn.Conv2d(8, self.__num_outputs, 4)
        else:
            self.deconv5 = nn.Conv2d(128, 64, 4)
            self.deconv4 = nn.Conv2d(64, 32, 4)
            self.deconv3 = nn.Conv2d(32, 16, 4)
            self.deconv2 = nn.Conv2d(16, 8, 4)
            self.deconv1 = nn.Conv2d(8, self.__num_outputs, 4)

        self.mapping_layer = nn.Conv2d(self.__num_outputs,self.__num_outputs,1,groups=self.__num_outputs)

    def init_params(self):
        def printf(m):
            if (type(m) != type(self)):
                try:
                    torch.nn.init.xavier_normal_(m.weight.data)
                except:
                    pass
                try:
                    torch.nn.init.normal_(m.bias, mean = 0.0, std = 0.02)
                except:
                    pass

        self.apply(printf)

    def forward(self, x):
        # store the terrain data
        is_wind = x[:,0, :, :].unsqueeze(1).clone()
        is_wind.sign_()

        if (self.skipping):
            x = self.leakyrelu(self.conv1(self.pad_conv(x)))
            x1 = x.clone()
            x = F.max_pool2d(x, 2)
            x = self.leakyrelu(self.conv2(self.pad_conv(x)))
            x2 = x.clone()
            x = F.max_pool2d(x, 2)
            x = self.leakyrelu(self.conv3(self.pad_conv(x)))
            x3 = x.clone()
            x = F.max_pool2d(x, 2)
            x = self.leakyrelu(self.conv4(self.pad_conv(x)))
            x4 = x.clone()
            x = F.max_pool2d(x, 2)
            x = self.leakyrelu(self.conv5(self.pad_conv(x)))
            x5 = x.clone()
            x = F.max_pool2d(x, 2)
        else:
            x = F.max_pool2d(self.leakyrelu(self.conv1(self.pad_conv(x))), 2)
            x = F.max_pool2d(self.leakyrelu(self.conv2(self.pad_conv(x))), 2)
            x = F.max_pool2d(self.leakyrelu(self.conv3(self.pad_conv(x))), 2)
            x = F.max_pool2d(self.leakyrelu(self.conv4(self.pad_conv(x))), 2)
            x = F.max_pool2d(self.leakyrelu(self.conv5(self.pad_conv(x))), 2)

        shape = x.size()
        x = x.view(-1, self.num_flat_features(x))
        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = x.view(shape)

        if (self.skipping):
            x = self.deconv52(self.pad_deconv(self.deconv51(self.pad_deconv(torch.cat([x5, self.upsampling(x)], 1)))))
            x = self.deconv42(self.pad_deconv(self.deconv41(self.pad_deconv(torch.cat([x4, self.upsampling(x)], 1)))))
            x = self.deconv32(self.pad_deconv(self.deconv31(self.pad_deconv(torch.cat([x3, self.upsampling(x)], 1)))))
            x = self.deconv22(self.pad_deconv(self.deconv21(self.pad_deconv(torch.cat([x2, self.upsampling(x)], 1)))))
            x = self.deconv12(self.pad_deconv(self.deconv11(self.pad_deconv(torch.cat([x1, self.upsampling(x)], 1)))))
        else:
            x = self.deconv5(self.pad_deconv(self.upsampling(x)))
            x = self.deconv4(self.pad_deconv(self.upsampling(x)))
            x = self.deconv3(self.pad_deconv(self.upsampling(x)))
            x = self.deconv2(self.pad_deconv(self.upsampling(x)))
            x = self.deconv1(self.pad_deconv(self.upsampling(x)))

        x = self.mapping_layer(x)

        # multiply the outputs by the terrain boolean
        if (self.__num_outputs == 2):
            x = torch.cat([is_wind, is_wind], 1) * x
        else:
            x = torch.cat([is_wind, is_wind, is_wind], 1) * x

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
