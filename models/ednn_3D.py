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
class ModelEDNN3D(nn.Module):
    def __init__(self, n_input_layers, interpolation_mode, align_corners, skipping, high_resolution):
        super(ModelEDNN3D, self).__init__()

        self.conv1 = nn.Conv3d(n_input_layers, 8, 3, padding = 1)
        self.conv2 = nn.Conv3d(8, 16, 3, padding = 1)
        self.conv3 = nn.Conv3d(16, 32, 3, padding = 1)
        self.conv4 = nn.Conv3d(32, 64, 3, padding = 1)
        self.conv5 = nn.Conv3d(64, 128, 3, padding = 1)

        self.leakyrelu = nn.LeakyReLU(0.1)

        if (high_resolution):
            self.fc1 = nn.Linear(8192, 4096)
            self.fc2 = nn.Linear(4096, 8192)
        else:
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 1024)

        if (align_corners):
            self.upsampling = nn.Upsample(scale_factor=2, mode=interpolation_mode, align_corners=True)
        else:
            self.upsampling = nn.Upsample(scale_factor=2, mode=interpolation_mode)

        self.skipping = skipping
        if (skipping):
            self.deconv51 = nn.Conv3d(256, 128, 3, padding = 1)
            self.deconv52 = nn.Conv3d(128, 64, 3, padding = 1)
            self.deconv41 = nn.Conv3d(128, 64, 3, padding = 1)
            self.deconv42 = nn.Conv3d(64, 32, 3, padding = 1)
            self.deconv31 = nn.Conv3d(64, 32, 3, padding = 1)
            self.deconv32 = nn.Conv3d(32, 16, 3, padding = 1)
            self.deconv21 = nn.Conv3d(32, 16, 3, padding = 1)
            self.deconv22 = nn.Conv3d(16, 8, 3, padding = 1)
            self.deconv11 = nn.Conv3d(16, 8, 3, padding = 1)
            self.deconv12 = nn.Conv3d(8, 2, 3, padding = 1)
        else:
            self.deconv5 = nn.Conv3d(128, 64, 3, padding = 1)
            self.deconv4 = nn.Conv3d(64, 32, 3, padding = 1)
            self.deconv3 = nn.Conv3d(32, 16, 3, padding = 1)
            self.deconv2 = nn.Conv3d(16, 8, 3, padding = 1)
            self.deconv1 = nn.Conv3d(8, 2, 3, padding = 1)

        self.mapping_layer = nn.Conv3d(2,2,1,groups=2) # for each channel a separate filter

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
            x = F.max_pool3d(self.leakyrelu(self.conv1(x)), 2)
            x1 = x.clone()
            x = F.max_pool3d(self.leakyrelu(self.conv2(x)), 2)
            x2 = x.clone()
            x = F.max_pool3d(self.leakyrelu(self.conv3(x)), 2)
            x3 = x.clone()
            x = F.max_pool3d(self.leakyrelu(self.conv4(x)), 2)
            x4 = x.clone()
            x = F.max_pool3d(self.leakyrelu(self.conv5(x)), 2)
            x5 = x.clone()
        else:
            x = F.max_pool3d(self.leakyrelu(self.conv1(x)), 2)
            x = F.max_pool3d(self.leakyrelu(self.conv2(x)), 2)
            x = F.max_pool3d(self.leakyrelu(self.conv3(x)), 2)
            x = F.max_pool3d(self.leakyrelu(self.conv4(x)), 2)
            x = F.max_pool3d(self.leakyrelu(self.conv5(x)), 2)

        shape = x.size()
        x = x.view(-1, self.num_flat_features(x))
        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = x.view(shape)

        if (self.skipping):
            x = self.deconv52(self.deconv51(self.upsampling(torch.cat([x5, x], 1))))
            x = self.deconv42(self.deconv41(self.upsampling(torch.cat([x4, x], 1))))
            x = self.deconv32(self.deconv31(self.upsampling(torch.cat([x3, x], 1))))
            x = self.deconv22(self.deconv21(self.upsampling(torch.cat([x2, x], 1))))
            x = self.deconv12(self.deconv11(self.upsampling(torch.cat([x1, x], 1))))
        else:
            x = self.deconv5(self.upsampling(x))
            x = self.deconv4(self.upsampling(x))
            x = self.deconv3(self.upsampling(x))
            x = self.deconv2(self.upsampling(x))
            x = self.deconv1(self.upsampling(x))

        x = self.mapping_layer(x)

        # multiply the outputs by the terrain boolean
        #x = torch.cat([is_wind, is_wind, is_wind], 1) * x
        x = torch.cat([is_wind, is_wind], 1) * x

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    