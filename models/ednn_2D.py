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

    def __init__(self, n_input_layers, interpolation_mode, align_corners):
        super(ModelEDNN2D, self).__init__()

        self.conv1 = nn.Conv2d(n_input_layers, 8, 3, padding = 1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding = 1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding = 1)
        self.conv4 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding = 1)
        
        self.leakyrelu = nn.LeakyReLU(0.1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 1024)

        self.upsampling = nn.Upsample(scale_factor=2, mode=interpolation_mode, align_corners=align_corners)
        
        self.deconv5 = nn.Conv2d(128, 64, 3, padding = 1)
        self.deconv4 = nn.Conv2d(64, 32, 3, padding = 1)
        self.deconv3 = nn.Conv2d(32, 16, 3, padding = 1)
        self.deconv2 = nn.Conv2d(16, 8, 3, padding = 1)
        self.deconv1 = nn.Conv2d(8, 3, 3, padding = 1)
        
#         self.deconv5 = nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=1, padding = 1)
#         self.deconv4 = nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1, padding = 1)
#         self.deconv3 = nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1, padding = 1)
#         self.deconv2 = nn.ConvTranspose2d(16, 8, 3, stride=2, output_padding=1, padding = 1)
#         self.deconv1 = nn.ConvTranspose2d(8, 3, 3, stride=2, output_padding=1, padding = 1)
        
        self.mapping_layer = nn.Conv2d(3,3,1,groups=3) # for each channel a separate filter

    def forward(self, x):
        # store the terrain data
        is_wind = x[:,0, :, :].unsqueeze(1).clone()

        x = F.max_pool2d(self.leakyrelu(self.conv1(x)), 2)
        x = F.max_pool2d(self.leakyrelu(self.conv2(x)), 2)
        x = F.max_pool2d(self.leakyrelu(self.conv3(x)), 2)
        x = F.max_pool2d(self.leakyrelu(self.conv4(x)), 2)
        x = F.max_pool2d(self.leakyrelu(self.conv5(x)), 2)

        shape = x.size()
        x = x.view(-1, self.num_flat_features(x))
        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = x.view(shape)

        x = self.deconv5(self.upsampling(x))
        x = self.deconv4(self.upsampling(x))
        x = self.deconv3(self.upsampling(x))
        x = self.deconv2(self.upsampling(x))
        x = self.deconv1(self.upsampling(x))

#         x = self.deconv5(x)
#         x = self.deconv4(x)
#         x = self.deconv3(x)
#         x = self.deconv2(x)
#         x = self.deconv1(x)
        
        x = self.mapping_layer(x)

        # multiply the outputs by the terrain boolean
        x = torch.cat([is_wind, is_wind, is_wind], 1) * x

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    