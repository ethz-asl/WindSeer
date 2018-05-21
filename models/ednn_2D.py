import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelEDNN2D(nn.Module):

    def __init__(self, n_input_layers):
        super(ModelEDNN2D, self).__init__()

        self.conv1 = nn.Conv2d(n_input_layers, 8, 3, padding = 1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding = 1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding = 1)
        self.conv4 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding = 1)
        
        self.leakyrelu = nn.LeakyReLU(0.1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 1024)
        
        self.deconv5 = nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=1, padding = 1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1, padding = 1)
        self.deconv3 = nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1, padding = 1)
        self.deconv2 = nn.ConvTranspose2d(16, 8, 3, stride=2, output_padding=1, padding = 1)
        self.deconv1 = nn.ConvTranspose2d(8, 3, 3, stride=2, output_padding=1, padding = 1)
        
        self.mapping_layer = nn.Conv2d(3,3,1,groups=3) # for each channel a separate filter

    def forward(self, x):
        x = F.max_pool2d(self.leakyrelu(self.conv1(x)), 2)
        x = F.max_pool2d(self.leakyrelu(self.conv2(x)), 2)
        x = F.max_pool2d(self.leakyrelu(self.conv3(x)), 2)
        x = F.max_pool2d(self.leakyrelu(self.conv4(x)), 2)
        x = F.max_pool2d(self.leakyrelu(self.conv5(x)), 2)

        shape = x.size()
        x = x.view(-1, self.num_flat_features(x))
        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = x.view(shape) #TODO fix here right dimension
        
        x = self.deconv5(x)
        x = self.deconv4(x)
        x = self.deconv3(x)
        x = self.deconv2(x)
        x = self.deconv1(x)
        
        x = self.mapping_layer(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    