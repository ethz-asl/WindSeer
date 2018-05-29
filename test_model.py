import models
import torch

input = torch.randn(1, 3, 128, 64)

net = models.ModelEDNN2D(3, 'bilinear', True)

output = net(input)

print('Input of', input.shape, ' generates an output of', output.shape)
