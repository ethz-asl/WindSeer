import models
import torch

input = torch.randn(1, 3, 128, 64)

net = models.ModelEDNN2D(3)

output = net(input)

print(output)