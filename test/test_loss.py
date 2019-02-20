import nn_wind_prediction.nn as nn
import time
import torch

# test the log likelihood loss
loss_fn = nn.GaussianLogLikelihoodLoss()

label = torch.randn(32,4,64,64,64, requires_grad=True)
output = torch.randn(32,8,64,64,64, requires_grad=True)

start_time = time.time()
loss = loss_fn(output, label)
print('GLLL: Forward took', (time.time() - start_time), 'seconds')

start_time = time.time()
loss.backward()
print('GLLL: Backward took', (time.time() - start_time), 'seconds')