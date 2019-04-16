import nn_wind_prediction.nn as nn
import time
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# test the log likelihood loss
loss_fn = nn.GaussianLogLikelihoodLoss()

label = torch.randn(20,4,64,64,64, requires_grad=False)
output = torch.randn(20,8,64,64,64, requires_grad=True)

label, output = label.to(device), output.to(device)

start_time = time.time()
loss = loss_fn(output, label)
print('GLLL: Forward took', (time.time() - start_time), 'seconds')

start_time = time.time()
loss.backward()
print('GLLL: Backward took', (time.time() - start_time), 'seconds')


# test the custom loss
input = torch.randn(32,4,64,64,64, requires_grad=False)
label = torch.randn(32,4,64,64,64, requires_grad=False)
output = torch.randn(32,4,64,64,64, requires_grad=True)
input[:,0,:10,:,:] = 0.0 # generate some terrain

label, output, input = label.to(device), output.to(device), input.to(device)

my_loss = nn.ScaledLoss(exclude_terrain = True)
start_time = time.time()
loss = my_loss(output, label, input)
print('ScaledLoss exclude terrain: Forward took', (time.time() - start_time), 'seconds')

start_time = time.time()
loss.backward()
print('ScaledLoss exclude terrain: Backward took', (time.time() - start_time), 'seconds')

my_loss = nn.ScaledLoss(exclude_terrain = False)
start_time = time.time()
loss = my_loss(output, label, input)
print('ScaledLoss include terrain: Forward took', (time.time() - start_time), 'seconds')

start_time = time.time()
loss.backward()
print('ScaledLoss include terrain: Backward took', (time.time() - start_time), 'seconds')

my_loss = nn.ScaledLoss(exclude_terrain = True, no_scaling = True)
start_time = time.time()
loss = my_loss(output, label, input)
print('ScaledLoss mse no terrain: Forward took', (time.time() - start_time), 'seconds')

start_time = time.time()
loss.backward()
print('ScaledLoss mse no terrain: Backward took', (time.time() - start_time), 'seconds')
