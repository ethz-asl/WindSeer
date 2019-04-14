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

label, output, input = label.to(device), output.to(device), input.to(device)

my_loss = nn.ScaledLoss()
start_time = time.time()
loss = my_loss(output, label, input)
print('ScaledLoss: Forward took', (time.time() - start_time), 'seconds')

start_time = time.time()
loss.backward()
print('ScaledLoss: Backward took', (time.time() - start_time), 'seconds')