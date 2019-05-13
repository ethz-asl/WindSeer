import nn_wind_prediction.nn as nn_custom
import time
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\t\t\t\t\t\tTESTING LOSS FUNCTIONS')
print('Using device: ', device)
print('------------------------------------------------------------------')

#test the GLLLoss first
label = torch.randn(20,4,64,64,64, requires_grad=False)
output = torch.randn(20,8,64,64,64, requires_grad=True)
input = torch.randn(20,4,64,64,64, requires_grad=False)
input[:,0,:10,:,:] = 0.0 # generate some terrain

label, output, input = label.to(device), output.to(device), input.to(device)

# add GLLloss configs to test here
GLLconfigs = []
GLLconfigs.append({'Loss': 'GaussianLogLikelihoodLoss', 'loss_kwargs':{'exclude_terrain': True} })
GLLconfigs.append({'Loss': 'GaussianLogLikelihoodLoss', 'loss_kwargs':{'exclude_terrain': False} })

# GLLL testing for loop
for i, config in enumerate(GLLconfigs):
    print('\t\t\t\t', 'Test {}: '.format(i), config['Loss'], '\n')
    print('[{}] '.format(i), 'loss kwargs: ', config['loss_kwargs'])
    loss_fn = getattr(nn_custom, config['Loss'])
    loss_fn = loss_fn(**config['loss_kwargs'])
    start_time = time.time()
    loss = loss_fn(output, label, input)
    print('[{}] '.format(i),'Forward took', (time.time() - start_time), 'seconds')
    print('[{}] '.format(i), 'Computed Loss:', loss.item())

    start_time = time.time()
    loss.backward()
    print('[{}] '.format(i),'Backward took', (time.time() - start_time), 'seconds')
    print('------------------------------------------------------------------')


# test the custom losses
input = torch.rand(32,4,64,64,64, requires_grad=False)
label = torch.rand(32,4,64,64,64, requires_grad=False)
output = torch.rand(32,4,64,64,64, requires_grad=True)
input[:,0,:10,:,:] = 0.0 # generate some terrain

label, output, input = label.to(device), output.to(device), input.to(device)

# add loss configs to test here
configs = []
configs.append({'Loss': 'ScaledLoss', 'loss_kwargs':{'exclude_terrain': True, 'no_scaling': True} })
configs.append({'Loss': 'ScaledLoss','loss_kwargs':{'exclude_terrain': True, 'no_scaling': False }})
configs.append({'Loss': 'ScaledLoss','loss_kwargs':{'exclude_terrain': False, 'no_scaling': True }})
configs.append({'Loss': 'ScaledLoss','loss_kwargs':{'exclude_terrain': False, 'no_scaling': False }})
configs.append({'Loss': 'MSELoss','loss_kwargs':{'exclude_terrain': True}})
configs.append({'Loss': 'MSELoss','loss_kwargs':{'exclude_terrain': False}})
configs.append({'Loss': 'L1Loss','loss_kwargs':{'exclude_terrain': True}})
configs.append({'Loss': 'L1Loss','loss_kwargs':{'exclude_terrain': False}})
configs.append({'Loss': 'DivergenceFreeLoss','loss_kwargs':{'loss_method': 'L1','exclude_terrain': True }})
configs.append({'Loss': 'DivergenceFreeLoss','loss_kwargs':{'loss_method': 'L1','exclude_terrain': False}})
configs.append({'Loss': 'DivergenceFreeLoss','loss_kwargs':{'loss_method': 'MSE','exclude_terrain': True}})
configs.append({'Loss': 'DivergenceFreeLoss','loss_kwargs':{'loss_method': 'MSE','exclude_terrain': False}})
configs.append({'Loss': 'VelocityGradientLoss','loss_kwargs':{'loss_method': 'L1','exclude_terrain': True}})
configs.append({'Loss': 'VelocityGradientLoss','loss_kwargs':{'loss_method': 'L1','exclude_terrain': False}})
configs.append({'Loss': 'VelocityGradientLoss','loss_kwargs':{'loss_method': 'MSE','exclude_terrain': True}})
configs.append({'Loss': 'VelocityGradientLoss','loss_kwargs':{'loss_method': 'MSE','exclude_terrain': False}})


# custom loss testing for loop
for k, config in enumerate(configs):
    k+= i+1
    print('\t\t\t\t', 'Test {}: '.format(k), config['Loss'], '\n')
    print('[{}] '.format(k),'loss kwargs: ', config['loss_kwargs'])
    loss_fn = getattr(nn_custom, config['Loss'])
    loss_fn = loss_fn(**config['loss_kwargs'])
    start_time = time.time()
    loss = loss_fn(output, label, input)
    print('[{}] '.format(k),'Forward took', (time.time() - start_time), 'seconds')
    print('[{}] '.format(k), 'Computed Loss:', loss.item())

    start_time = time.time()
    loss.backward()
    print('[{}] '.format(k),'Backward took', (time.time() - start_time), 'seconds')
    print('------------------------------------------------------------------')

