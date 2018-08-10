#!/usr/bin/env python

import models
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import utils

# ---- Params --------------------------------------------------------------
# learning parameters
learning_rate = 1e-3
plot_every_n_batches = 10
n_epochs = 20
batchsize = 32
num_workers = 8

# options to store data
save_model = True
save_learning_curve = True
evaluate_testset = False
warm_start = False
custom_loss = False

# dataset parameter
trainset_name = 'data/converted_3d.tar'
validationset_name = 'data/converted_3d.tar'
testset_name = 'data/converted_3d.tar'
stride_hor = 2
stride_vert = 1
uhor_scaling = 6.0
uz_scaling = 2.5
turbulence_scaling = 4.5

# model parameter
d3 = True
model_name = 'ednn_3D_scaled_nearest_skipping_boolean'
n_input_layers = 4
n_output_layers = 3
n_x = 64
n_y = 64
n_z = 64
n_downsample_layers = 5
interpolation_mode = 'nearest'
align_corners = False
skipping = True
use_terrain_mask = True
pooling_method = 'maxpool'
# --------------------------------------------------------------------------

if d3:
    if n_output_layers > 3:
        use_turbulence = True
    else:
        use_turbulence = False
else:
    if n_output_layers > 2:
        use_turbulence = True
    else:
        use_turbulence = False

# define dataset and dataloader
trainset = utils.MyDataset(trainset_name, stride_hor = stride_hor, stride_vert = stride_vert, turbulence_label = use_turbulence, scaling_uhor = uhor_scaling, scaling_uz = uz_scaling, scaling_nut = turbulence_scaling)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                          shuffle=True, num_workers=num_workers)

validationset = utils.MyDataset(validationset_name, stride_hor = stride_hor, stride_vert = stride_vert, turbulence_label = use_turbulence, scaling_uhor = uhor_scaling, scaling_uz = uz_scaling, scaling_nut = turbulence_scaling)

validationloader = torch.utils.data.DataLoader(validationset, batch_size=1,
                                          shuffle=False, num_workers=num_workers)

#check if gpu is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define model and move to gpu if available
if d3:
    net = models.ModelEDNN3D(n_input_layers, n_output_layers, n_x, n_y, n_z, n_downsample_layers, interpolation_mode, align_corners, skipping, use_terrain_mask, pooling_method)
else:
    net = models.ModelEDNN2D(n_input_layers, interpolation_mode = interpolation_mode, align_corners = align_corners, skipping = skipping, predict_turbulence = use_turbulence)

if (warm_start):
    try:
        net.load_state_dict(torch.load('models/trained_models/' + model_name + '.model', map_location=lambda storage, loc: storage))
    except:
        print('Warning: Failed to load the model parameter, initializing parameter.')
        net.init_params()
else:
    net.init_params()

net.to(device)

# define optimizer and objective
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

if custom_loss:
    loss_fn = utils.MyLoss(device)
else:
    loss_fn = torch.nn.MSELoss()

# initialize variable to store the learning curve
learning_curve = np.zeros([n_epochs, 2])

print('-----------------------------------------------------------------------------')
print('INFO: Start training on device %s' % device)
print(' ')
print('Train Settings:')
print('\tWarm start:\t\t', warm_start)
print('\tLearning rate:\t\t', learning_rate)
print('\tBatchsize:\t\t', batchsize)
print('\tEpochs:\t\t\t', n_epochs)
print(' ')
print('Model Settings:')
print('\tModel name:\t\t', model_name)
print('\t3D:\t\t\t', d3)
print('\tNumber of inputs:\t', n_input_layers)
print('\tNumber of outputs:\t', n_output_layers)
print('\tNx:\t\t\t', n_x)
print('\tNy:\t\t\t', n_y)
print('\tNz:\t\t\t', n_z)
print('\tNumber conv layers:\t', n_downsample_layers)
print('\tInterpolation mode:\t', interpolation_mode)
print('\tAlign corners:\t\t', align_corners)
print('\tSkip connection:\t', skipping)
print('\tUse terrain mask:\t', use_terrain_mask)
print('\tPooling method:\t\t', pooling_method)
print(' ')
print('Dataset Settings:')
print('\tUhor scaling:\t\t', uhor_scaling)
print('\tUz scaling:\t\t', uz_scaling)
print('\tTurbulence scaling:\t', turbulence_scaling)
print('\tHorizontal stride:\t', stride_hor)
print('\tVertical stride:\t', stride_vert)
print('-----------------------------------------------------------------------------')

# start the training for n_epochs
start_time = time.time()
for epoch in range(n_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % plot_every_n_batches == (plot_every_n_batches - 1):    # print every plot_every_n_batches mini-batches
            print('[%d, %5d] averaged loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / (plot_every_n_batches - 1)))
            running_loss = 0.0

    with torch.no_grad():
        train_loss = 0.0
        for data in trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            train_loss += loss.item()
        train_loss /= len(trainloader)

        validation_loss = 0.0
        for data in validationloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            validation_loss += loss.item()
        validation_loss /= len(validationset)

        learning_curve[epoch, 0] = train_loss
        learning_curve[epoch, 1] = validation_loss
        print(('[%d] train loss: %.5f, validation loss: %.5f' %
                      (epoch + 1, train_loss, validation_loss)))


print("INFO: Finished training in %s seconds" % (time.time() - start_time))

# save the model parameter and learning curve if requested
if (save_learning_curve):
    np.save('models/trained_models/' + model_name + '_learningcurve.npy', learning_curve)

if (save_model):
    # save the model
    torch.save(net.state_dict(), 'models/trained_models/' + model_name + '.model')

    # save the model parameters
    model_params = {
        'n_input_layers': n_input_layers,
        'n_output_layers': n_output_layers,
        'n_x': n_x,
        'n_y': n_y,
        'n_z': n_z,
        'n_downsample_layers': n_downsample_layers,
        'interpolation_mode': interpolation_mode,
        'align_corners': align_corners,
        'skipping': skipping,
        'use_terrain_mask': use_terrain_mask,
        'pooling_method': pooling_method,
        'uhor_scaling': uhor_scaling,
        'uz_scaling': uz_scaling,
        'turbulence_scaling': turbulence_scaling,
        'stride_hor': stride_hor,
        'stride_vert': stride_vert,
        'use_turbulence': use_turbulence,
        'd3': d3
        }
    np.save('models/trained_models/' + model_name + '_params.npy', model_params)

# evaluate the model performance on the testset if requested
if (evaluate_testset):
    testset = utils.MyDataset(testset_name, stride_hor = stride_hor, stride_vert = stride_vert, turbulence_label = use_turbulence, scaling_uhor = uhor_scaling, scaling_uz = uz_scaling, scaling_nut = turbulence_scaling)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=num_workers)

    with torch.no_grad():
        loss = 0.0
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss += loss_fn(outputs, labels)

        print('INFO: Average loss on test set: %s' % (loss.item()/len(testset)))
