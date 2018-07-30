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
n_epochs = 10
batchsize = 32
num_workers = 4

# options to store data
save_model = True
save_learning_curve = True
evaluate_testset = True
warm_start = False
custom_loss = False

# dataset parameter
trainset_name = 'data/converted_train_new_boolean.tar'
validationset_name = 'data/converted_validation_new_boolean.tar'
testset_name = 'data/converted_test_new_boolean.tar'

# model parameter
model_name = 'ednn_2D_scaled_nearest_skipping_new_boolean'
uhor_scaling = 9.0
uz_scaling = 2.5
turbulence_scaling = 4.5
use_turbulence = False
interpolation_mode = 'nearest'
align_corners = False
number_input_layers = 3
skipping = True
# --------------------------------------------------------------------------

# define dataset and dataloader
trainset = utils.MyDataset(trainset_name, turbulence_label = use_turbulence, scaling_uhor = uhor_scaling, scaling_uz = uz_scaling, scaling_nut = turbulence_scaling)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                          shuffle=True, num_workers=num_workers)

validationset = utils.MyDataset(validationset_name, turbulence_label = use_turbulence, scaling_uhor = uhor_scaling, scaling_uz = uz_scaling, scaling_nut = turbulence_scaling)

validationloader = torch.utils.data.DataLoader(validationset, batch_size=1,
                                          shuffle=False, num_workers=num_workers)


#check if gpu is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('INFO: Start training on device %s' % device)

# define model and move to gpu if available
net = models.ModelEDNN2D(number_input_layers, interpolation_mode = interpolation_mode, align_corners = align_corners, skipping = skipping)

if (warm_start):
    net.load_state_dict(torch.load('models/trained_models/' + model_name + '.model', map_location=lambda storage, loc: storage))
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
        'uhor_scaling': uhor_scaling,
        'uz_scaling': uz_scaling,
        'turbulence_scaling': turbulence_scaling,
        'interpolation_mode': interpolation_mode,
        'align_corners': align_corners,
        'number_input_layers': number_input_layers,
        'skipping': skipping,
        'use_turbulence': use_turbulence
        }
    np.save('models/trained_models/' + model_name + '_params.npy', model_params)

# evaluate the model performance on the testset if requested
if (evaluate_testset):
    testset = utils.MyDataset(testset_name, turbulence_label = use_turbulence, scaling_uhor = uhor_scaling, scaling_uz = uz_scaling, scaling_nut = turbulence_scaling)
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
