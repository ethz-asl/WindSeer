#!/usr/bin/env python

import models
import numpy as np
from tensorboardX import SummaryWriter
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import utils

# ---- Params --------------------------------------------------------------
# learning parameters
plot_every_n_batches = 10
n_epochs = 10000
batchsize = 1
num_workers = 1
learning_rate_initial = 1e-3
learning_rate_decay = 0.5
learning_rate_decay_step_size = 2000
compute_validation_loss = False

# options to store data
save_model = True
save_metadata = True
evaluate_testset = False
warm_start = False
custom_loss = False
save_model_every_n_epoch = 100000
save_params_hist_every_n_epoch = 200

# dataset parameter
trainset_name = 'data/converted_3d.tar'
validationset_name = 'data/converted_3d.tar'
testset_name = 'data/converted_3d.tar'
stride_hor = 4
stride_vert = 2
uhor_scaling = 6.0
uz_scaling = 2.5
turbulence_scaling = 4.5

# model parameter
d3 = True
model_name = 'ednn_3D_n_sb3s_10000epochs_noterrain'
n_input_layers = 4
n_output_layers = 3
n_x = 32
n_y = 32
n_z = 32
n_downsample_layers = 3
interpolation_mode = 'nearest'
align_corners = False
skipping = True
use_terrain_mask = False
pooling_method = 'striding'
# --------------------------------------------------------------------------

# decide if turbulence is used (somewhat a hack maybe there is something better in the future)
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
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate_initial)
scheduler = StepLR(optimizer, step_size=learning_rate_decay_step_size, gamma=learning_rate_decay)

if custom_loss:
    loss_fn = utils.MyLoss(device)
else:
    loss_fn = torch.nn.MSELoss()

# initialize the tensorboard writer
writer = SummaryWriter('models/trained_models/' + model_name + '_learningcurve')

# save the model parameter in the beginning
if (save_model):
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

print('-----------------------------------------------------------------------------')
print('INFO: Start training on device %s' % device)
print(' ')
print('Train Settings:')
print('\tWarm start:\t\t', warm_start)
print('\tLearning rate initial:\t', learning_rate_initial)
print('\tLearning rate step size:', learning_rate_decay_step_size)
print('\tLearning rate decay:\t', learning_rate_decay)
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
        # adjust the learning rate if necessary
        scheduler.step()

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

    if epoch % save_model_every_n_epoch == (save_model_every_n_epoch - 1):    # save model every save_model_every_n_epoch epochs
        torch.save(net.state_dict(), 'models/trained_models/' + model_name + '_{}.model'.format(epoch+1))

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
        if compute_validation_loss:
            for data in validationloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = loss_fn(outputs, labels)
                validation_loss += loss.item()
            validation_loss /= len(validationset)

        writer.add_scalar('Train/Loss', train_loss, epoch+1)
        writer.add_scalar('Summary/TrainLoss', train_loss, epoch+1)
        writer.add_scalar('Val/Loss', validation_loss, epoch+1)
        writer.add_scalar('Summary/ValidationLoss', validation_loss, epoch+1)
        writer.add_scalar('Summary/LearningRate', scheduler.get_lr()[0], epoch+1)

        if epoch % save_params_hist_every_n_epoch == (save_params_hist_every_n_epoch - 1):
            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram(tag, value.data.cpu().numpy(), epoch+1)
                writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)

        print(('[%d] train loss: %.5f, validation loss: %.5f' %
                      (epoch + 1, train_loss, validation_loss)))


print("INFO: Finished training in %s seconds" % (time.time() - start_time))

if (save_model):
    # save the model
    torch.save(net.state_dict(), 'models/trained_models/' + model_name + '.model')

#writer.add_graph(net.cpu(), inputs.cpu()) # do not save the graph by default as the visualization does not work anyway that well
writer.close()

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
