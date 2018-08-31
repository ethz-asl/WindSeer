#!/usr/bin/env python

import matplotlib.pyplot as plt
import models
import numpy as np
import torch
from torch.utils.data import DataLoader
import utils

# ---- Params --------------------------------------------------------------
compressed = True
dataset = 'data/test.tar'
index = 0 # plot the prediction for the following sample in the set
model_name = '64_r1_naKd4sF2mK'
model_version = 'e95'
compute_prediction_error = False
use_terrain_mask = True
plot_worst_prediction = False
# --------------------------------------------------------------------------

# check if gpu available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the model parameter
params = utils.EDNNParameters('models/trained_models/' + model_name + '/params.yaml')

# load dataset
testset = utils.MyDataset(dataset, compressed = compressed, **params.MyDataset_kwargs())
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=0)
# load the model and its learnt parameters
if params.model['d3']:
    net = models.ModelEDNN3D(params.model['n_input_layers'], params.model['n_output_layers'], params.model['n_x'], params.model['n_y'], params.model['n_z'],
                             params.model['n_downsample_layers'], params.model['interpolation_mode'], params.model['align_corners'], params.model['skipping'],
                             use_terrain_mask, params.model['pooling_method'], params.model['use_mapping_layer'], params.model['use_fc_layers'], params.model['fc_scaling'])
else:
    net = models.ModelEDNN2D(params.model['n_input_layers'], params.model['interpolation_mode'], params.model['align_corners'], params.model['skipping'], params.model['use_turbulence'])

net.load_state_dict(torch.load('models/trained_models/' + model_name + '/' + model_version + '.model', map_location=lambda storage, loc: storage))
net.to(device)

# define loss function
loss_fn = torch.nn.MSELoss()

#Compute the average loss
with torch.no_grad():
    worst_index = -1
    maxloss = -200000.0 #arbitrary large negative number, the loss should always be higher than that number

    if compute_prediction_error:
        loss = 0.0
        loss_ux = 0.0
        loss_uy = 0.0
        loss_uz = 0.0
        loss_nut = 0.0
        for i, data in enumerate(testloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)

            dloss = loss_fn(outputs, labels)
            if dloss > maxloss:
                maxloss = dloss
                worst_index = i

            loss += dloss
            if params.model['d3']:
                loss_ux += loss_fn(outputs[:,0,:,:,:], labels[:,0,:,:,:])
                loss_uy += loss_fn(outputs[:,1,:,:,:], labels[:,1,:,:,:])
                loss_uz += loss_fn(outputs[:,2,:,:,:], labels[:,2,:,:,:])
                if params.model['use_turbulence']:
                    loss_nut += loss_fn(outputs[:,3,:,:,:], labels[:,3,:,:,:])
            else:
                loss_ux += loss_fn(outputs[:,0,:,:], labels[:,0,:,:])
                loss_uz += loss_fn(outputs[:,1,:,:], labels[:,1,:,:])
                if params.model['use_turbulence']:
                    loss_nut += loss_fn(outputs[:,2,:,:], labels[:,2,:,:])
    
        print('INFO: Average loss on test set: %s' % (loss.item()/len(testset)))
        print('INFO: Average loss on test set for ux: %s' % (loss_ux.item()/len(testset)))
        if params.model['d3']:
            print('INFO: Average loss on test set for uz: %s' % (loss_uz.item()/len(testset)))
        print('INFO: Average loss on test set for uz: %s' % (loss_uz.item()/len(testset)))
        if params.model['use_turbulence']:
            print('INFO: Average loss on test set for turbulence: %s' % (loss_nut.item()/len(testset)))

    # plot prediction
    if plot_worst_prediction and (worst_index != -1):
        print('Plotting sample with the highest loss: {}'.format(maxloss))
        print(worst_index)
        input, label = testset[worst_index]
    else:
        input, label = testset[index]

    input, label = input.to(device), label.to(device)
    output = net(input.unsqueeze(0))
    input = input.squeeze()
    output = output.squeeze()
    print('Plotting sample with loss: {}'.format(loss_fn(output, label)))

    utils.plot_prediction(output, label)
