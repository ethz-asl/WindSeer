#!/usr/bin/env python

import matplotlib.pyplot as plt
import models
import numpy as np
import torch
from torch.utils.data import DataLoader
import utils

# ---- Params --------------------------------------------------------------
dataset = 'data/converted_3d.tar'
index = 0 # plot the prediction for the following sample in the set
model_name = 'ednn_3D_n_sb3s_10000epochs'
compute_prediction_error = True
use_terrain_mask = True
# --------------------------------------------------------------------------

# check if gpu available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the model parameter
params = np.load('models/trained_models/' + model_name + '_params.npy')
params = params.item()

# load dataset
testset = utils.MyDataset(dataset, stride_hor = params['stride_hor'], stride_vert = params['stride_vert'], turbulence_label = params['use_turbulence'], scaling_uhor = params['uhor_scaling'], scaling_uz = params['uz_scaling'], scaling_nut = params['turbulence_scaling'])
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=0)
# load the model and its learnt parameters
if params['d3']:
    net = models.ModelEDNN3D(params['n_input_layers'], params['n_output_layers'], params['n_x'], params['n_y'], params['n_z'], params['n_downsample_layers'], params['interpolation_mode'], params['align_corners'], params['skipping'], use_terrain_mask, params['pooling_method'])
else:
    net = models.ModelEDNN2D(params['n_input_layers'], params['interpolation_mode'], params['align_corners'], params['skipping'], params['use_turbulence'])

net.load_state_dict(torch.load('models/trained_models/' + model_name + '.model', map_location=lambda storage, loc: storage))
net.to(device)

# define loss function
loss_fn = torch.nn.MSELoss()

#Compute the average loss
with torch.no_grad():
    if compute_prediction_error:
        loss = 0.0
        loss_ux = 0.0
        loss_uy = 0.0
        loss_uz = 0.0
        loss_nut = 0.0
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss += loss_fn(outputs, labels)
            if d3:
                loss_ux += loss_fn(outputs[:,0,:,:,:], labels[:,0,:,:,:])
                loss_uy += loss_fn(outputs[:,1,:,:,:], labels[:,1,:,:,:])
                loss_uz += loss_fn(outputs[:,2,:,:,:], labels[:,2,:,:,:])
                if params['use_turbulence']:
                    loss_nut += loss_fn(outputs[:,3,:,:,:], labels[:,3,:,:,:])
            else:
                loss_ux += loss_fn(outputs[:,0,:,:], labels[:,0,:,:])
                loss_uz += loss_fn(outputs[:,1,:,:], labels[:,1,:,:])
                if params['use_turbulence']:
                    loss_nut += loss_fn(outputs[:,2,:,:], labels[:,2,:,:])
    
        print('INFO: Average loss on test set: %s' % (loss.item()/len(testset)))
        print('INFO: Average loss on test set for ux: %s' % (loss_ux.item()/len(testset)))
        if params['d3']:
            print('INFO: Average loss on test set for uz: %s' % (loss_uz.item()/len(testset)))
        print('INFO: Average loss on test set for uz: %s' % (loss_uz.item()/len(testset)))
        if params['use_turbulence']:
            print('INFO: Average loss on test set for turbulence: %s' % (loss_nut.item()/len(testset)))

    # plot prediction
    input, label = testset[index]
    input, label = input.to(device), label.to(device)
    output = net(input.unsqueeze(0))
    input = input.squeeze()
    output = output.squeeze()

    utils.plot_prediction(output, label)
