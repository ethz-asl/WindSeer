#!/usr/bin/env python

import matplotlib.pyplot as plt
from math import trunc
import models
import numpy as np
import torch
from torch.utils.data import DataLoader
import utils

# ---- Params --------------------------------------------------------------
compressed = False
dataset = 'data/test64.tar'
index = 0 # plot the prediction for the following sample in the set, 1434
model_name = 'run7_naKd4sF4mK'
model_version = 'e105'
compute_prediction_error = False
use_terrain_mask = True
plot_worst_prediction = False
compute_velocity_error = False
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
        velocity_errors = np.zeros(16)

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

            if compute_velocity_error:
                error_stats = utils.prediction_error.compute_prediction_error(labels, outputs, params.data['uhor_scaling'], params.data['uz_scaling'])
                velocity_errors[0] += error_stats['avg_abs_error']
                velocity_errors[1] += error_stats['avg_abs_error_x']
                velocity_errors[2] += error_stats['avg_abs_error_y']
                velocity_errors[3] += error_stats['avg_abs_error_z']
                velocity_errors[4] += error_stats['low_error_x']
                velocity_errors[5] += error_stats['low_error_y']
                velocity_errors[6] += error_stats['low_error_z']
                velocity_errors[7] += error_stats['high_error_x']
                velocity_errors[8] += error_stats['high_error_y']
                velocity_errors[9] += error_stats['high_error_z']
                velocity_errors[10] += error_stats['max_low_x']
                velocity_errors[11] += error_stats['max_low_y']
                velocity_errors[12] += error_stats['max_low_z']
                velocity_errors[13] += error_stats['max_high_x']
                velocity_errors[14] += error_stats['max_high_y']
                velocity_errors[15] += error_stats['max_high_z']

            if ((i % np.ceil(len(testloader)/20.0)) == 0.0):
                print(trunc((i+1)/len(testloader)*100), '%')


        print('INFO: Average loss on test set: %s' % (loss.item()/len(testloader)))
        print('INFO: Average loss on test set for ux: %s' % (loss_ux.item()/len(testloader)))
        if params.model['d3']:
            print('INFO: Average loss on test set for uz: %s' % (loss_uz.item()/len(testloader)))
        print('INFO: Average loss on test set for uz: %s' % (loss_uz.item()/len(testloader)))
        if params.model['use_turbulence']:
            print('INFO: Average loss on test set for turbulence: %s' % (loss_nut.item()/len(testloader)))

        if compute_velocity_error:
            velocity_errors /= len(testloader)
            print('INFO: Average absolute error total: %s [m/s]' % (velocity_errors[0]))
            print('INFO: Average absolute error x:     %s [m/s]' % (velocity_errors[1]))
            if params.model['d3']:
                print('INFO: Average absolute error y:     %s [m/s]' % (velocity_errors[2]))
            print('INFO: Average absolute error z:     %s [m/s]' % (velocity_errors[3]))
            print('INFO: Close terrain error x:        %s [m/s]' % (velocity_errors[4]))
            if params.model['d3']:
                print('INFO: Close terrain error y:        %s [m/s]' % (velocity_errors[5]))
            print('INFO: Close terrain error z:        %s [m/s]' % (velocity_errors[6]))
            print('INFO: High above terrain error x:   %s [m/s]' % (velocity_errors[7]))
            if params.model['d3']:
                print('INFO: High above terrain error y:   %s [m/s]' % (velocity_errors[8]))
            print('INFO: High above terrain error z:   %s [m/s]' % (velocity_errors[9]))
            print('INFO: Close terrain max error x:    %s [m/s]' % (velocity_errors[10]))
            if params.model['d3']:
                print('INFO: Close terrain max error y:    %s [m/s]' % (velocity_errors[11]))
            print('INFO: Close terrain max error z:    %s [m/s]' % (velocity_errors[12]))
            print('INFO: High above terrain maxerr x:  %s [m/s]' % (velocity_errors[13]))
            if params.model['d3']:
                print('INFO: High above terrain maxerr y:  %s [m/s]' % (velocity_errors[14]))
            print('INFO: High above terrain maxerr z:  %s [m/s]' % (velocity_errors[15]))


    # plot prediction
    if plot_worst_prediction and (worst_index != -1):
        print('Plotting sample with the highest loss: {}'.format(maxloss))
        print(worst_index)
        input, label = testset[worst_index]
    else:
        input, label = testset[index]

    input, label = input.to(device), label.to(device)
    print('start')
    start_time = time.time()
    output = net(input.unsqueeze(0))
    print('INFO: Inference time: ', (time.time() - start_time), 'seconds')
    input = input.squeeze()
    output = output.squeeze()
    print('Plotting sample with loss: {}'.format(loss_fn(output, label)))

    if params.model['d3']:
        output[0,:,:,:] *= params.data['uhor_scaling']
        output[1,:,:,:] *= params.data['uhor_scaling']
        output[2,:,:,:] *= params.data['uz_scaling']
        label[0,:,:,:] *= params.data['uhor_scaling']
        label[1,:,:,:] *= params.data['uhor_scaling']
        label[2,:,:,:] *= params.data['uz_scaling']
    else:
        output[0,:,:] *= params.data['uhor_scaling']
        output[1,:,:] *= params.data['uz_scaling']
        label[0,:,:] *= params.data['uhor_scaling']
        label[1,:,:] *= params.data['uz_scaling']
    utils.plot_prediction(output, label)
