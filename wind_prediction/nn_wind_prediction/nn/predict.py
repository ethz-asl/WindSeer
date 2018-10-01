#!/usr/bin/env python

from math import trunc
import nn_wind_prediction.utils as utils
import numpy as np
import time
import torch

def dataset_prediction_error(net, device, params, loss_fn, loader_testset):
    #Compute the average loss
    with torch.no_grad():
        worst_index = -1
        maxloss = -200000.0 #arbitrary large negative number, the loss should always be higher than that number

        loss = 0.0
        loss_ux = 0.0
        loss_uy = 0.0
        loss_uz = 0.0
        loss_nut = 0.0
        velocity_errors = np.zeros((16, len(loader_testset)))

        for i, data in enumerate(loader_testset):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)

            if params.model['predict_uncertainty']:
                num_channels = outputs.shape[1]
                dloss = loss_fn(outputs[:,:int(num_channels/2),:], labels)
            else:
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

            error_stats = utils.prediction_error.compute_prediction_error(labels, outputs, params.data['uhor_scaling'], params.data['uz_scaling'], params.model['predict_uncertainty'])
            velocity_errors[0, i] = error_stats['avg_abs_error']
            velocity_errors[1, i] = error_stats['avg_abs_error_x']
            velocity_errors[2, i] = error_stats['avg_abs_error_y']
            velocity_errors[3, i] = error_stats['avg_abs_error_z']
            velocity_errors[4, i] = error_stats['low_error_hor']
            velocity_errors[5, i] = error_stats['low_error_vert']
            velocity_errors[6, i] = error_stats['low_error_tot']
            velocity_errors[7, i] = error_stats['high_error_hor']
            velocity_errors[8, i] = error_stats['high_error_vert']
            velocity_errors[9, i] = error_stats['high_error_tot']
            velocity_errors[10, i] = error_stats['max_low_hor']
            velocity_errors[11, i] = error_stats['max_low_vert']
            velocity_errors[12, i] = error_stats['max_low_tot']
            velocity_errors[13, i] = error_stats['max_high_hor']
            velocity_errors[14, i] = error_stats['max_high_vert']
            velocity_errors[15, i] = error_stats['max_high_tot']

            if ((i % np.ceil(len(loader_testset)/20.0)) == 0.0):
                print(trunc((i+1)/len(loader_testset)*100), '%')

        print('INFO: Average loss on test set: %s' % (loss.item()/len(loader_testset)))
        print('INFO: Average loss on test set for ux: %s' % (loss_ux.item()/len(loader_testset)))
        if params.model['d3']:
            print('INFO: Average loss on test set for uz: %s' % (loss_uz.item()/len(loader_testset)))
        print('INFO: Average loss on test set for uz: %s' % (loss_uz.item()/len(loader_testset)))
        if params.model['use_turbulence']:
            print('INFO: Average loss on test set for turbulence: %s' % (loss_nut.item()/len(loader_testset)))

        print('INFO: Average absolute error total:   %s [m/s]' % (np.mean(velocity_errors[0, :])))
        print('INFO: Average absolute error x:       %s [m/s]' % (np.mean(velocity_errors[1, :])))
        if params.model['d3']:
            print('INFO: Average absolute error y:       %s [m/s]' % (np.mean(velocity_errors[2, :])))
        print('INFO: Average absolute error z:       %s [m/s]' % (np.mean(velocity_errors[3, :])))
        print('INFO: Close terrain error hor:        %s [m/s]' % (np.mean(velocity_errors[4, :])))
        print('INFO: Close terrain error vert:       %s [m/s]' % (np.mean(velocity_errors[5, :])))
        print('INFO: Close terrain error tot:        %s [m/s]' % (np.mean(velocity_errors[6, :])))
        print('INFO: High above terrain error hor:   %s [m/s]' % (np.mean(velocity_errors[7, :])))
        print('INFO: High above terrain error vert:  %s [m/s]' % (np.mean(velocity_errors[8, :])))
        print('INFO: High above terrain error tot:   %s [m/s]' % (np.mean(velocity_errors[9, :])))
        print('INFO: Close terrain max error hor:    %s [m/s]' % (np.mean(velocity_errors[10, :])))
        print('INFO: Close terrain max error vert:   %s [m/s]' % (np.mean(velocity_errors[11, :])))
        print('INFO: Close terrain max error tot:    %s [m/s]' % (np.mean(velocity_errors[12, :])))
        print('INFO: High above terrain maxerr hor:  %s [m/s]' % (np.mean(velocity_errors[13, :])))
        print('INFO: High above terrain maxerr vert: %s [m/s]' % (np.mean(velocity_errors[14, :])))
        print('INFO: High above terrain maxerr tot:  %s [m/s]' % (np.mean(velocity_errors[15, :])))
        
        return velocity_errors, worst_index, maxloss

def predict_wind_and_turbulence(input, label, device, net, params, plotting_prediction, loss_fn = None):
    with torch.no_grad():
        input, label = input.to(device), label.to(device)
        start_time = time.time()
        output = net(input.unsqueeze(0))
        print('INFO: Inference time: ', (time.time() - start_time), 'seconds')
        input = input.squeeze()
        output = output.squeeze()
    
        if params.model['predict_uncertainty']:
            num_channels = output.shape[0]
            if loss_fn:
                print('Loss: {}'.format(loss_fn(output[:int(num_channels/2),:], label)))
        else:
            if loss_fn:
                print('Loss: {}'.format(loss_fn(output, label)))
    
        if plotting_prediction:
            if params.model['d3']:
                output[0,:,:,:] *= params.data['uhor_scaling']
                output[1,:,:,:] *= params.data['uhor_scaling']
                output[2,:,:,:] *= params.data['uz_scaling']
                label[0,:,:,:] *= params.data['uhor_scaling']
                label[1,:,:,:] *= params.data['uhor_scaling']
                label[2,:,:,:] *= params.data['uz_scaling']
                if params.model['use_turbulence']:
                    output[3,:,:,:] *= params.data['turbulence_scaling']
                    label[3,:,:,:] *= params.data['turbulence_scaling']
        
            else:
                output[0,:,:] *= params.data['uhor_scaling']
                output[1,:,:] *= params.data['uz_scaling']
                label[0,:,:] *= params.data['uhor_scaling']
                label[1,:,:] *= params.data['uz_scaling']
        
            utils.plot_prediction(output, label, params.model['predict_uncertainty'])
