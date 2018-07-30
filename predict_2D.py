#!/usr/bin/env python

import matplotlib.pyplot as plt
import models
import numpy as np
import torch
from torch.utils.data import DataLoader
import utils

# ---- Params --------------------------------------------------------------
dataset = 'data/converted_test_new_boolean.tar'
index = 12 # plot the prediction for the following sample in the set
model_name = 'ednn_2D_scaled_nearest_skipping_new_boolean'

# --------------------------------------------------------------------------

# check if gpu available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the model parameter
params = np.load('models/trained_models/' + model_name + '_params.npy')
params = params.item()

# load dataset
testset = utils.MyDataset(dataset, turbulence_label = params['use_turbulence'], scaling_uhor = params['uhor_scaling'], scaling_uz = params['uz_scaling'], scaling_nut = params['turbulence_scaling'])
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=0)
# load the model and its learnt parameters
net = models.ModelEDNN2D(params['number_input_layers'], params['interpolation_mode'], params['align_corners'], params['skipping'])
net.load_state_dict(torch.load('models/trained_models/' + model_name + '.model', map_location=lambda storage, loc: storage))
net.to(device)

# define loss function
loss_fn = torch.nn.MSELoss()

#Compute the average loss
with torch.no_grad():
    loss = 0.0
    loss_ux = 0.0
    loss_uz = 0.0
    loss_nut = 0.0
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss += loss_fn(outputs, labels)
        loss_ux += loss_fn(outputs[0,0,:,:], labels[0,0,:,:])
        loss_uz += loss_fn(outputs[0,1,:,:], labels[0,1,:,:])
        #loss_nut += loss_fn(outputs[0,2,:,:], labels[0,2,:,:])

    print('INFO: Average loss on test set: %s' % (loss.item()/len(testset)))
    print('INFO: Average loss on test set for ux: %s' % (loss_ux.item()/len(testset)))
    print('INFO: Average loss on test set for uz: %s' % (loss_uz.item()/len(testset)))
    #print('INFO: Average loss on test set for turbulence: %s' % (loss_nut.item()/len(testset)))

    # plot prediction
    input, label = testset[index]
    input, label = input.to(device), label.to(device)
    output = net(input.unsqueeze(0))
    input = input.squeeze()
    output = output.squeeze()

    error = label - output
    
    fh_in, ah_in = plt.subplots(2, 3)
    fh_in.set_size_inches([6.2, 10.2])

    h_ux_lab = ah_in[0][0].imshow(label[0,:,:], origin='lower', vmin=label[0,:,:].min(), vmax=label[0,:,:].max())
    h_ux_out = ah_in[0][1].imshow(output[0,:,:], origin='lower', vmin=label[0,:,:].min(), vmax=label[0,:,:].max())
    h_ux_err = ah_in[0][2].imshow(error[0,4:-4,4:-4], origin='lower', vmin=error[0,4:-4,4:-4].min(), vmax=error[0,4:-4,4:-4].max())
    ah_in[0][0].set_title('Ux in')
    ah_in[0][1].set_title('Ux predicted')
    ah_in[0][2].set_title('Ux error')
    fh_in.colorbar(h_ux_lab, ax=ah_in[0][0])
    fh_in.colorbar(h_ux_out, ax=ah_in[0][1])
    fh_in.colorbar(h_ux_err, ax=ah_in[0][2])
    
    h_uz_lab = ah_in[1][0].imshow(label[1,:,:], origin='lower', vmin=label[1,:,:].min(), vmax=label[1,:,:].max())
    h_uz_out = ah_in[1][1].imshow(output[1,:,:], origin='lower', vmin=label[1,:,:].min(), vmax=label[1,:,:].max())
    h_uz_err = ah_in[1][2].imshow(error[1,4:-4,4:-4], origin='lower', vmin=error[1,4:-4,4:-4].min(), vmax=error[1,4:-4,4:-4].max())
    ah_in[1][0].set_title('Uz in')
    ah_in[1][1].set_title('Uz predicted')
    ah_in[1][2].set_title('Uz error')
    fh_in.colorbar(h_uz_lab, ax=ah_in[1][0])
    fh_in.colorbar(h_uz_out, ax=ah_in[1][1])
    fh_in.colorbar(h_uz_err, ax=ah_in[1][2])
    
#     h_turb_lab = ah_in[2][0].imshow(label[2,:,:], origin='lower', vmin=label[2,:,:].min(), vmax=label[2,:,:].max())
#     h_turb_out = ah_in[2][1].imshow(output[2,:,:], origin='lower', vmin=label[2,:,:].min(), vmax=label[2,:,:].max())
#     h_turb_err = ah_in[2][2].imshow(error[2,4:-4,4:-4], origin='lower', vmin=error[2,4:-4,4:-4].min(), vmax=error[2,4:-4,4:-4].max())
#     ah_in[2][0].set_title('Turbulence viscosity in')
#     ah_in[2][1].set_title('Turbulence viscosity predicted')
#     ah_in[2][2].set_title('Turbulence viscosity error')
#     fh_in.colorbar(h_turb_lab, ax=ah_in[2][0])
#     fh_in.colorbar(h_turb_out, ax=ah_in[2][1])
#     fh_in.colorbar(h_turb_err, ax=ah_in[2][2])
    
    plt.show()
