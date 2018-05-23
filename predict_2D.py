#!/usr/bin/env python

import matplotlib.pyplot as plt
import models
import torch
from torch.utils.data import DataLoader
import utils

# params
dataset = 'data/test.zip'
index = 3 # plot the prediction for the following sample in the set
model = 'models/trained_models/ednn_2D_v1.model'


# load sample
testset = utils.MyDataset(dataset)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = models.ModelEDNN2D(3)
net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))
net.to(device)
loss_fn = torch.nn.MSELoss()

testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=2)

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
        loss_nut += loss_fn(outputs[0,2,:,:], labels[0,2,:,:])

    print('INFO: Average loss on test set: %s' % (loss.item()/len(testset)))
    print('INFO: Average loss on test set for ux: %s' % (loss_ux.item()/len(testset)))
    print('INFO: Average loss on test set for uz: %s' % (loss_uz.item()/len(testset)))
    print('INFO: Average loss on test set for turbulence: %s' % (loss_nut.item()/len(testset)))


    # plot prediction
    input, label = testset[index]
    input, label = input.to(device), label.to(device)
    output = net(input.unsqueeze(0))
    input = input.squeeze()
    output = output.squeeze()

    
    fh_in, ah_in = plt.subplots(3, 2)
    fh_in.set_size_inches([6.2, 10.2])

    h_ux_in = ah_in[0][0].imshow(label[0,:,:], origin='lower', vmin=label[0,:,:].min(), vmax=label[0,:,:].max())
    h_uz_in = ah_in[0][1].imshow(output[0,:,:], origin='lower', vmin=label[0,:,:].min(), vmax=label[0,:,:].max())
    ah_in[0][0].set_title('Ux in')
    ah_in[0][1].set_title('Ux predicted')
    fh_in.colorbar(h_ux_in, ax=ah_in[0][0])
    fh_in.colorbar(h_uz_in, ax=ah_in[0][1])
    
    h_ux_in = ah_in[1][0].imshow(label[1,:,:], origin='lower', vmin=label[1,:,:].min(), vmax=label[1,:,:].max())
    h_uz_in = ah_in[1][1].imshow(output[1,:,:], origin='lower', vmin=label[1,:,:].min(), vmax=label[1,:,:].max())
    ah_in[1][0].set_title('Uz in')
    ah_in[1][1].set_title('Uz predicted')
    fh_in.colorbar(h_ux_in, ax=ah_in[1][0])
    fh_in.colorbar(h_uz_in, ax=ah_in[1][1])
    
    h_ux_in = ah_in[2][0].imshow(label[2,:,:], origin='lower', vmin=label[2,:,:].min(), vmax=label[2,:,:].max())
    h_uz_in = ah_in[2][1].imshow(output[2,:,:], origin='lower', vmin=label[2,:,:].min(), vmax=label[2,:,:].max())
    ah_in[2][0].set_title('Turbulence viscosity in')
    ah_in[2][1].set_title('Turbulence viscosity predicted')
    fh_in.colorbar(h_ux_in, ax=ah_in[2][0])
    fh_in.colorbar(h_uz_in, ax=ah_in[2][1])
    
    plt.show()
