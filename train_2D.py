#!/usr/bin/env python

import models
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import utils

# ---- Params --------------------------------------------------------------
learning_rate = 1e-3
plot_every_n_batches = 10
n_epochs = 10
batchsize = 32
save_model = True
savepath = 'models/trained_models/ednn_2D_v3_scaled.model'
trainset_name = 'data/converted_train.tar'
testset_name = 'data/converted_test.tar'
evaluate_testset = True
ux_scaling = 9.0
uz_scaling = 2.5
turbulence_scaling = 4.5
num_workers = 4

# --------------------------------------------------------------------------

# define dataset and dataloader
trainset = utils.MyDataset(trainset_name,  scaling_ux = ux_scaling, scaling_uz = uz_scaling, scaling_nut = turbulence_scaling)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                          shuffle=True, num_workers=num_workers)

#check if gpu is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('INFO: Start training on device %s' % device)

# define model and move to gpu if available
net = models.ModelEDNN2D(3)
net.to(device)

# define optimizer and objective
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

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
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / (plot_every_n_batches - 1)))
            running_loss = 0.0

print("INFO: Finished training in %s seconds" % (time.time() - start_time))

# save the model parameter if requested
if (save_model):
    torch.save(net.state_dict(), savepath)

# evaluate the model performance on the testset if requested
if (evaluate_testset):
    testset = utils.MyDataset(testset_name,  scaling_ux = ux_scaling, scaling_uz = uz_scaling, scaling_nut = turbulence_scaling)
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
