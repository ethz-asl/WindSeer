#!/usr/bin/env python

import models
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import utils

# params
learning_rate = 1e-4
plot_every_n_batches = 10
n_epochs = 10
save_model = True
savepath = 'models/trained_models/ednn_2D_v2_scaled.model'
evaluate_testset = True

trainset = utils.MyDataset('data/clean_train.zip',  scaling_ux = 10.0, scaling_uz = 2.5, scaling_nut = 10.0)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

#check if gpu is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('INFO: Start training on device %s' % device)

net = models.ModelEDNN2D(3)
net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

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

if (save_model):
    torch.save(net.state_dict(), savepath)

if (evaluate_testset):
    testset = utils.MyDataset('data/test.zip',  scaling_ux = 10.0, scaling_uz = 2.5, scaling_nut = 10.0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=2)

    with torch.no_grad():
        loss = 0.0
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss += loss_fn(outputs, labels)

        print('INFO: Average loss on test set: %s' % (loss.item()/len(testset)))
