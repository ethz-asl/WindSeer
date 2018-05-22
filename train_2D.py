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
savepath = 'models/trained_models/ednn_2D_v1.model'



trainset = utils.MyDataset('data/clean_train.zip')
testset = utils.MyDataset('data/test.zip')

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=2)

net = models.ModelEDNN2D(3)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

start_time = time.time()
for epoch in range(n_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

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

