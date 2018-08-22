#!/usr/bin/env python

import argparse
import models
import numpy as np
import os
from tensorboardX import SummaryWriter
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import utils

now_time = time.strftime("%Y_%m_%d-%H_%M")

parser = argparse.ArgumentParser(description='Training an EDNN for predicting wind data from terrain')
parser.add_argument('-np', '--no-plots', dest='make_plots', action='store_false', help='Turn off plots (default False)')
parser.add_argument('-y', '--yaml-config', required=True, help='YAML config file')
parser.add_argument('-o', '--output-dir', default='models/trained_models/', help='Output directory')

args = parser.parse_args()

run_params = utils.EDNNParameters(args.yaml_config)
model_dir = os.path.join(args.output_dir, run_params.name)
if run_params.run['save_model'] and (not os.path.exists(model_dir)):
    os.mkdir(model_dir)
# --------------------------------------------------------------------------

if (os.path.isdir("/cluster/scratch/")):
    print('Script is running on the cluster')
    trainset_name = '/scratch/train.tar'
    validationset_name = '/scratch/validation.tar'
    testset_name = '/scratch/test.tar'
    os.system('cp '  + run_params.data['trainset_name'] + ' ' + trainset_name)
    os.system('cp '  + run_params.data['validationset_name'] + ' ' + validationset_name)
    os.system('cp '  + run_params.data['testset_name'] + ' ' + testset_name)

else:
    print('Script is running on the a local machine')
    trainset_name = run_params.data['trainset_name']
    validationset_name = run_params.data['validationset_name']
    testset_name = run_params.data['testset_name']


# define dataset and dataloader
trainset = utils.MyDataset(trainset_name, **run_params.MyDataset_kwargs())

trainloader = torch.utils.data.DataLoader(trainset, batch_size=run_params.run['batchsize'],
                                          shuffle=True, num_workers=run_params.run['num_workers'])

validationset = utils.MyDataset(validationset_name, **run_params.MyDataset_kwargs())

validationloader = torch.utils.data.DataLoader(validationset, shuffle=False, batch_size=run_params.run['batchsize'],
                                          num_workers=run_params.run['num_workers'])

#check if gpu is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define model and move to gpu if available
if run_params.model['d3']:
    net = models.ModelEDNN3D(**run_params.model3d_kwargs())
else:
    net = models.ModelEDNN2D(**run_params.model2d_kwargs())

if (run_params.run['warm_start']):
    try:
        net.load_state_dict(torch.load(os.path.join(model_dir, 'latest.model'), map_location=lambda storage, loc: storage))
    except:
        print('Warning: Failed to load the model parameter, initializing parameter.')
        if run_params.run['custom_init']:
            net.init_params()
else:
    if run_params.run['custom_init']:
        net.init_params()

net.to(device)

# define optimizer and objective
optimizer = torch.optim.Adam(net.parameters(), lr=run_params.run['learning_rate_initial'])
scheduler = StepLR(optimizer, step_size=run_params.run['learning_rate_decay_step_size'],
                   gamma=run_params.run['learning_rate_decay'])

if run_params.run['custom_loss']:
    loss_fn = utils.MyLoss(device)
    loss_fn_val = torch.nn.MSELoss()
else:
    loss_fn = torch.nn.MSELoss()
    loss_fn_val = torch.nn.MSELoss()

# initialize the tensorboard writer
writer = SummaryWriter(os.path.join(model_dir, 'learningcurve'))

# save the model parameter in the beginning
run_params.save(model_dir)

print('-----------------------------------------------------------------------------')
print('INFO: Start training on device %s' % device)
print(' ')
run_params.print()
print('-----------------------------------------------------------------------------')

# start the training for n_epochs
start_time = time.time()
for epoch in range(run_params.run['n_epochs']):  # loop over the dataset multiple times

    train_loss = 0
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
        train_loss += loss.item()

        # print every plot_every_n_batches mini-batches
        if i % run_params.run['plot_every_n_batches'] == (run_params.run['plot_every_n_batches'] - 1):
            print('[%d, %5d] averaged loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / (run_params.run['plot_every_n_batches'] - 1)))
            running_loss = 0.0

    # save model every save_model_every_n_epoch epochs
    if epoch % run_params.run['save_model_every_n_epoch'] == (run_params.run['save_model_every_n_epoch'] - 1):
        torch.save(net.state_dict(), os.path.join(model_dir, 'e{}.model'.format(epoch+1)))

    with torch.no_grad():
        if run_params.run['minibatch_epoch_loss']:
            train_loss /= len(trainloader)
        else:
            train_loss = 0.0
            for data in trainloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = loss_fn_val(outputs, labels)
                train_loss += loss.item()
            train_loss /= len(trainloader)

        validation_loss = 0.0
        if run_params.run['compute_validation_loss']:
            for data in validationloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = loss_fn_val(outputs, labels)
                validation_loss += loss.item()
            validation_loss /= len(validationset)

        writer.add_scalar('Train/Loss', train_loss, epoch+1)
        writer.add_scalar('Summary/TrainLoss', train_loss, epoch+1)
        writer.add_scalar('Val/Loss', validation_loss, epoch+1)
        writer.add_scalar('Summary/ValidationLoss', validation_loss, epoch+1)
        writer.add_scalar('Summary/LearningRate', scheduler.get_lr()[0], epoch+1)

        if epoch % run_params.run['save_params_hist_every_n_epoch'] == (run_params.run['save_params_hist_every_n_epoch'] - 1):
            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram(tag, value.data.cpu().numpy(), epoch+1)
                writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)

        print(('[%d] train loss: %.5f, validation loss: %.5f' %
                      (epoch + 1, train_loss, validation_loss)))


print("INFO: Finished training in %s seconds" % (time.time() - start_time))

if (run_params.run['save_model']):
    # save the model
    torch.save(net.state_dict(), os.path.join(model_dir, 'latest.model'))

#writer.add_graph(net.cpu(), inputs.cpu()) # do not save the graph by default as the visualization does not work anyway that well
writer.close()

# evaluate the model performance on the testset if requested
if (run_params.run['evaluate_testset']):
    testset = utils.MyDataset(testset_name, **run_params.MyDataset_kwargs())
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=run_params.data['num_workers'])

    with torch.no_grad():
        loss = 0.0
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss += loss_fn(outputs, labels)

        print('INFO: Average loss on test set: %s' % (loss.item()/len(testset)))

# clean up the scratch folder of the cluster
if (os.path.isdir("/cluster/scratch/")):
    os.system('rm '  + trainset_name)
    os.system('rm '  + validationset_name)
    os.system('rm '  + testset_name)
