#!/usr/bin/env python

from __future__ import print_function

import argparse
import nn_wind_prediction.data as data
import nn_wind_prediction.models as models
import nn_wind_prediction.nn as nn_custom
import nn_wind_prediction.utils as utils
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

now_time = time.strftime("%Y_%m_%d-%H_%M")

parser = argparse.ArgumentParser(description='Training an EDNN for predicting wind data from terrain')
parser.add_argument('-np', '--no-plots', dest='make_plots', action='store_false', help='Turn off plots (default False)')
parser.add_argument('-y', '--yaml-config', required=True, help='YAML config file')
parser.add_argument('-o', '--output-dir', default='trained_models/', help='Output directory')
parser.add_argument('-w', '--writer', dest='use_writer', default=True, action='store_false', help='Don\'t use a SummaryWriter to log the learningcurve')

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
    t_start = time.time()
    os.system('cp '  + run_params.data['trainset_name'] + ' ' + trainset_name)
    print("INFO: Finished copying trainset in %s seconds" % (time.time() - t_start))
    t_intermediate = time.time()
    os.system('cp '  + run_params.data['validationset_name'] + ' ' + validationset_name)
    print("INFO: Finished copying validationset in %s seconds" % (time.time() - t_intermediate))
    if run_params.run['evaluate_testset']:
        t_intermediate = time.time()
        os.system('cp '  + run_params.data['testset_name'] + ' ' + testset_name)
        print("INFO: Finished copying testset in %s seconds" % (time.time() - t_intermediate))

else:
    print('Script is running on the a local machine')
    trainset_name = run_params.data['trainset_name']
    validationset_name = run_params.data['validationset_name']
    testset_name = run_params.data['testset_name']


# define dataset and dataloader
trainset = data.MyDataset(trainset_name, compressed = run_params.data['compressed'], **run_params.MyDataset_kwargs())

trainloader = torch.utils.data.DataLoader(trainset, batch_size=run_params.run['batchsize'],
                                          shuffle=True, num_workers=run_params.run['num_workers'])

validationset = data.MyDataset(validationset_name, compressed = run_params.data['compressed'], **run_params.MyDataset_kwargs())

validationloader = torch.utils.data.DataLoader(validationset, shuffle=False, batch_size=run_params.run['batchsize'],
                                          num_workers=run_params.run['num_workers'])

#check if gpu is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define model and move to gpu if available
if run_params.model['d3']:
    if run_params.model['predict_uncertainty']:
        net = models.ModelEDNN3D_Twin(**run_params.model3d_kwargs())
    else:
        net = models.ModelEDNN3D(**run_params.model3d_kwargs())
else:
    net = models.ModelEDNN2D(**run_params.model2d_kwargs())

if (run_params.run['warm_start']):
    try:
        net.load_state_dict(torch.load(os.path.join(model_dir, 'pretrained.model'), map_location=lambda storage, loc: storage))
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

if run_params.run['loss_function'] == 1:
    loss_fn = torch.nn.L1Loss()
elif run_params.run['loss_function'] == 2:
    loss_fn = nn_custom.GaussianLogLikelihoodLoss()
elif run_params.run['loss_function'] == 3:
    loss_fn = utils.MyLoss(device)
else:
    loss_fn = torch.nn.MSELoss()

# save the model parameter in the beginning
run_params.save(model_dir)

print('-----------------------------------------------------------------------------')
print('INFO: Start training on device %s' % device)
print(' ')
run_params.print()
print('-----------------------------------------------------------------------------')

# start the actual training
net = nn_custom.train_model(net, trainloader, validationloader, scheduler, optimizer,
                       loss_fn, device, run_params.run['n_epochs'],
                       run_params.run['plot_every_n_batches'], run_params.run['save_model_every_n_epoch'],
                       run_params.run['save_params_hist_every_n_epoch'], run_params.run['minibatch_epoch_loss'],
                       run_params.run['compute_validation_loss'], model_dir, args.use_writer, run_params.model['predict_uncertainty'])

# save the model if requested
if (run_params.run['save_model']):
    torch.save(net.state_dict(), os.path.join(model_dir, 'latest.model'))

# evaluate the model performance on the testset if requested
if (run_params.run['evaluate_testset']):
    testset = utils.MyDataset(testset_name, compressed = run_params.data['compressed'], **run_params.MyDataset_kwargs())
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
    if run_params.run['evaluate_testset']:
        os.system('rm '  + testset_name)
