#!/usr/bin/env python

from __future__ import print_function

import argparse
import nn_wind_prediction.data as data
import nn_wind_prediction.models as models
import nn_wind_prediction.nn as nn_custom
import nn_wind_prediction.utils as utils
import os
import random
import string
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
    tempfolder = os.environ['TMPDIR'] + '/'
    print('Script is running on the cluster, copying files to', tempfolder)
    trainset_name = tempfolder + 'train.tar'
    validationset_name = tempfolder + 'validation.tar'
    testset_name = tempfolder + 'test.tar'
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

#check if gpu is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define dataset and dataloader
trainset = data.MyDataset(trainset_name, compressed = run_params.data['compressed'],
                          augmentation = run_params.data['augmentation'],
                          augmentation_mode = run_params.data['augmentation_mode'],
                          augmentation_kwargs = run_params.data['augmentation_kwargs'],
                          **run_params.MyDataset_kwargs())

trainloader = torch.utils.data.DataLoader(trainset, batch_size=run_params.run['batchsize'],
                                          shuffle=True, num_workers=run_params.run['num_workers'])

validationset = data.MyDataset(validationset_name, compressed = run_params.data['compressed'],
                               augmentation = False, **run_params.MyDataset_kwargs())

validationloader = torch.utils.data.DataLoader(validationset, shuffle=False, batch_size=run_params.run['batchsize'],
                                          num_workers=run_params.run['num_workers'])

# define model and move to gpu if available
NetworkType = getattr(models, run_params.model['model_type'])
net = NetworkType(**run_params.model_kwargs())

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
optimizer = torch.optim.Adam(net.parameters(), lr=run_params.run['learning_rate_initial'],
                             betas=(run_params.run['beta1'], run_params.run['beta2']),
                             eps = run_params.run['eps'], weight_decay = run_params.run['weight_decay'],
                             amsgrad = run_params.run['amsgrad'])
scheduler = StepLR(optimizer, step_size=run_params.run['learning_rate_decay_step_size'],
                   gamma=run_params.run['learning_rate_decay'])

custom_loss = False
if run_params.run['loss_function'] == 1:
    loss_fn = torch.nn.L1Loss()
elif run_params.run['loss_function'] == 2:
    loss_fn = nn_custom.GaussianLogLikelihoodLoss(run_params.run['uncertainty_loss_eps'])
elif run_params.run['loss_function'] == 3:
    custom_loss = True
    loss_fn = nn_custom.MyLoss(**run_params.run['custom_loss_kwargs'])
else:
    loss_fn = torch.nn.MSELoss()

# save the model parameter in the beginning
run_params.save(model_dir)

print('-----------------------------------------------------------------------------')
print('INFO: Start training on device %s' % device)
print(' ')
run_params.print()
print('-----------------------------------------------------------------------------')

try:
    predict_uncertainty = run_params.model['model_args']['predict_uncertainty']
except KeyError as e:
    predict_uncertainty = False
    print('train.py: predict_uncertainty key not available, setting default value: False')

try:
    uncertainty_train_mode = run_params.model['model_args']['uncertainty_train_mode']
except KeyError as e:
    uncertainty_train_mode = 'alternating'
    print('train.py: predict_uncertainty key not available, setting default value: alternating')

# start the actual training
net = nn_custom.train_model(net, trainloader, validationloader, scheduler, optimizer,
                       loss_fn, device, run_params.run['n_epochs'],
                       run_params.run['plot_every_n_batches'], run_params.run['save_model_every_n_epoch'],
                       run_params.run['save_params_hist_every_n_epoch'], run_params.run['minibatch_epoch_loss'],
                       run_params.run['compute_validation_loss'], model_dir, args.use_writer,
                       predict_uncertainty, uncertainty_train_mode, custom_loss)

# save the model if requested
if (run_params.run['save_model']):
    torch.save(net.state_dict(), os.path.join(model_dir, 'latest.model'))

# evaluate the model performance on the testset if requested
if (run_params.run['evaluate_testset']):
    testset = utils.MyDataset(testset_name, compressed = run_params.data['compressed'],
                              augmentation = False, **run_params.MyDataset_kwargs())
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=run_params.data['num_workers'])

    with torch.no_grad():
        loss = 0.0
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            if run_params.run['loss_function']:
                loss += loss_fn(outputs, labels, inputs)
            else:
                loss += loss_fn(outputs, labels)

        print('INFO: Average loss on test set: %s' % (loss.item()/len(testset)))

# clean up the scratch folder of the cluster
if (os.path.isdir("/cluster/scratch/")):
    os.system('rm -r '  + tempfolder + '*')
