#!/usr/bin/env python

from __future__ import print_function

import argparse
import nn_wind_prediction.data as data
import nn_wind_prediction.models as models
import nn_wind_prediction.nn as nn_custom
import nn_wind_prediction.utils as utils
import os
import re
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR


now_time = time.strftime("%Y_%m_%d-%H_%M")
t1 = time.time()
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
    trainset_name = tempfolder + 'train.hdf5'
    validationset_name = tempfolder + 'validation.hdf5'
    testset_name = tempfolder + 'test.hdf5'
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
    print('Script is running on the local machine')
    trainset_name = run_params.data['trainset_name']
    validationset_name = run_params.data['validationset_name']
    testset_name = run_params.data['testset_name']


# check if gpu is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define dataset and dataloader
if run_params.data['apply_curriculum_training']:
    trainloader = []
    validationloader = []
    for i in run_params.data['curriculum_percentages']:
        trainset = data.HDF5Dataset(trainset_name, i,
                              augmentation = run_params.data['augmentation'],
                              augmentation_mode = run_params.data['augmentation_mode'],
                              augmentation_kwargs = run_params.data['augmentation_kwargs'],
                              **run_params.Dataset_kwargs())

        trainloader += [torch.utils.data.DataLoader(trainset, batch_size=run_params.run['batchsize'],
                        shuffle=True, num_workers=run_params.run['num_workers'])]

        validationset = data.HDF5Dataset(validationset_name, i,
                        subsample=False, augmentation=False, **run_params.Dataset_kwargs())

        validationloader += [torch.utils.data.DataLoader(validationset, shuffle=False, batch_size=run_params.run['batchsize'],
                        num_workers=run_params.run['num_workers'])]
else:
    trainset = data.HDF5Dataset(trainset_name,
                                augmentation=run_params.data['augmentation'],
                                augmentation_mode=run_params.data['augmentation_mode'],
                                augmentation_kwargs=run_params.data['augmentation_kwargs'],
                                **run_params.Dataset_kwargs())

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=run_params.run['batchsize'],
                                                shuffle=True, num_workers=run_params.run['num_workers'])

    validationset = data.HDF5Dataset(validationset_name,
                                     subsample=False, augmentation=False, **run_params.Dataset_kwargs())

    validationloader = torch.utils.data.DataLoader(validationset, shuffle=False, batch_size=run_params.run['batchsize'],
                                    num_workers=run_params.run['num_workers'])


# define model
NetworkType = getattr(models, run_params.model['model_type'])

# get grid size and pass to model and loss kwargs
grid_size = data.get_grid_size(trainset_name)
run_params.model_kwargs()['grid_size'] = grid_size
run_params.pass_grid_size_to_loss(grid_size)

# initialize model
net = NetworkType(**run_params.model_kwargs())

# initialize loss function
loss_fn = nn_custom.CombinedLoss(**run_params.loss)

warm_start_epoch = 0
if (run_params.run['warm_start']):
    try:
        net.load_state_dict(torch.load(os.path.join(model_dir, 'pretrained.model'), map_location=lambda storage, loc: storage))
        if loss_fn.learn_scaling:
            loss_fn.load_state_dict(torch.load(os.path.join(model_dir, 'pretrained.loss'), map_location=lambda storage, loc: storage))
    except:
        try:
            print('Warm start warning: Failed to load pretrained.model. Using models from latest saved epoch.')

            saved_model_epochs = []
            for filename in os.listdir(model_dir):
                if filename.startswith('e') and filename.endswith('model'):
                    saved_model_epochs += [int(re.search(r'\d+', filename).group())]
            warm_start_epoch = max(saved_model_epochs)
            net.load_state_dict(torch.load(os.path.join(model_dir, 'e{}.model'.format(warm_start_epoch)),
                                           map_location=lambda storage, loc: storage))
            if loss_fn.learn_scaling:
                loss_fn.load_state_dict(torch.load(os.path.join(model_dir, 'e{}.loss'.format(warm_start_epoch)),
                                           map_location=lambda storage, loc: storage))
        except:
            print('Warm start warning: Failed to load the model parameter, initializing parameter.')
            if run_params.run['custom_init']:
                net.init_params()
else:
    if run_params.run['custom_init']:
        net.init_params()

# parallelize the data if multiple gpus can be used
if torch.cuda.device_count() > 1:
  print("Using ", torch.cuda.device_count(), " GPUs!")
  net = nn.DataParallel(net)

net.to(device)

# get model parameters to learn
param_list = [{'params': net.parameters()}]

# add the learnable parameters from the loss to the list of optimizable params if they should be learned
if loss_fn.learn_scaling:
    param_list.append({'params': loss_fn.parameters()})

# define optimizer and objective
optimizer = torch.optim.Adam(param_list, lr=run_params.run['learning_rate_initial'],
                             betas=(run_params.run['beta1'], run_params.run['beta2']),
                             eps = run_params.run['eps'], weight_decay = run_params.run['weight_decay'],
                             amsgrad = run_params.run['amsgrad'])
scheduler = StepLR(optimizer, step_size=run_params.run['learning_rate_decay_step_size'],
                   gamma=run_params.run['learning_rate_decay'])
scheduler.last_epoch = warm_start_epoch


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

# if the loss components and their respective factors should be plotted in tensorboard
try:
    log_loss_components = run_params.loss['log_loss_components']
except:
    log_loss_components = False
    print('train.py: log_loss_components key not available, setting default value: ', log_loss_components)

t2 = time.time()
print('Time until actual training: ', t2-t1, 's')
# start the actual training
net = nn_custom.train_model(net, trainloader, validationloader, scheduler, optimizer, loss_fn, device,
                       run_params.run['n_epochs'], run_params.run['plot_every_n_batches'],
                       run_params.run['save_model_every_n_epoch'], run_params.run['save_params_hist_every_n_epoch'],
                       run_params.run['minibatch_epoch_loss'],run_params.run['compute_validation_loss'],
                       run_params.data['apply_curriculum_training'],
                       log_loss_components, model_dir, args.use_writer, predict_uncertainty,
                       uncertainty_train_mode, warm_start_epoch)

# save the model if requested
if (run_params.run['save_model']):
    try:
        state_dict = net.module.state_dict() #for when the model is trained on multi-gpu
    except AttributeError:
        state_dict = net.state_dict()

    torch.save(state_dict, os.path.join(model_dir, 'latest.model'))

    if run_params.loss['learn_scaling']:
        torch.save(loss_fn.state_dict(), os.path.join(model_dir, 'latest.loss'))

# evaluate the model performance on the testset if requested
if (run_params.run['evaluate_testset']):

    testset = data.HDF5Dataset(testset_name,
                    subsample = False, augmentation = False, **run_params.Dataset_kwargs())


    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=run_params.data['num_workers'])

    with torch.no_grad():
        loss = 0.0
        for data in testloader:
            inputs = data[0]
            labels = data[1]
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss += loss_fn(outputs, labels, inputs)

        print('INFO: Average loss on test set: %s' % (loss.item()/len(testset)))

# clean up the scratch folder of the cluster
if (os.path.isdir("/cluster/scratch/")):
    os.system('rm -r '  + tempfolder + '*')
