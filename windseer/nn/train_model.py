#!/usr/bin/env python

from __future__ import print_function

import os
import re
import signal
from torch.utils.tensorboard import SummaryWriter
import time
import torch
from torch.nn.functional import mse_loss
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import windseer.data as ws_data
import windseer.nn.models as models
import windseer.nn as nn_custom
import windseer.utils as utils

should_exit = False
sig_dict = dict((k, v) for v, k in reversed(sorted(signal.__dict__.items()))
                if v.startswith('SIG') and not v.startswith('SIG_'))


def signal_handler(sig, frame):
    global should_exit
    try:
        print('INFO: Received signal: ', sig_dict[sig], ', exit training loop')
    except:
        print('INFO: Received signal: ', sig, ', exit training loop')
    should_exit = True


def train_model(configs, output_dir, use_writer, copy_datasets):
    '''
    Train a neural network according to the config settings

    Parameters
    ----------
    configs : WindseerParams
        Parameter class
    output_dir : str
        Output directory where the models and the training data is stored
    use_writer : bool
        If True the learning curve is stored with the SummaryWriter.
    copy_datasets : bool
        If True the datasets are copied to the folder specified with the environment variable TMPDIR
    '''
    # setup the signal handling
    global should_exit
    should_exit = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGUSR2, signal_handler)
    signal.signal(signal.SIGQUIT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # check if gpu is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_dir = os.path.join(output_dir, configs.name)
    if configs.run['save_model'] and (not os.path.exists(model_dir)):
        os.makedirs(model_dir)

    # setup dataset classes
    t_start = time.time()
    if copy_datasets and "TMPDIR" in os.environ:
        tempfolder = os.environ['TMPDIR'] + '/'
        print('Copying dataset files to', tempfolder)
        trainset_name = tempfolder + 'train.hdf5'
        validationset_name = tempfolder + 'validation.hdf5'
        os.system('cp ' + configs.data['trainset_name'] + ' ' + trainset_name)
        print("INFO: Finished copying trainset in %s seconds" % (time.time() - t_start))
        t_intermediate = time.time()
        os.system('cp ' + configs.data['validationset_name'] + ' ' + validationset_name)
        print(
            "INFO: Finished copying validationset in %s seconds" %
            (time.time() - t_intermediate)
            )

    else:
        trainset_name = configs.data['trainset_name']
        validationset_name = configs.data['validationset_name']

    # define dataset and dataloader
    trainset = ws_data.HDF5Dataset(
        trainset_name,
        augmentation=configs.data['augmentation'],
        augmentation_mode=configs.data['augmentation_mode'],
        augmentation_kwargs=configs.data['augmentation_kwargs'],
        **configs.Dataset_kwargs()
        )

    loader_trainset = torch.utils.data.DataLoader(
        trainset,
        batch_size=configs.run['batchsize'],
        shuffle=True,
        num_workers=configs.run['num_workers']
        )

    validationset = ws_data.HDF5Dataset(
        validationset_name,
        subsample=False,
        augmentation=False,
        **configs.Dataset_kwargs()
        )

    loader_validationset = torch.utils.data.DataLoader(
        validationset,
        shuffle=False,
        batch_size=configs.run['batchsize'],
        num_workers=configs.run['num_workers']
        )

    # get grid size and pass to model and loss kwargs
    grid_size = trainset.get_ds().tolist()
    configs.model_kwargs()['grid_size'] = grid_size
    configs.pass_grid_size_to_loss(grid_size)

    # initialize loss function
    loss_fn = nn_custom.CombinedLoss(**configs.loss)

    # setup neural network
    NetworkType = getattr(models, configs.model['model_type'])
    net = NetworkType(**configs.model_kwargs())

    start_epoch = 0
    if (configs.run['warm_start']):
        try:
            net.load_state_dict(
                torch.load(
                    os.path.join(model_dir, 'pretrained.model'),
                    map_location=lambda storage, loc: storage
                    )
                )
            if loss_fn.learn_scaling:
                loss_fn.load_state_dict(
                    torch.load(
                        os.path.join(model_dir, 'pretrained.loss'),
                        map_location=lambda storage, loc: storage
                        )
                    )
        except:
            try:
                print(
                    'Warm start warning: Failed to load pretrained.model. Using models from latest saved epoch.'
                    )

                saved_model_epochs = []
                for filename in os.listdir(model_dir):
                    if filename.startswith('e') and filename.endswith('model'):
                        saved_model_epochs += [int(re.search(r'\d+', filename).group())]
                start_epoch = max(saved_model_epochs)
                net.load_state_dict(
                    torch.load(
                        os.path.join(model_dir, 'e{}.model'.format(start_epoch)),
                        map_location=lambda storage, loc: storage
                        )
                    )
                if loss_fn.learn_scaling:
                    loss_fn.load_state_dict(
                        torch.load(
                            os.path.join(model_dir, 'e{}.loss'.format(start_epoch)),
                            map_location=lambda storage, loc: storage
                            )
                        )
            except:
                print(
                    'Warm start warning: Failed to load the model parameter, initializing parameter.'
                    )
                if configs.run['custom_init']:
                    net.init_params()
    else:
        if configs.run['custom_init']:
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
    if configs.run['optimizer_type'] == 'ranger':
        from ranger import Ranger
        optimizer = Ranger(param_list, **configs.run['optimizer_kwargs'])
    elif configs.run['optimizer_type'] == 'radam':
        from radam import RAdam
        optimizer = RAdam(param_list, **configs.run['optimizer_kwargs'])

    elif configs.run['optimizer_type'] == 'adam':
        optimizer = torch.optim.Adam(param_list, **configs.run['optimizer_kwargs'])
    else:
        print('Invalid optimizer_type, defaulting to Adam')
        optimizer = torch.optim.Adam(param_list, **configs.run['optimizer_kwargs'])

    scheduler_lr = StepLR(
        optimizer,
        step_size=configs.run['learning_rate_decay_step_size'],
        gamma=configs.run['learning_rate_decay']
        )

    if start_epoch > 0:
        for i in range(start_epoch):
            scheduler_lr.step()

    # save the model parameter in the beginning
    configs.save(model_dir)

    print(
        '-----------------------------------------------------------------------------'
        )
    print('INFO: Start training on device %s' % device)
    print(' ')
    configs.print()
    print(
        '-----------------------------------------------------------------------------'
        )

    # if the loss components and their respective factors should be plotted in tensorboard
    try:
        log_loss_components = configs.loss['log_loss_components']
    except:
        log_loss_components = False
        print(
            'train.py: log_loss_components key not available, setting default value: ',
            log_loss_components
            )

    if use_writer:
        # initialize the tensorboard writer
        writer = SummaryWriter(os.path.join(model_dir, 'learningcurve'))

    print('Setup time: ', time.time() - t_start, 's')

    # start the training for n_epochs
    start_time = time.time()
    for epoch in range(
        start_epoch, configs.run['n_epochs']
        ):  # loop over the dataset multiple times
        if should_exit:
            break
        epoch_start = time.time()

        # access to new epoch callback depends on of the model has been parallelized
        try:
            net.module.new_epoch_callback(epoch)
        except AttributeError:
            net.new_epoch_callback(epoch)

        train_loss = 0
        train_loss_components = dict.fromkeys(
            loss_fn.last_computed_loss_components, 0.0
            )
        running_loss = 0.0
        train_avg_mean = 0.0
        train_avg_uncertainty = 0.0
        train_max_uncertainty = float('-inf')
        train_min_uncertainty = float('inf')

        for i, data in enumerate(loader_trainset, 0):
            if should_exit:
                break

            # get the inputs, labels and loss weights
            inputs = data[0]
            labels = data[1]
            W = data[2]
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_fn(outputs, labels, inputs, W)

            if 'logvar' in outputs.keys():
                uncertainty_exp = outputs['logvar'].exp()
                train_avg_uncertainty += uncertainty_exp.mean().item()
                train_max_uncertainty = max(
                    train_max_uncertainty,
                    uncertainty_exp.max().item()
                    )
                train_min_uncertainty = min(
                    train_min_uncertainty,
                    uncertainty_exp.min().item()
                    )

            del outputs, inputs, labels, W

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_loss += loss.item()
            for k, v in train_loss_components.items():
                train_loss_components[k] += loss_fn.last_computed_loss_components[k]

            # print every plot_every_n_batches mini-batches
            if i % configs.run['plot_every_n_batches'] == (
                configs.run['plot_every_n_batches'] - 1
                ) and configs.run['plot_every_n_batches'] > 0:
                print(
                    '[%d, %5d] averaged loss: %.5f' % (
                        epoch + 1, i + 1, running_loss /
                        (configs.run['plot_every_n_batches'])
                        )
                    )
                running_loss = 0.0

        print(('[%d] Training time: %.1f s' % (epoch + 1, time.time() - epoch_start)))

        # save model every save_model_every_n_epoch epochs
        if (
            epoch % configs.run['save_model_every_n_epoch']
            == (configs.run['save_model_every_n_epoch'] - 1)
            ) and configs.run['save_model_every_n_epoch'] > 0:
            try:
                state_dict = net.module.state_dict(
                )  #for when the model is trained on multi-gpu
            except AttributeError:
                state_dict = net.state_dict()

            torch.save(
                state_dict, os.path.join(model_dir, 'e{}.model'.format(epoch + 1))
                )

            # save the learnable scaling of the loss function
            if loss_fn.learn_scaling:
                torch.save(
                    loss_fn.state_dict(),
                    os.path.join(model_dir, 'e{}.loss'.format(epoch + 1))
                    )

        with torch.no_grad():
            if not configs.run['minibatch_epoch_loss']:
                train_loss = 0.0
                train_loss_components = dict.fromkeys(
                    loss_fn.last_computed_loss_components, 0.0
                    )
                train_avg_mean = 0.0
                train_avg_uncertainty = 0.0
                train_max_uncertainty = float('-inf')
                train_min_uncertainty = float('inf')
                for data in loader_trainset:
                    if should_exit:
                        break

                    inputs = data[0]
                    labels = data[1]
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = net(inputs)
                    # mean of loss_fn is taken in case the loss is parallelized over multiple-gpus
                    loss = loss_fn(outputs, labels, inputs)
                    train_loss += loss.item()
                    for k, v in train_loss_components.items():
                        train_loss_components[
                            k] += loss_fn.last_computed_loss_components[k]

                    if 'logvar' in outputs.keys():
                        uncertainty_exp = outputs['logvar'].exp()
                        train_avg_uncertainty += uncertainty_exp.mean().item()
                        train_max_uncertainty = max(
                            train_max_uncertainty,
                            uncertainty_exp.max().item()
                            )
                        train_min_uncertainty = min(
                            train_min_uncertainty,
                            uncertainty_exp.min().item()
                            )

            train_loss /= len(loader_trainset)
            for k, v in train_loss_components.items():
                train_loss_components[k] /= len(loader_trainset)
            train_avg_mean /= len(loader_trainset)
            train_avg_uncertainty /= len(loader_trainset)

            validation_loss = 0.0
            validation_loss_components = dict.fromkeys(
                loss_fn.last_computed_loss_components, 0.0
                )
            validation_avg_mean = 0.0
            validation_avg_uncertainty = 0.0
            validation_max_uncertainty = float('-inf')
            validation_min_uncertainty = float('inf')
            compute_validation_loss_this_epoch = (
                epoch % configs.run['compute_validation_loss_every_n_epochs']
                == (configs.run['compute_validation_loss_every_n_epochs'] - 1)
                ) and configs.run['compute_validation_loss']
            if compute_validation_loss_this_epoch:
                for data in loader_validationset:
                    if should_exit:
                        break

                    inputs = data[0]
                    labels = data[1]
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = net(inputs)
                    loss = loss_fn(outputs, labels, inputs)
                    validation_loss += loss.item()
                    for k, v in validation_loss_components.items():
                        validation_loss_components[
                            k] += loss_fn.last_computed_loss_components[k]

                    if 'logvar' in outputs.keys():
                        uncertainty_exp = outputs['logvar'].exp()
                        validation_avg_uncertainty += uncertainty_exp.mean().item()
                        validation_max_uncertainty = max(
                            train_max_uncertainty,
                            uncertainty_exp.max().item()
                            )
                        validation_min_uncertainty = min(
                            train_min_uncertainty,
                            uncertainty_exp.min().item()
                            )

                validation_loss /= len(loader_validationset)
                for k, v in validation_loss_components.items():
                    validation_loss_components[k] /= len(loader_validationset)

            if use_writer and not should_exit:
                writer.add_scalar('Train/Loss', train_loss, epoch + 1)
                writer.add_scalar('Train/MeanMSELoss', train_avg_mean, epoch + 1)
                writer.add_scalar(
                    'Train/AverageUncertainty', train_avg_uncertainty, epoch + 1
                    )
                writer.add_scalar(
                    'Train/MaxUncertainty', train_max_uncertainty, epoch + 1
                    )
                writer.add_scalar(
                    'Train/MinUncertainty', train_min_uncertainty, epoch + 1
                    )

                if compute_validation_loss_this_epoch:
                    writer.add_scalar('Val/Loss', validation_loss, epoch + 1)
                    writer.add_scalar('Val/MeanMSELoss', validation_avg_mean, epoch + 1)
                    writer.add_scalar(
                        'Val/AverageUncertainty', validation_avg_uncertainty, epoch + 1
                        )
                    writer.add_scalar(
                        'Val/MaxUncertainty', validation_max_uncertainty, epoch + 1
                        )
                    writer.add_scalar(
                        'Val/MinUncertainty', validation_min_uncertainty, epoch + 1
                        )

                writer.add_scalar(
                    'Training/LearningRate',
                    scheduler_lr.get_last_lr()[0], epoch + 1
                    )

                # record components of the loss
                if log_loss_components:
                    for name, value in train_loss_components.items():
                        writer.add_scalar('Train/LC_' + name, value, epoch + 1)
                    if compute_validation_loss_this_epoch:
                        for name, value in validation_loss_components.items():
                            writer.add_scalar('Val/LC_' + name, value, epoch + 1)

                    # record learnable loss factors
                    for i, loss_component in enumerate(loss_fn.loss_component_names):
                        writer.add_scalar(
                            'LossFactors/' + loss_component, loss_fn.loss_factors[i],
                            epoch + 1
                            )

                if epoch % configs.run['save_params_hist_every_n_epoch'] == (
                    configs.run['save_params_hist_every_n_epoch'] - 1
                    ):
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram(tag, value.data.cpu().numpy(), epoch + 1)
                        if value.requires_grad:
                            writer.add_histogram(
                                tag + '/grad',
                                value.grad.data.cpu().numpy(), epoch + 1
                                )
                    del tag, value

            print((
                '[%d] train loss: %.6f, validation loss: %.6f, epoch_time: %.2f s' %
                (epoch + 1, train_loss, validation_loss, time.time() - epoch_start)
                ))

        # adjust the learning rate if necessary
        scheduler_lr.step()

        # adjust the eps in the loss if set
        loss_fn.step()

    print("INFO: Finished training in %s seconds" % (time.time() - start_time))

    if use_writer:
        #writer.add_graph(net.cpu(), inputs.cpu()) # do not save the graph by default as the visualization does not work anyway that well
        writer.close()

    # save the model if requested
    if (configs.run['save_model']):
        try:
            state_dict = net.module.state_dict(
            )  #for when the model is trained on multi-gpu
        except AttributeError:
            state_dict = net.state_dict()

        torch.save(state_dict, os.path.join(model_dir, 'latest.model'))

        if configs.loss['learn_scaling']:
            torch.save(loss_fn.state_dict(), os.path.join(model_dir, 'latest.loss'))

    trainset.print_dataset_stats()

    # clean up the scratch folder of the cluster
    if copy_datasets and "TMPDIR" in os.environ:
        os.system('rm ' + trainset_name)
        os.system('rm ' + validationset_name)
