#!/usr/bin/env python

from __future__ import print_function

import os
import signal
from tensorboardX import SummaryWriter
import time
import torch
from torch.nn.functional import mse_loss
import nn_wind_prediction.utils as utils
import random

should_exit = False
sig_dict = dict((k, v) for v, k in reversed(sorted(signal.__dict__.items())) if v.startswith('SIG') and not v.startswith('SIG_'))


def signal_handler(sig, frame):
    global should_exit
    try:
        print('INFO: Received signal: ', sig_dict[sig], ', exit training loop')
    except:
        print('INFO: Received signal: ', sig, ', exit training loop')
    should_exit = True


def train_model(net, loader_trainset, loader_validationset, scheduler_lr, optimizer,
                loss_fn, device, n_epochs, plot_every_n_batches, save_model_every_n_epoch,
                save_params_hist_every_n_epoch, minibatch_loss, compute_validation_loss, log_loss_components,
                model_directory, use_writer, start_epoch=0):
    '''
    Train the model according to the specified loss function and params

    Input params:
        net:
            The network which is trained
        loader_trainset:
            Dataset loader for the train set
        loader_validationset:
            Dataset loader for the validation set
        scheduler_lr:
            Instance of a learning rate scheduler
        optimizer:
            Instance of an optimizer
        loss_fn:
            Definition of the loss function
        device:
            Device on which the tensors are stored
        n_epochs:
            Number of epochs the model is trained
        plot_every_n_batches:
            Plot the loss every n batches during one epoch, to disable set to 0
        save_model_every_n_epoch:
            Save the model parameter every nth epoch
        minibatch_loss:
            Use the minibatch loss for the epoch train loss.
            If false at the end of each epoch an addition loop over the train loss is executed
            to compute the correct loss for that epoch.
        compute_validation_loss:
            Indicates whether the validatio loss should be computed at the end of each epoch
        model_directory:
            The target directory where the model and the training log data should be stored
        use_writer:
            Indicates if the SummaryWrite should be used to log the learning curve and the gradients

    Return:
        net: The trained network
    '''

    # setup the signal handling
    global should_exit
    should_exit = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGUSR2, signal_handler)
    signal.signal(signal.SIGQUIT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if use_writer:
        # initialize the tensorboard writer
        writer = SummaryWriter(os.path.join(model_directory, 'learningcurve'))

    # start the training for n_epochs
    start_time = time.time()
    for epoch in range(start_epoch, n_epochs):  # loop over the dataset multiple times
        if should_exit:
            break
        epoch_start = time.time()

        # access to new epoch callback depends on of the model has been parallelized
        try:
            net.module.new_epoch_callback(epoch)
        except AttributeError:
            net.new_epoch_callback(epoch)

        train_loss = 0
        train_loss_components = dict.fromkeys(loss_fn.last_computed_loss_components, 0.0)
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
                train_max_uncertainty = max(train_max_uncertainty, uncertainty_exp.max().item())
                train_min_uncertainty = min(train_min_uncertainty, uncertainty_exp.min().item())

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_loss += loss.item()
            for k,v in train_loss_components.items(): train_loss_components[k]+= loss_fn.last_computed_loss_components[k]

            # print every plot_every_n_batches mini-batches
            if i % plot_every_n_batches == (plot_every_n_batches - 1) and plot_every_n_batches > 0:
                print('[%d, %5d] averaged loss: %.5f' %
                      (epoch + 1, i + 1, running_loss / (plot_every_n_batches)))
                running_loss = 0.0

        print(('[%d] Training time: %.1f s' %
               (epoch + 1, time.time() - epoch_start)))

        # save model every save_model_every_n_epoch epochs
        if (epoch % save_model_every_n_epoch == (save_model_every_n_epoch - 1)) and save_model_every_n_epoch > 0:
            try:
                state_dict = net.module.state_dict() #for when the model is trained on multi-gpu
            except AttributeError:
                state_dict = net.state_dict()

            torch.save(state_dict, os.path.join(model_directory, 'e{}.model'.format(epoch + 1)))

            # save the learnable scaling of the loss function
            if loss_fn.learn_scaling:
                torch.save(loss_fn.state_dict(), os.path.join(model_directory, 'e{}.loss'.format(epoch + 1)))

        with torch.no_grad():
            if not minibatch_loss:
                train_loss = 0.0
                train_loss_components = dict.fromkeys(loss_fn.last_computed_loss_components, 0.0)
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
                    for k,v in train_loss_components.items(): train_loss_components[k]+= loss_fn.last_computed_loss_components[k]

                    if 'logvar' in outputs.keys():
                        uncertainty_exp = outputs['logvar'].exp()
                        train_avg_uncertainty += uncertainty_exp.mean().item()
                        train_max_uncertainty = max(train_max_uncertainty, uncertainty_exp.max().item())
                        train_min_uncertainty = min(train_min_uncertainty, uncertainty_exp.min().item())

            train_loss /= len(loader_trainset)
            for k,v in train_loss_components.items(): train_loss_components[k] /= len(loader_trainset)
            train_avg_mean /= len(loader_trainset)
            train_avg_uncertainty /= len(loader_trainset)

            validation_loss = 0.0
            validation_loss_components = dict.fromkeys(loss_fn.last_computed_loss_components, 0.0)
            validation_avg_mean = 0.0
            validation_avg_uncertainty = 0.0
            validation_max_uncertainty = float('-inf')
            validation_min_uncertainty = float('inf')
            if compute_validation_loss:
                for data in loader_validationset:
                    if should_exit:
                        break

                    inputs = data[0]
                    labels = data[1]
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = net(inputs)
                    loss = loss_fn(outputs, labels, inputs)
                    validation_loss += loss.item()
                    for k, v in validation_loss_components.items(): validation_loss_components[k] += loss_fn.last_computed_loss_components[k]

                    if 'logvar' in outputs.keys():
                        uncertainty_exp = outputs['logvar'].exp()
                        validation_avg_uncertainty += uncertainty_exp.mean().item()
                        validation_max_uncertainty = max(train_max_uncertainty, uncertainty_exp.max().item())
                        validation_min_uncertainty = min(train_min_uncertainty, uncertainty_exp.min().item())

                validation_loss /= len(loader_validationset)
                for k, v in validation_loss_components.items(): validation_loss_components[k] /=len(loader_validationset)

            if use_writer and not should_exit:
                writer.add_scalar('Train/Loss', train_loss, epoch + 1)
                writer.add_scalar('Train/MeanMSELoss', train_avg_mean, epoch + 1)
                writer.add_scalar('Train/AverageUncertainty', train_avg_uncertainty, epoch + 1)
                writer.add_scalar('Train/MaxUncertainty', train_max_uncertainty, epoch + 1)
                writer.add_scalar('Train/MinUncertainty', train_min_uncertainty, epoch + 1)

                if compute_validation_loss:
                    writer.add_scalar('Val/Loss', validation_loss, epoch + 1)
                    writer.add_scalar('Val/MeanMSELoss', validation_avg_mean, epoch + 1)
                    writer.add_scalar('Val/AverageUncertainty', validation_avg_uncertainty, epoch + 1)
                    writer.add_scalar('Val/MaxUncertainty', validation_max_uncertainty, epoch + 1)
                    writer.add_scalar('Val/MinUncertainty', validation_min_uncertainty, epoch + 1)

                writer.add_scalar('Training/LearningRate', scheduler_lr.get_lr()[0], epoch + 1)

                # record components of the loss
                if log_loss_components:
                    for name, value in train_loss_components.items():
                        writer.add_scalar('Train/LC_' + name, value, epoch + 1)
                    if compute_validation_loss:
                        for name, value in validation_loss_components.items():
                            writer.add_scalar('Val/LC_' + name, value, epoch + 1)

                    # record learnable loss factors
                    for i, loss_component in enumerate(loss_fn.loss_component_names):
                        writer.add_scalar('LossFactors/'+loss_component, loss_fn.loss_factors[i], epoch + 1)

                if epoch % save_params_hist_every_n_epoch == (save_params_hist_every_n_epoch - 1):
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram(tag, value.data.cpu().numpy(), epoch + 1)
                        if value.requires_grad:
                            writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)
                    del tag, value

            print(('[%d] train loss: %.6f, validation loss: %.6f, epoch_time: %.2f s' %
                   (epoch + 1, train_loss, validation_loss, time.time() - epoch_start)))

        # adjust the learning rate if necessary
        scheduler_lr.step()

    print("INFO: Finished training in %s seconds" % (time.time() - start_time))

    if use_writer:
        #writer.add_graph(net.cpu(), inputs.cpu()) # do not save the graph by default as the visualization does not work anyway that well
        writer.close()

    return net