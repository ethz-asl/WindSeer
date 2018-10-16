#!/usr/bin/env python

from __future__ import print_function

import os
from tensorboardX import SummaryWriter
import time
import torch
from torch.nn.functional import mse_loss

def train_model(net, loader_trainset, loader_validationset, scheduler_lr, optimizer,
                loss_fn, device, n_epochs, plot_every_n_batches, save_model_every_n_epoch,
                save_params_hist_every_n_epoch, minibatch_loss, compute_validation_loss,
                model_directory, use_writer, predict_uncertainty, uncertainty_train_mode):
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
        predict_uncertainty:
            Indicates if for each channel and pixel the uncertainty of the model is predicted
        uncertainty_train_mode:
            Specifies the mode in which the uncertainty is trained:
                1: Train only the mean prediction of the model
                2: Train only the uncertainty of the model
                else: Train the uncertainty and the mean alternatively per epoch

    Return:
        net: The trained network
    '''

    if use_writer:
        # initialize the tensorboard writer
        writer = SummaryWriter(os.path.join(model_directory, 'learningcurve'))

    # start the training for n_epochs
    start_time = time.time()
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        if predict_uncertainty:
            if uncertainty_train_mode == 1:
                net.freeze_uncertainty()
                net.unfreeze_mean()
            elif uncertainty_train_mode == 2:
                net.unfreeze_uncertainty()
                net.freeze_mean()
            else:
                if epoch % 2 == 0:
                    net.freeze_uncertainty()
                    net.unfreeze_mean()
                else:
                    net.unfreeze_uncertainty()
                    net.freeze_mean()

        train_loss = 0
        running_loss = 0.0
        train_avg_mean = 0.0
        train_avg_uncertainty = 0.0
        train_max_uncertainty = float('-inf')
        train_min_uncertainty = float('inf')

        # adjust the learning rate if necessary
        scheduler_lr.step()

        for i, data in enumerate(loader_trainset, 0):
            # get the inputs
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if predict_uncertainty:
                mean = net.predict_mean(inputs)
                uncertainty = net.predict_uncertainty(inputs)

                if uncertainty_train_mode == 1:
                    loss = mse_loss(mean, labels)
                else:
                    loss = loss_fn.compute_loss(mean, uncertainty, labels)

                # compute training statistics
                train_avg_mean += mse_loss(mean, labels).item()
                uncertainty_exp = uncertainty.exp()
                train_avg_uncertainty += uncertainty_exp.mean().item()
                train_max_uncertainty = max(train_max_uncertainty, uncertainty_exp.max().item())
                train_min_uncertainty = min(train_min_uncertainty, uncertainty_exp.min().item())
            else:
                outputs = net(inputs)
                loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_loss += loss.item()

            # print every plot_every_n_batches mini-batches
            if i % plot_every_n_batches == (plot_every_n_batches - 1) and plot_every_n_batches > 0:
                print('[%d, %5d] averaged loss: %.5f' %
                      (epoch + 1, i + 1, running_loss / (plot_every_n_batches)))
                running_loss = 0.0

        # save model every save_model_every_n_epoch epochs
        if (epoch % save_model_every_n_epoch == (save_model_every_n_epoch - 1)) and save_model_every_n_epoch > 0:
            torch.save(net.state_dict(), os.path.join(model_directory, 'e{}.model'.format(epoch+1)))

        with torch.no_grad():
            if not minibatch_loss:
                train_loss = 0.0
                train_avg_mean = 0.0
                train_avg_uncertainty = 0.0
                train_max_uncertainty = float('-inf')
                train_min_uncertainty = float('inf')
                for data in loader_trainset:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    if predict_uncertainty:
                        mean = net.predict_mean(inputs)
                        uncertainty = net.predict_uncertainty(inputs)

                        if uncertainty_train_mode == 1:
                            loss = mse_loss(mean, labels)
                        else:
                            loss = loss_fn.compute_loss(mean, uncertainty, labels)

                        # compute training statistics
                        train_loss += loss.item()
                        train_avg_mean += mse_loss(mean, labels).item()
                        uncertainty_exp = uncertainty.exp()
                        train_avg_uncertainty += uncertainty_exp.mean().item()
                        train_max_uncertainty = max(train_max_uncertainty, uncertainty_exp.max().item()) 
                        train_min_uncertainty = min(train_min_uncertainty, uncertainty_exp.min().item())
                    else:
                        outputs = net(inputs)
                        loss = loss_fn(outputs, labels)
                        train_loss += loss.item()

            train_loss /= len(loader_trainset)
            train_avg_mean /= len(loader_trainset)
            train_avg_uncertainty /= len(loader_trainset)

            validation_loss = 0.0
            validation_avg_mean = 0.0
            validation_avg_uncertainty = 0.0
            validation_max_uncertainty = float('-inf')
            validation_min_uncertainty = float('inf')
            if compute_validation_loss:
                for data in loader_validationset:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    if predict_uncertainty:
                        mean = net.predict_mean(inputs)
                        uncertainty = net.predict_uncertainty(inputs)

                        if uncertainty_train_mode == 1:
                            loss = mse_loss(mean, labels)
                        else:
                            loss = loss_fn.compute_loss(mean, uncertainty, labels)

                        # compute training statistics
                        validation_loss += loss.item()
                        validation_avg_mean += mse_loss(mean, labels).item()
                        uncertainty_exp = uncertainty.exp()
                        validation_avg_uncertainty += uncertainty_exp.mean().item()
                        validation_max_uncertainty = max(validation_max_uncertainty, uncertainty_exp.max().item()) 
                        validation_min_uncertainty = min(validation_min_uncertainty, uncertainty_exp.min().item())
                    else:
                        outputs = net(inputs)
                        loss = loss_fn(outputs, labels)
                        validation_loss += loss.item()
                validation_loss /= len(loader_validationset)


            if use_writer:
                writer.add_scalar('Train/Loss', train_loss, epoch+1)
                if predict_uncertainty:
                    writer.add_scalar('Train/MeanMSELoss', train_avg_mean, epoch+1)
                    writer.add_scalar('Train/AverageUncertainty', train_avg_uncertainty, epoch+1)
                    writer.add_scalar('Train/MaxUncertainty', train_max_uncertainty, epoch+1)
                    writer.add_scalar('Train/MinUncertainty', train_min_uncertainty, epoch+1)

                if compute_validation_loss:
                    writer.add_scalar('Val/Loss', validation_loss, epoch+1)
                    if predict_uncertainty:
                        writer.add_scalar('Val/MeanMSELoss', validation_avg_mean, epoch+1)
                        writer.add_scalar('Val/AverageUncertainty', validation_avg_uncertainty, epoch+1)
                        writer.add_scalar('Val/MaxUncertainty', validation_max_uncertainty, epoch+1)
                        writer.add_scalar('Val/MinUncertainty', validation_min_uncertainty, epoch+1)

                writer.add_scalar('Training/LearningRate', scheduler_lr.get_lr()[0], epoch+1)

                if epoch % save_params_hist_every_n_epoch == (save_params_hist_every_n_epoch - 1):
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram(tag, value.data.cpu().numpy(), epoch+1)
                        if value.requires_grad:
                            writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)
                    del tag, value

            print(('[%d] train loss: %.5f, validation loss: %.5f' %
                          (epoch + 1, train_loss, validation_loss)))

    print("INFO: Finished training in %s seconds" % (time.time() - start_time))

    if use_writer:
        #writer.add_graph(net.cpu(), inputs.cpu()) # do not save the graph by default as the visualization does not work anyway that well
        writer.close()

    return net
