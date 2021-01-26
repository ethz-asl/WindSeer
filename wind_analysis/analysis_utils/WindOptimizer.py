import numpy as np
import sys
import torch

from .utils import get_optimizer, predict
from nn_wind_prediction.utils.interpolation import DataInterpolation

class WindOptimizer(object):
    def __init__(self, net, loss_fn, device):
        self.net = net
        self.loss_fn = loss_fn
        self.device = device

    def run(self, generate_input_fn, params, terrain, measurements, mask, scale, config):
        optimizer = get_optimizer([params], config['optimizer'])

        nz, ny, nx = terrain.squeeze().shape
        device = measurements.device

        num_channels = len(config['model']['input_channels']) - 1
        interpolator = DataInterpolation(device, num_channels, nx, ny, nz)

        iter = 0
        loss_difference = float("Inf")
        max_gradient = float("Inf")

        losses = []
        gradients = []
        parameter = []

        if config['run']['verbose']:
            print('Starting optimization')

        if config['run']['masked_loss']:
            measurements_masked = torch.masked_select(measurements, mask[0] > 0)
        else:
            measurements_masked = measurements

        while (iter < config['run']['max_iter']) and (loss_difference > config['run']['loss_change_threshold']) and (max_gradient > config['run']['grad_threshold']):
            if config['run']['verbose']:
                sys.stdout.write("\rIteration %i" % iter)
                sys.stdout.flush()

            optimizer.zero_grad()

            input_vel = generate_input_fn(params, terrain, interpolator).to(device)
            input = torch.cat([terrain, input_vel], dim = 1)
            input_unscaled = input.clone().detach()

            prediction = predict(self.net, input, scale, config['model'])

            if config['run']['masked_loss']:
                prediction_masked = torch.masked_select(prediction['pred'], mask[0] > 0)
            else:
                prediction_masked = prediction['pred'] * mask

            loss = self.loss_fn(prediction_masked, measurements_masked)
            losses.append(loss.cpu().item())
            if len(losses) > config['run']['loss_change_window'] + 1:
                loss_difference = np.mean(np.abs(np.diff(losses[-config['run']['loss_change_window'] - 1:])))

            loss.backward()
            max_gradient = params.grad.abs().max().cpu().item()

            gradients.append(params.grad.detach().cpu().clone())

            parameter.append(params.detach().cpu().clone())

            optimizer.step()

            iter += 1

        if config['run']['verbose']:
            print('\nFinished optimization')

        return prediction, params, losses, parameter, gradients, input_unscaled
