import numpy as np
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

        while (iter < config['run']['max_iter']) and (loss_difference > config['run']['loss_change_threshold']) and (max_gradient > config['run']['grad_threshold']):
            optimizer.zero_grad()

            input_vel = generate_input_fn(params, terrain, interpolator).to(device)
            input = torch.cat([terrain, input_vel], dim = 1)

            prediction = predict(self.net, input, scale, config['model'])
            prediction_masked = prediction['pred'] * mask

            loss = self.loss_fn(prediction_masked, measurements)
            losses.append(loss.cpu().item())
            if len(losses) > config['run']['loss_change_window'] + 1:
                loss_difference = np.mean(np.abs(np.diff(losses[-config['run']['loss_change_window'] - 1:])))

            loss.backward()
            max_gradient = params.grad.abs().max().cpu().item()

            gradients.append(params.grad.detach().cpu().clone())
            
            parameter.append(params.detach().cpu().clone())

            optimizer.step()
            
            iter += 1

        return prediction, params, losses, parameter, gradients, input
