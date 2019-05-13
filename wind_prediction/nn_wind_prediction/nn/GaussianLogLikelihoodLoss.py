import torch
from torch.nn import Module
import nn_wind_prediction.utils as utils

class GaussianLogLikelihoodLoss(Module):
    '''
    Gaussian Log Likelihood Loss according to https://arxiv.org/pdf/1705.07115.pdf
    '''

    __default_eps = 1e-8
    __default_exclude_terrain = True

    def __init__(self, **kwargs):
        super(GaussianLogLikelihoodLoss, self).__init__()

        try:
            self.__eps = kwargs['uncertainty_loss_eps']
        except KeyError:
            self.__eps = self.__default_eps
            print('GaussianLogLikelihoodLoss: uncertainty_loss_eps not present in kwargs, using default value:', self.__eps)

        try:
           self.__exclude_terrain =  kwargs['exclude_terrain']
        except KeyError:
            self.__exclude_terrain = self.__default_exclude_terrain
            print('DivergenceFreeLoss: exclude_terrain not present in kwargs, using default value:',
                  self.__default_exclude_terrain)

    def forward(self, output, label, input):
        num_channels = output.shape[1]

        if (output.shape[1] != 2 * label.shape[1]) and (num_channels % 2 != 0):
            raise ValueError('The output has to have twice the number of channels of the labels')

        return self.compute_loss(output[:,:int(num_channels/2),:], output[:,int(num_channels/2):,:], label, input)

    def compute_loss(self, mean, log_variance, label, input):
        if (mean.shape[1] != log_variance.shape[1]):
            raise ValueError('The variance and the mean need to have the same number of channels')

        mean_error =  mean - label

        # compute terrain correction factor if exclude_terrain
        terrain_correction_factor = 1
        if self.__exclude_terrain:
            terrain = input[:, 0]
            terrain_correction_factor = utils.compute_terrain_factor(label,terrain)

        loss = log_variance + (mean_error * mean_error) / log_variance.exp().clamp(self.__eps)

        return loss.mean()*terrain_correction_factor
