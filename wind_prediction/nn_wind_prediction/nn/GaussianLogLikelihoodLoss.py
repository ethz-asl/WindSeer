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

        # compute terrain correction factor for each sample in batch
        if self.__exclude_terrain:
            terrain = input[:, 0:1]
            terrain_correction_factors = utils.compute_terrain_factor(mean_error, terrain)
        else:
            terrain_correction_factors = torch.ones(mean_error.shape[0]).to(mean_error.device)

        # compute loss for all elements
        loss = log_variance + (mean_error * mean_error) / log_variance.exp().clamp(self.__eps)

        # average loss over each sample in batch
        loss = loss.mean(tuple(range(1, len(mean_error.shape))))

        # apply terrain correction factor to loss of each sample in batch
        loss *= terrain_correction_factors

        # return batchwise mean of loss
        return loss.mean()
