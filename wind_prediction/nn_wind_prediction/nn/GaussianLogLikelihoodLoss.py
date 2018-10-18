import torch
from torch.nn import Module

class GaussianLogLikelihoodLoss(Module):
    '''
    Gaussian Log Likelihood Loss according to https://arxiv.org/pdf/1705.07115.pdf
    '''
    def __init__(self, eps = 1e-8):
        super(GaussianLogLikelihoodLoss, self).__init__()

        self.__eps = eps

    def forward(self, output, label):
        num_channels = output.shape[1]

        if (output.shape[1] != 2 * label.shape[1]) and (num_channels % 2 != 0):
            raise ValueError('The output has to have twice the number of channels of the labels')

        return self.compute_loss(output[:,:int(num_channels/2),:], output[:,int(num_channels/2):,:], label)

    def compute_loss(self, mean, log_variance, label):
        if (mean.shape[1] != log_variance.shape[1]):
            raise ValueError('The variance and the mean need to have the same number of channels')

        mean_error =  mean - label

        loss = log_variance + (mean_error * mean_error) / log_variance.exp().clamp(self.__eps)

        return loss.mean()
