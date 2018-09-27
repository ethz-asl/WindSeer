import torch
from torch.nn import Module
from torch.nn.functional import mse_loss


class GaussianLogLikelihoodLoss(Module):
    '''
    Gaussian Log Likelihood Loss according to https://arxiv.org/pdf/1705.07115.pdf
    '''
    def __init__(self):
        super(GaussianLogLikelihoodLoss, self).__init__()

    def forward(self, output, label):
        if (output.shape[1] - 1 - label.shape[1] != 0):
            raise ValueError('The output has to have one channel more than the label')

        return output[:,-1,:].mean() + 0.5 * (mse_loss(output[:,:-1,:], label, reduce=False).mean(1) / torch.exp(output[:,-1,:])).mean()
