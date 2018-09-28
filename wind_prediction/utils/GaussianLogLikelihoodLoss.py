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
        num_channels = output.shape[1]

        if (output.shape[1] != 2 * label.shape[1]) and (num_channels % 2 != 0):
            raise ValueError('The output has to have twice the number of channels of the labels')

        output_mean = output[:,:int(num_channels/2),:]
        output_variance = output[:,int(num_channels/2):,:].exp() # todo check if it really is the channels i am splitting

        mean_error =  output_mean - label

        loss = output[:,int(num_channels/2):,:] + (mean_error * mean_error) / output_variance
#         pdb.set_trace()

        return loss.mean()
