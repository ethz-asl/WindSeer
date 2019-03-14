from torch.nn import Module
import torch
import torch.nn.functional as f
import nn_wind_prediction.utils as utils

class VelocityGradientLoss(Module):
    '''
    Upgraded version of Stream Function Loss according to https://arxiv.org/abs/1806.02071
    '''

    def __init__(self, loss_type):
        super(VelocityGradientLoss, self).__init__()

        self.__loss_type = loss_type

    def forward(self, net_output, target):
        if (net_output.shape != target.shape):
            raise ValueError('Prediction and target do not have the same shape')

        if (len(net_output.shape) != 5):
            raise ValueError('The loss is only defined for 5D data')

        return self.compute_loss(net_output, target)

    def compute_loss(self, net_output, target):
        if self.__loss_type == 'L1':
            loss = f.l1_loss(target, net_output)
            loss += f.l1_loss(utils.gradient(target), utils.gradient(net_output))
        elif self.__loss_type == 'MSE':
            loss = f.mse_loss(target, net_output)
            loss += f.mse_loss(utils.gradient(target), utils.gradient(net_output))
        else:
            raise ValueError('Only L1 and MSE loss_type supported')
        return loss