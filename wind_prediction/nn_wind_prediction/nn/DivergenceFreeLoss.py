from torch.nn import Module
import torch
import torch.nn.functional as f
import nn_wind_prediction.utils as utils

class DivergenceFreeLoss(Module):
    '''
    Loss function which forces the target field to be divergence free.
    '''
    def __init__(self, loss_type, grid_size):
        super(DivergenceFreeLoss, self).__init__()

        self.__loss_type = loss_type
        self.__grid_size = grid_size

    def forward(self, net_output, target):
        if (net_output.shape != target.shape):
            raise ValueError('Prediction and target do not have the same shape')

        if (len(net_output.shape) != 5):
            raise ValueError('The loss is only defined for 5D data')

        return self.compute_loss(net_output, target)

    def compute_loss(self, net_output, target):
        if self.__loss_type == 'L1':
            loss = f.l1_loss(target, net_output)
            loss += f.l1_loss(utils.divergence_(net_output, self.__grid_size), utils.divergence_(net_output * 0, self.__grid_size))
        elif self.__loss_type == 'MSE':
            loss = f.mse_loss(target, net_output)
            loss += f.mse_loss(utils.divergence_(net_output, self.__grid_size), utils.divergence_(net_output * 0, self.__grid_size))
        else:
            raise ValueError('Only L1 and MSE loss_type supported')
        return loss