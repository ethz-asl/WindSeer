from torch.nn import Module
import torch
import torch.nn.functional as f
import nn_wind_prediction.utils as utils

class VelocityGradientLoss(Module):
    '''
    Upgraded version of Stream Function Loss according to https://arxiv.org/abs/1806.02071.
    Two component loss function in which predicted flow field is compared to target flow field and their spatial gradient
    tensors are also compared.

    Init params:
        loss_type: whether to use a L1 or a MSE based method.
        grid_size: list containing the grid spacing in directions X, Y and Z of the dataset. [m]
        grad_scaling: scaling factor used to balance the two components of the loss.
    '''
    __default_loss_method_type = 'MSE'
    __default_grid_size = [1, 1, 1]
    __default_scaling_factor = 1200 #found to work well (trial and error)

    def __init__(self, **kwargs):
        super(VelocityGradientLoss, self).__init__()
        try:
            loss_method_type = kwargs['loss_method_type']
        except KeyError:
            loss_method_type = self.__default_loss_method_type
            print('DivergenceFreeLoss: loss_method not present in kwargs, using default value:', self.__default_loss_method)

        try:
            self.__grid_size = kwargs['grid_size']
        except KeyError:
            self.__grid_size = self.__default_grid_size
            print('DivergenceFreeLoss: grid_size not present in kwargs, using default value:', self.__default_grid_size)

        try:
            self.__grad_scaling = kwargs['scaling_factor']
        except KeyError:
            self.__grad_scaling = self.__default_scaling_factor
            print('DivergenceFreeLoss: scaling_factor not present in kwargs, using default value:',
                  self.__default_scaling_factor)

        if (loss_method_type == 'MSE'):
            self.__loss = torch.nn.MSELoss()
        elif (loss_method_type == 'L1'):
            self.__loss = torch.nn.L1Loss()
        else:
            raise ValueError('Only L1 and MSE loss_type supported')

    def forward(self, net_output, target):
        if (net_output.shape != target.shape):
            raise ValueError('Prediction and target do not have the same shape')

        if (len(net_output.shape) != 5):
            if (len(net_output.shape) == 4) and net_output.shape[0]==3: # corresponds to a single sample
                net_output = net_output.unsqueeze(0)
                target = target.unsqueeze(0)
            else:
                raise ValueError('The loss is only defined for 5D data')

        return self.compute_loss(net_output, target)

    def compute_loss(self, net_output, target):

        loss = self.__loss(target, net_output)
        loss += self.__grad_scaling*self.__loss(utils.gradient(target,self.__grid_size),
                                                  utils.gradient(net_output,self.__grid_size))
        return loss