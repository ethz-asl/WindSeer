from torch.nn import Module
import torch
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
    __default_loss_method = 'MSE'
    __default_grid_size = [1, 1, 1]
    __default_scaling_factor = 1200 #found to work well (trial and error)
    __default_exclude_terrain = True

    def __init__(self, **kwargs):
        super(VelocityGradientLoss, self).__init__()
        try:
            self.__loss_method = kwargs['loss_method']
        except KeyError:
            self.__loss_method = self.__default_loss_method
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
        try:
           self.__exclude_terrain =  kwargs['exclude_terrain']
        except KeyError:
            self.__exclude_terrain = self.__default_exclude_terrain
            print('DivergenceFreeLoss: exclude_terrain not present in kwargs, using default value:',
                  self.__default_exclude_terrain)

        if (self.__loss_method == 'MSE'):
            self.__loss = torch.nn.MSELoss()
        elif (self.__loss_method == 'L1'):
            self.__loss = torch.nn.L1Loss()
        else:
            raise ValueError('Unknown loss type: ', self.__loss_method)

    def forward(self, net_output, target, input):
        if (net_output.shape != target.shape):
            raise ValueError('Prediction and target do not have the same shape')

        if (len(net_output.shape) != 5):
            raise ValueError('The loss is only defined for 5D data')

        return self.compute_loss(net_output, target, input)

    def compute_loss(self, net_output, target, input):
        target_grad = utils.gradient(target,self.__grid_size)
        net_output_grad = utils.gradient(net_output,self.__grid_size)

        # compute terrain correction factor if exclude_terrain
        terrain_correction_factor = 1
        if self.__exclude_terrain:
            terrain = input[:, 0]
            terrain_correction_factor = utils.compute_terrain_factor(net_output, terrain)

        loss = self.__loss(target, net_output)
        loss += self.__grad_scaling*self.__loss(net_output_grad, target_grad)
        return loss*terrain_correction_factor