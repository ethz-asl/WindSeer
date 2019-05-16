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
    __default_loss_type = 'MSE'
    __default_grid_size = [1, 1, 1]
    __default_scaling_factor = 1200 #found to work well (trial and error)
    __default_exclude_terrain = True

    def __init__(self, **kwargs):
        super(VelocityGradientLoss, self).__init__()
        try:
            self.__loss_type = kwargs['loss_type']
        except KeyError:
            self.__loss_type = self.__default_loss_type
            print('VelocityGradientLoss: loss_type not present in kwargs, using default value:', self.__default_loss_type)

        try:
            self.__grid_size = kwargs['grid_size']
        except KeyError:
            self.__grid_size = self.__default_grid_size
            print('VelocityGradientLoss: grid_size not present in kwargs, using default value:', self.__default_grid_size)

        try:
            self.__grad_scaling = kwargs['scaling_factor']
        except KeyError:
            self.__grad_scaling = self.__default_scaling_factor
            print('VelocityGradientLoss: scaling_factor not present in kwargs, using default value:',
                  self.__default_scaling_factor)
        try:
           self.__exclude_terrain =  kwargs['exclude_terrain']
        except KeyError:
            self.__exclude_terrain = self.__default_exclude_terrain
            print('VelocityGradientLoss: exclude_terrain not present in kwargs, using default value:',
                  self.__default_exclude_terrain)

        if (self.__loss_type == 'MSE'):
            self.__loss = torch.nn.MSELoss(reduction='none')
        elif (self.__loss_type == 'L1'):
            self.__loss = torch.nn.L1Loss(reduction='none')
        else:
            raise ValueError('VelocityGradientLoss: unknown loss type ', self.__loss_type)

    def forward(self, predicted, target, input):
        if (predicted.shape != target.shape):
            raise ValueError('VelocityGradientLoss: predicted and target do not have the same shape, pred:{}, target:{}'
                             .format(predicted.shape, target.shape))

        if (len(predicted.shape) != 5):
            raise ValueError('VelocityGradientLoss: the loss is only defined for 5D data. Unsqueeze single samples!')

        return self.compute_loss(predicted, target, input)

    def compute_loss(self, predicted, target, input):
        # compute gradients of prediction and target
        target_grad = utils.gradient(target,self.__grid_size)
        predicted_grad = utils.gradient(predicted,self.__grid_size)

        # compute terrain correction factor for each sample in batch
        if self.__exclude_terrain:
            terrain = input[:, 0:1]
            terrain_correction_factors = utils.compute_terrain_factor(predicted, terrain)
        else:
            terrain_correction_factors = torch.ones(predicted.shape[0]).to(predicted.device)

        # compute pixel loss for all elements and average losses over each sample in batch
        loss = self.__loss(target, predicted).mean(tuple(range(1, len(predicted.shape))))

        # add scaled mean physics loss for each samples
        loss += self.__grad_scaling * self.__loss(predicted_grad, target_grad).mean(tuple(range(1, len(predicted.shape))))

        # apply terrain correction factor to loss of each sample in batch
        loss *= terrain_correction_factors

        # return batchwise mean of loss
        return loss.mean()