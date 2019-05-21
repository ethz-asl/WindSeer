from torch.nn import Module
import torch
import torch.nn.functional as f
import nn_wind_prediction.utils as utils

class DivergenceFreeLoss(Module):
    '''
    Two component loss function in which predicted flow field is compared to target flow field and predicted divergence
    is compared to 0 divergence.

    Init params:
        loss_type: whether to use a L1 or a MSE based method.
        grid_size: list containing the grid spacing in directions X, Y and Z of the dataset. [m]
        div_scaling: scaling factor used to balance the two components of the loss.
    '''
    __default_loss_type = 'MSE'
    __default_grid_size = [1, 1, 1]
    __default_scaling_factor = 500 #found to work well (trial and error)
    __default_exclude_terrain = True

    def __init__(self, **kwargs):
        super(DivergenceFreeLoss, self).__init__()
        try:
            self.__loss_type = kwargs['loss_type']
        except KeyError:
            self.__loss_type = self.__default_loss_type
            print('DivergenceFreeLoss: loss_type not present in kwargs, using default value:', self.__default_loss_type)

        try:
            self.__grid_size = kwargs['grid_size']
        except KeyError:
            self.__grid_size = self.__default_grid_size
            print('DivergenceFreeLoss: grid_size not present in kwargs, using default value:', self.__default_grid_size)

        try:
            self.__div_scaling = kwargs['scaling_factor']
        except KeyError:
            self.__div_scaling = self.__default_scaling_factor
            print('DivergenceFreeLoss: scaling_factor not present in kwargs, using default value:',
                  self.__default_scaling_factor)

        try:
           self.__exclude_terrain =  kwargs['exclude_terrain']
        except KeyError:
            self.__exclude_terrain = self.__default_exclude_terrain
            print('DivergenceFreeLoss: exclude_terrain not present in kwargs, using default value:',
                  self.__default_exclude_terrain)

        if (self.__loss_type == 'MSE'):
            self.__loss = torch.nn.MSELoss(reduction='none')
        elif (self.__loss_type == 'L1'):
            self.__loss = torch.nn.L1Loss(reduction='none')
        else:
            raise ValueError('DivergenceFreeLoss: unknown loss type ', self.__loss_type)


    def forward(self, predicted, target, input):
        if (predicted.shape != target.shape):
            raise ValueError('DivergenceFreeLoss: predicted and target do not have the same shape, pred:{}, target:{}'
                    .format(predicted.shape, target.shape))

        if (len(predicted.shape) != 5):
            raise ValueError('DivergenceFreeLoss: the loss is only defined for 5D data. Unsqueeze single samples!')

        return self.compute_loss(predicted, target, input)

    def compute_loss(self, predicted, target, input):
        # compute divergence of the prediction
        predicted_divergence = utils.divergence(predicted, self.__grid_size).unsqueeze(1)

        # compute terrain correction factor for each sample in batch
        if self.__exclude_terrain:
            terrain = input[:, 0:1]
            terrain_correction_factors = utils.compute_terrain_factor(predicted, terrain)
        else:
            terrain_correction_factors = torch.ones(predicted.shape[0]).to(predicted.device)

        # # compute pixel loss for all elements and average losses over each sample in batch
        loss = self.__loss(target, predicted).mean(tuple(range(1, len(predicted.shape))))

        # add scaled mean physics loss for each samples
        loss += self.__div_scaling * self.__loss(predicted_divergence,torch.zeros_like(predicted_divergence))\
                                                                    .mean(tuple(range(1, len(predicted.shape))))

        # apply terrain correction factor to loss of each sample in batch
        loss *= terrain_correction_factors

        # return batchwise mean of loss
        return loss.mean()