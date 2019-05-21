from torch.nn import Module
import torch
import nn_wind_prediction.utils as utils


class MSELoss(Module):
    '''
    Modified version of the default MSE loss, where a correction factor can be applied to account for the amount of
    terrain data in the samples.
    '''
    __default_exclude_terrain = True

    def __init__(self, **kwargs):
        super(MSELoss, self).__init__()

        self.__loss = torch.nn.MSELoss(reduction='none')

        try:
            self.__exclude_terrain = kwargs['exclude_terrain']
        except KeyError:
            self.__exclude_terrain = self.__default_exclude_terrain
            print('MSELoss: exclude_terrain not present in kwargs, using default value:',
                  self.__default_exclude_terrain)

    def forward(self, predicted, target, input):
        if (predicted.shape != target.shape):
            raise ValueError('MSELoss: predicted and target do not have the same shape, pred:{}, target:{}'
                             .format(predicted.shape, target.shape))

        if (len(predicted.shape) != 5) or (len(input.shape) != 5):
            raise ValueError('MSELoss: the loss is only defined for 5D data. Unsqueeze single samples!')

        return self.compute_loss(predicted, target, input)

    def compute_loss(self, predicted, target, input):

        # first compute mean loss for each sample in batch
        loss = self.__loss(target, predicted).mean(tuple(range(1, len(predicted.shape))))

        # compute terrain correction factor for each sample in batch
        if self.__exclude_terrain:
            terrain = input[:, 0:1]
            terrain_correction_factors = utils.compute_terrain_factor(predicted, terrain)
        else:
            terrain_correction_factors = torch.ones(predicted.shape[0]).to(predicted.device)

        # apply terrain correction factor to loss of each sample in batch
        loss *= terrain_correction_factors

        # return batchwise mean of loss
        return loss.mean()

class L1Loss(Module):
    '''
    Modified version of the default L1 loss, where a correction factor can be applied to account for the amount of
    terrain data in the samples.
    '''
    __default_exclude_terrain = True

    def __init__(self, **kwargs):
        super(L1Loss, self).__init__()

        self.__loss = torch.nn.L1Loss(reduction='none')

        try:
            self.__exclude_terrain = kwargs['exclude_terrain']
        except KeyError:
            self.__exclude_terrain = self.__default_exclude_terrain
            print('L1Loss: exclude_terrain not present in kwargs, using default value:',
                  self.__default_exclude_terrain)

    def forward(self, predicted, target, input):
        if (predicted.shape != target.shape):
            raise ValueError('L1Loss: predicted and target do not have the same shape, pred:{}, target:{}'
                             .format(predicted.shape, target.shape))

        if (len(predicted.shape) != 5) or (len(input.shape) != 5):
            raise ValueError('L1Loss: the loss is only defined for 5D data. Unsqueeze single samples!')

        return self.compute_loss(predicted, target, input)

    def compute_loss(self, predicted, target, input):

        # first compute mean loss for each sample in batch
        loss = self.__loss(target, predicted).mean(tuple(range(1, len(predicted.shape))))

        # compute terrain correction factor for each sample in batch
        if self.__exclude_terrain:
            terrain = input[:, 0:1]
            terrain_correction_factors = utils.compute_terrain_factor(predicted, terrain)
        else:
            terrain_correction_factors = torch.ones(predicted.shape[0]).to(predicted.device)

        # apply terrain correction factor to loss of each sample in batch
        loss *= terrain_correction_factors

        # return batchwise mean of loss
        return loss.mean()