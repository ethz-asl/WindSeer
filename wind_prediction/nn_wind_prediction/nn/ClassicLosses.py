from torch.nn import Module
import torch
import nn_wind_prediction.utils as utils


class MSELoss(Module):
    '''
    Modified version of the default MSE loss, where terrain data can be masked out in loss computation.
    '''
    __default_exclude_terrain = True

    def __init__(self, **kwargs):
        super(MSELoss, self).__init__()

        self.__loss = torch.nn.MSELoss()

        try:
            self.__exclude_terrain = kwargs['exclude_terrain']
        except KeyError:
            self.__exclude_terrain = self.__default_exclude_terrain
            print('DivergenceFreeLoss: exclude_terrain not present in kwargs, using default value:',
                  self.__default_exclude_terrain)

    def forward(self, net_output, target, input):

        return self.compute_loss(net_output, target, input)

    def compute_loss(self, net_output, target, input):

        # compute terrain correction factor if exclude_terrain
        terrain_correction_factor = 1
        if self.__exclude_terrain:
            terrain = input[:, 0]
            terrain_correction_factor = utils.compute_terrain_factor(net_output,terrain)

        loss = self.__loss(target, net_output)*terrain_correction_factor
        return loss

class L1Loss(Module):
    '''
    Modified version of the default L1 loss, where terrain data can be masked out in loss computation.
    '''
    __default_exclude_terrain = True

    def __init__(self, **kwargs):
        super(L1Loss, self).__init__()

        self.__loss = torch.nn.L1Loss()

        try:
            self.__exclude_terrain = kwargs['exclude_terrain']
        except KeyError:
            self.__exclude_terrain = self.__default_exclude_terrain
            print('DivergenceFreeLoss: exclude_terrain not present in kwargs, using default value:',
                  self.__default_exclude_terrain)

    def forward(self, net_output, target, input):

        return self.compute_loss(net_output, target, input)

    def compute_loss(self, net_output, target, input):

        # compute terrain correction factor if exclude_terrain
        terrain_correction_factor = 1
        if self.__exclude_terrain:
            terrain = input[:, 0]
            terrain_correction_factor = utils.compute_terrain_factor(net_output,terrain)

        loss = self.__loss(target, net_output)*terrain_correction_factor
        return loss