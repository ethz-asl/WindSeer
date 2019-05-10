from torch.nn import Module
import torch
from nn_wind_prediction.utils import remove_terrain_data


class MSELoss(Module):
    '''
    Modified version of the default MSE loss, where terrain data can be masked out in loss computation.
    '''
    def __init__(self):
        super(MSELoss, self).__init__()

        self.__loss = torch.nn.MSELoss()

    def forward(self, net_output, target, terrain=None):

        return self.compute_loss(net_output, target, terrain)

    def compute_loss(self, net_output, target, terrain):

        # remove data in terrain
        if terrain is not None:
            target = remove_terrain_data(target, terrain)
            net_output = remove_terrain_data(net_output, terrain)

        loss = self.__loss(target, net_output)
        return loss

class L1Loss(Module):
    '''
    Modified version of the default L1 loss, where terrain data can be masked out in loss computation.
    '''
    def __init__(self):
        super(L1Loss, self).__init__()

        self.__loss = torch.nn.L1Loss()

    def forward(self, net_output, target, terrain=None):

        return self.compute_loss(net_output, target, terrain)

    def compute_loss(self, net_output, target, terrain):

        # remove data in terrain
        if terrain is not None:
            target = remove_terrain_data(target, terrain)
            net_output = remove_terrain_data(net_output, terrain)

        loss = self.__loss(target, net_output)
        return loss