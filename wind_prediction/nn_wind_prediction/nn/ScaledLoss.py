import numpy as np
import torch
from torch.nn import Module
import torch.nn.functional as f

class ScaledLoss(Module):
    __default_loss_type = 'MSE'
    __default_max_scale = 4.0
    __default_norm_threshold = 0.5
    __default_exclude_terrain = True
    __default_no_scaling = False

    def __init__(self, **kwargs):
        super(ScaledLoss, self).__init__()

        try:
            self.__loss_type = kwargs['loss_type']
        except KeyError:
            self.__loss_type = self.__default_loss_type
            print('ScaledLoss: loss_type not present in kwargs, using default value:', self.__default_loss_type)

        try:
            self.__max_scale = kwargs['max_scale']
        except KeyError:
            self.__max_scale = self.__default_max_scale
            print('ScaledLoss: max_scale not present in kwargs, using default value:', self.__default_max_scale)

        try:
            self.__norm_threshold = kwargs['norm_threshold']
        except KeyError:
            self.__norm_threshold = self.__default_norm_threshold
            print('ScaledLoss: norm_threshold not present in kwargs, using default value:', self.__default_norm_threshold)

        try:
            self.__exclude_terrain = kwargs['exclude_terrain']
        except KeyError:
            self.__exclude_terrain = self.__default_exclude_terrain
            print('ScaledLoss: exclude_terrain not present in kwargs, using default value:', self.__default_exclude_terrain)

        try:
            self.__no_scaling = kwargs['no_scaling']
        except KeyError:
            self.__no_scaling = self.__default_no_scaling
            print('ScaledLoss: no_scaling not present in kwargs, using default value:', self.__default_no_scaling)


        if (self.__loss_type == 'MSE'):
            self.__loss = torch.nn.MSELoss(reduction='none')
        elif (self.__loss_type == 'L1'):
            self.__loss = torch.nn.L1Loss(reduction='none')
        else:
            raise ValueError('Unknown loss type: ', self.__loss_type)

        # input sanity checks
        if (self.__max_scale < 1.0):
            raise ValueError('max_scale must be larger than 1.0')

        if (self.__norm_threshold <= 0.0):
            raise ValueError('max_scale must be larger than 0.0')

    def forward(self, prediction, label, input):
        # compute the loss per pixel
        loss = self.__loss(prediction, label).mean(dim=1)

        if self.__no_scaling:
            scale = torch.ones_like(loss)

        else:
            # compute the scale
            scale = label[:,0:3]-input[:,1:4]
            scale = (scale.norm(dim=1) / label[:,0:3].norm(dim=1))

            # set nans to 0.0
            scale = torch.where(torch.isnan(scale), torch.zeros_like(scale), scale)

            # map to a range of 1.0 to max_scale
            scale = scale.clamp(min=0.0, max=self.__norm_threshold) * (1.0 / self.__norm_threshold) * (self.__max_scale - 1.0) + 1.0

        if self.__exclude_terrain:
            # batchwise scaled loss
            loss = (loss*scale).sum(-1).sum(-1).sum(-1)

            # normalization factor per batch
            factor = torch.where(input[:,0,:,:,:]==0, torch.zeros_like(scale), scale).sum(-1).sum(-1).sum(-1)

            # normalize and compute the mean over the batches
            return (loss / factor.clamp(min=1.0)).mean()

        else:
            return (loss*scale).sum() / scale.sum()
