import numpy as np
import torch
from torch.nn import Module
import torch.nn.functional as f

class ScaledLoss(Module):
    def __init__(self, method = 'MSE', max_scale = 4.0, norm_threshold = 0.5, exclude_terrain = True):
        super(ScaledLoss, self).__init__()

        # input sanity checks
        if (max_scale < 1.0):
            raise ValueError('max_scale must be larger than 1.0')

        if (method == 'MSE'):
            self.__loss = torch.nn.MSELoss(reduction='none')
        elif (method == 'L1'):
            self.__loss = torch.nn.L1Loss(reduction='none')

        self.__max_scale = max_scale
        self.__exclude_terrain = exclude_terrain
        self.__norm_threshold = norm_threshold

    def forward(self, prediction, label, input):
        # compute the scale
        scale = label[:,0:3]-input[:,1:4]
        scale = (scale.norm(dim=1) / label[:,0:3].norm(dim=1))

        # set nans to 0.0
        scale = torch.where(torch.isnan(scale), torch.zeros_like(scale), scale)

        # map to a range of 1.0 to max_scale
        scale = scale.clamp(min=0.0, max=self.__norm_threshold) * (1.0 / self.__norm_threshold) * (self.__max_scale - 1.0) + 1.0

        # compute the loss per pixel
        loss = self.__loss(prediction, label).mean(dim=1)

        if self.__exclude_terrain:
            # batchwise scaled loss
            loss = (loss*scale).sum(-1).sum(-1).sum(-1)

            # normalization factor per batch
            factor = torch.where(input[:,0,:,:,:]==0, torch.zeros_like(scale), scale).sum(-1).sum(-1).sum(-1)

            # normalize and compute the mean over the batches
            return (loss / factor.clamp(min=1.0)).mean()

        else:
            return (loss*scale).sum() / scale.sum()
