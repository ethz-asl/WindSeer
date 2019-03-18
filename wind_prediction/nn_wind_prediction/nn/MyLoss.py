import numpy as np
import torch
from torch.nn import Module
import torch.nn.functional as f

class MyLoss(Module):
    def __init__(self, method = 'MSE', max_scale = 4.0):
        super(MyLoss, self).__init__()

        # input sanity checks
        if (max_scale < 1.0):
            raise ValueError('max_scale must be larger than 1.0')

        if (method == 'MSE'):
            self.__loss = torch.nn.MSELoss(reduction='none')
        elif (method == 'L1'):
            self.__loss = torch.nn.L1Loss(reduction='none')

        self.__max_scale = max_scale

    def forward(self, prediction, label, input):
        # compute the scale
        scale = label[:,0:3]-input[:,1:4]
        scale = (scale.norm(dim=1) / label[:,0:3].norm(dim=1))

        # set nans to 0.0
        nan_mask =  torch.isnan(scale)
        scale[nan_mask] = 0.0

        # map to a range of 1.0 to max_scale
        scale = scale.clamp(min=0.0, max=0.5).unsqueeze(0) * 2.0 * (self.__max_scale - 1.0) + 1.0

        # compute the loss per pixel
        loss = self.__loss(prediction, label).mean(dim=1)

        return (loss*scale).sum() / scale.sum()
