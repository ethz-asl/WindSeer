import numpy as np
import torch
from torch.nn import Module
import torch.nn.functional as f

class MyLoss(Module):
    def __init__(self, use_turbulence_scaling = False, turbulence_factor = 1.0):
        super(MyLoss, self).__init__()
        self.__turbulence_factor = turbulence_factor
        self.__use_turbulence_scaling = use_turbulence_scaling

    def forward(self, prediction, label):
        if (prediction.shape != label.shape):
            raise ValueError('Prediction and label do not have the same shape')

        if (len(prediction.shape) != 5):
            raise ValueError('The loss is only defined for 5D data')

        # compute mse loss for each channel for each batch
        mse_loss = f.mse_loss(prediction, label, reduction='none')
        loss = torch.sum(mse_loss, [2,3,4]) / label.abs().mean(-1).mean(-1).mean(-1)

        loss = loss.sum() / label.numel()

        if self.__use_turbulence_scaling:
            print('not implemented yet')

        return loss
