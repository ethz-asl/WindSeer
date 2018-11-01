import numpy as np
import torch
from torch.nn import Module
import torch.nn.functional as f

class MyLoss(Module):
    def __init__(self, device, derivation_scaling = 3.0):
        super(MyLoss, self).__init__()
        self.__derivation_scaling = derivation_scaling
        self.__device = device

    def forward(self, input, label):
        loss = torch.zeros(1).to(self.__device)

        if (len(list(input.size())) > 4):
            for j in range(input.shape[0]):
                for i in range(input.shape[1]):
                    loss += f.mse_loss(input[j,i,:,:,:], label[j,i,:,:,:])/(input.shape[1] * label[j,i,:,:,:].abs().mean().item())
        else:
            for j in range(input.shape[0]):
                for i in range(input.shape[1]):
                    loss += f.mse_loss(input[j,i,:,:], label[j,i,:,:])/(input.shape[1] * label[j,i,:,:].abs().mean().item())

        loss /= input.shape[0]

        return loss
