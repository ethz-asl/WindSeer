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

        num_features = int(np.ceil(0.5 * input.shape[2] * input.shape[3] * input.shape[4]))
        for j in range(input.shape[0]):
            for i in range(input.shape[1]):
                val, idx = (input[j,i,:,:,:] - label[j,i,:,:,:]).abs().view(1, -1).topk(num_features, sorted = False)
                loss += f.mse_loss(input[j,i,:,:,:].view(1, -1)[0, idx], label[j,i,:,:,:].view(1, -1)[0, idx])

        loss /= input.shape[0]

        return loss
