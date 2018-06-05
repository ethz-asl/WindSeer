import torch
from torch.nn import Module
import torch.nn.functional as f

class MyLoss(Module):
    def __init__(self, device, derivation_scaling = 3.0):
        super(MyLoss, self).__init__()
        self.__derivation_scaling =  derivation_scaling
        self.__device = device
       
    def forward(self, input, label):
        loss = torch.zeros(1).to(self.__device)
        
        for i in range(input.shape[1]):
            loss += f.mse_loss(input[:,i,:,:], label[:,i,:,:])
            loss += self.__derivation_scaling * f.mse_loss(input[:,i,:,1:] - input[:,i,:,:-1], label[:,i,:,1:] - label[:,i,:,:-1])

        return loss
