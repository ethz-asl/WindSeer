from torch.nn import Module
import torch
import torch.nn.functional as f
import nn_wind_prediction.utils as utils

class StreamFunctionLoss(Module):
    '''
    Stream Function Loss according to https://arxiv.org/abs/1806.02071
    '''
    def __init__(self):
        super(StreamFunctionLoss, self).__init__()


    def forward(self, net_output, target):
        if (net_output.shape != target.shape):
            raise ValueError('Prediction and target do not have the same shape')

        if (len(net_output.shape) != 5):
            raise ValueError('The loss is only defined for 5D data')

        return self.compute_loss(net_output, target)

    def compute_loss(self, net_output, target):
        curled_output = torch.cat([utils.curl(net_output,ds=1), net_output[:, 3:, :]], 1)
        return f.l1_loss(curled_output, target)