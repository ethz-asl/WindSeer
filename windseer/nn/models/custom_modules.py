import torch.nn as nn
import torch


class SparseConv(nn.Module):
    """
    A very simple but inefficient way of implementing sparse convolution according to
    3D Semantic Segmentation with Submanifold Sparse Convolutional Networks, CVPR 2018
    """

    def __init__(self, conv_type, mask_exclude_first_dim=False, **kwargs):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(SparseConv, self).__init__()

        if (kwargs['kernel_size'] % 2) == 0:
            raise ValueError('SparseConv requires an odd kernel size')

        self._conv = conv_type(**kwargs)

        # convolution to get the mask in the same shape as the output
        mask_kwargs = kwargs.copy()
        mask_kwargs['in_channels'] = 1
        mask_kwargs['out_channels'] = 1
        mask_kwargs['bias'] = False
        center = int(mask_kwargs['kernel_size'] / 2)

        # set all but the center pixel to 0 and the center to 1
        self._conv_mask = conv_type(**mask_kwargs)
        self._conv_mask.weight.requires_grad = False
        self._conv_mask.weight *= 0
        idx = (torch.tensor(self._conv_mask.weight.shape) * 0.5).to(torch.long)
        self._conv_mask.weight[idx.split(1)] = 1.0

        self._mask_exclude_first_dim = mask_exclude_first_dim

    def forward(self, x):
        '''
        Module prediction function

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        output : torch.Tensor
            Module output
        '''
        # create mask by checking if all channels in a cell are 0
        if self._mask_exclude_first_dim:
            mask = (x[:, 1:].abs().sum(1) != 0).float().unsqueeze(1)
        else:
            mask = (x.abs().sum(1) != 0).float().unsqueeze(1)

        mask = self._conv_mask(mask)

        x = self._conv(x)

        return x * mask.expand(x.shape)
