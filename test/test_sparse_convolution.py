#!/usr/bin/env python
'''
Script to test and benchmark the implementation of the sparse convolution
'''

import matplotlib.pyplot as plt
import nn_wind_prediction.nn as nn_custom
import numpy as np
import torch
import torch.nn as nn



def main():
    conv_type = nn.Conv2d
    conv = nn_custom.SparseConv(conv_type, in_channels=1, out_channels=1, kernel_size=3)

    x = torch.randn(1,1,12,12)
    x = (x > 0).float()

    out = conv(torch.nn.functional.pad(x,[1, 1, 1, 1]))

    # convert the output again to a boolean mask
    out_bool = (out.abs() > 0).float()

    # check that only previously activated cells are also active
    print((out_bool - x).sum())

    with torch.no_grad():
        plt.figure()
        plt.imshow(x.squeeze().numpy())
        plt.title('input')
        plt.figure()
        plt.imshow(out.abs().squeeze().numpy())
        plt.title('output')
        plt.show()


if __name__ == '__main__':
    main()
