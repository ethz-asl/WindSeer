'''
Script to generate a test dataset from an existing dataset with resampling and data augmentation if requested.
'''

import nn_wind_prediction.data as data

from torch.utils.data import DataLoader

#---------------- configure the dataset settings here --------------------------------------------
# input dataset params
input_compressed = False
input_dataset = 'data/input.tar'

# output dataset params
output_dataset = 'data/test.tar'
nx = 64
ny = 64
nz = 64
input_mode = 0
augmentation = True
subsample = True
stride_hor = 1
stride_vert = 1

output_compressed = False
n_sampling_rounds = 24

#---------------- end of configurations ----------------------------------------------------------

dbloader = data.MyDataset(torch.device("cpu"), input_dataset, nx, ny, nz, input_mode, subsample, augmentation,
                    stride_hor = stride_hor, stride_vert = stride_vert, turbulence_label = True,
                    compressed = input_compressed, use_grid_size = False, return_grid_size = True,
                    return_name = True)

data.sample_dataset(dbloader, output_dataset, n_sampling_rounds, output_compressed)
