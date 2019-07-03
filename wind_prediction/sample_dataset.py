'''
Script to generate a test dataset from an existing dataset with resampling and data augmentation if requested.
'''

import argparse
import nn_wind_prediction.data as data
from torch.utils.data import DataLoader

#---------------- default config ---------------------------------------------------------------
# input dataset params
input_compressed = False
input_dataset = 'data/input.hdf5'

# output dataset params
output_dataset = 'data/resampled.hdf5'
input_channels = ['terrain', 'ux', 'uy', 'uz']
label_channels = ['ux', 'uy', 'uz']
nx = 64
ny = 64
nz = 64
input_mode = 1
augmentation = True
augmentation_mode = 1
augmentation_kwargs = {
    'subsampling': False,
    'rotating': True,
    }
stride_hor = 1
stride_vert = 1

output_compressed = False
n_sampling_rounds = 6

#---------------- end of default config ----------------------------------------------------------

# parse the command line arguments
parser = argparse.ArgumentParser(description='Script sample a fixed dataset for model performance evaluation')
parser.add_argument('-i', dest='infile', default=input_dataset, help='input dataset')
parser.add_argument('-o', dest='outfile', default=output_dataset, help='output dataset')
parser.add_argument('-ic', dest='input_compressed', action='store_true', help='input tensors are compressed')
parser.add_argument('-oc', dest='output_compressed', action='store_true', help='indicates if the output tensors should be compressed')
parser.add_argument('-nx', dest='nx', default=nx, type=int, help='number of cells in x-direction')
parser.add_argument('-ny', dest='ny', default=ny, type=int, help='number of cells in y-direction')
parser.add_argument('-nz', dest='nz', default=nz, type=int, help='number of cells in z-direction')
parser.add_argument('-im', dest='input_mode', default=input_mode, help='input_mode')
parser.add_argument('-am', dest='augmentation_mode', default=augmentation_mode, help='augmentation mode')
parser.add_argument('-n', dest='sampling_rounds', type=int, default=n_sampling_rounds, help='number of samples per original sample')

args = parser.parse_args()

dbloader = data.HDF5Dataset(args.infile, input_channels=input_channels, label_channels=label_channels, nx = args.nx,
                          ny = args.ny, nz = args.nz, input_mode = args.input_mode, augmentation = augmentation,
                          augmentation_mode = args.augmentation_mode, augmentation_kwargs = augmentation_kwargs,
                          stride_hor = stride_hor, stride_vert = stride_vert, turbulence_label = True,
                          compressed = args.input_compressed, return_grid_size = True,
                          return_name = True)

data.sample_dataset(dbloader, args.outfile, args.sampling_rounds, args.output_compressed)
