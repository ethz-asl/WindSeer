'''
Script to generate a test dataset from an existing dataset with resampling and data augmentation if requested.
'''

import argparse
import windseer.data as data

# parse the command line arguments
parser = argparse.ArgumentParser(
    description='Script sample a fixed dataset for model performance evaluation'
    )
parser.add_argument(
    '-i', dest='infile', default='data/input.hdf5', help='input dataset'
    )
parser.add_argument(
    '-o', dest='outfile', default='data/resampled.hdf5', help='output dataset'
    )
parser.add_argument(
    '-oc',
    dest='output_compressed',
    action='store_true',
    help='indicates if the output tensors should be compressed'
    )
parser.add_argument(
    '-nx', dest='nx', default=64, type=int, help='number of cells in x-direction'
    )
parser.add_argument(
    '-ny', dest='ny', default=64, type=int, help='number of cells in y-direction'
    )
parser.add_argument(
    '-nz', dest='nz', default=64, type=int, help='number of cells in z-direction'
    )
parser.add_argument(
    '-am', dest='augmentation_mode', default=1, help='augmentation mode'
    )
parser.add_argument(
    '-n',
    dest='sampling_rounds',
    type=int,
    default=6,
    help='number of samples per original sample'
    )

args = parser.parse_args()

input_mode = 0  # does not matter since only the labels are used

augmentation = True
augmentation_kwargs = {'subsampling': True, 'rotating': True, }

# get all channels, IMPORTANT: THE LABEL CHANNELS NEED TO BE IN THE CORRECT ORDER
input_channels = ['terrain', 'ux', 'uy', 'uz']
label_channels = ['ux', 'uy', 'uz', 'turb']

dataset = data.HDF5Dataset(
    args.infile,
    input_channels=input_channels,
    label_channels=label_channels,
    nx=args.nx,
    ny=args.ny,
    nz=args.nz,
    input_mode=input_mode,
    augmentation=augmentation,
    augmentation_mode=args.augmentation_mode,
    augmentation_kwargs=augmentation_kwargs,
    stride_hor=1,
    stride_vert=1,
    return_grid_size=True,
    return_name=True
    )

data.sample_dataset(dataset, args.outfile, args.sampling_rounds, args.output_compressed)
