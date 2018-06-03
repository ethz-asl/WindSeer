#!/usr/bin/env python

'''
Convert a dataset of csv files to serialized torch tensors.

TODO: Support for 3D data
'''

import argparse
from math import trunc
import numpy as np
import os
import pandas as pd
from scipy import ndimage
import shutil
import sys
import tarfile
import time
import torch

def convert_data(infile, outfile, vlim, nx, ny, nz, verbose = False):
    '''
    Function which loops through the files of the input tar file.
    The velocity is checked and files are rejected if a single dimension
    contains a value larger than vlim.
    The files are converted to numpy arrays, serialized and stored in a
    tar file.
    
    Params:
        infile: Input archive name
        outfile: Output archive name
        vlim: Maximum allowed velocity [m/s]
        nx: Number of points in x direction of the grid
        ny: Number of points in y direction of the grid
        nz: Number of points in z direction of the grid
        verbose: Show the stats of the deleted files
    '''
    # open the file
    tar = tarfile.open(infile, 'r')
    num_files = len(tar.getnames())

    # create temp directory to store all serialized arrays
    os.makedirs('tmp')

    # define types of csv files
    types = {"p": np.float32,
             "U:0": np.float32,
             "U:1": np.float32,
             "U:2": np.float32,
             "epsilon": np.float32,
             "k": np.float32,
             "nut": np.float32,
             "vtkValidPointMask": np.bool,
             "Points:0": np.float32,
             "Points:1": np.float32,
             "Points:2": np.float32}

    print('INFO: Looping through all the files')
    rejection_counter = 0
    for i, member in enumerate(tar.getmembers()):
        f = tar.extractfile(member)

        if f is not None:
            wind_data = pd.read_csv(f, header=0, dtype = types)

            # quick sanity check if the csv has the expected format
            if 'U:0' not in wind_data.keys():
                print('U:0 not in {0}'.format(member.name))
                raise IOError

            # check if the wind entries are fine (basically a check if the cfd simulation converged
            max_0 = np.max(wind_data['U:0'])
            min_0 = np.min(wind_data['U:0'])
            max_2 = np.max(wind_data['U:2'])
            min_2 = np.min(wind_data['U:2'])

            if ((np.abs(max_0) > vlim) or
                (np.abs(min_0) > vlim) or
                (np.abs(max_2) > vlim) or
                (np.abs(min_2) > vlim)):

                rejection_counter += 1
                if verbose:
                    print('------------------------------------')
                    print('Removing', member.name)
                    print('Statistics: max U0:', max_0, ', maxU2:', max_2, ', minU0:', min_0, ', minU2:', min_2)
                    print('------------------------------------')

            else:
                # generate the labels
                u_x_out = wind_data.get('U:0').values.reshape([nz, nx])
                u_z_out = wind_data.get('U:2').values.reshape([nz, nx])
                turbelence_viscosity_out = wind_data.get('nut').values.reshape([nz, nx])

                # generate the input
                is_wind_in = ndimage.distance_transform_edt(wind_data.get('vtkValidPointMask').values.reshape([nz, nx])).astype(np.float32)

                u_x_in = np.tile(u_x_out[:,0], [u_x_out.shape[1],1]).transpose()
                u_z_in = np.tile(u_z_out[:,0], [u_z_out.shape[1],1]).transpose()

                out = np.stack([is_wind_in, u_x_in, u_z_in, u_x_out, u_z_out, turbelence_viscosity_out])

                out_tensor = torch.from_numpy(out)

                torch.save(out_tensor, 'tmp/' + member.name.replace('.csv','') + '.tp')

        if ((i % np.ceil(num_files/10.0)) == 0.0):
            print(trunc((i+1)/num_files*100), '%')

    print('INFO: Finished parsing all the files, rejecting', rejection_counter, 'out of', num_files)

    # collecting all files in the tmp folder to a tar
    out_tar = tarfile.open(outfile, 'w')
    for filename in os.listdir('tmp'):
        out_tar.add('tmp/' + filename, arcname = filename)

    # cleaning up
    out_tar.close()
    tar.close()
    shutil.rmtree('tmp')


def main():
    '''
    Main function which parses the arguments and then calls convert_data
    '''
    parser = argparse.ArgumentParser(description='Script to remove bad data from a database')
    parser.add_argument('-i', dest='infile', required=True, help='input tar file')
    parser.add_argument('-o', dest='outfile', help='output tar file, if none provided the input file name is prepended with "converted_"')
    parser.add_argument('-vlim', type=float, default=1000.0, help='limit of the velocity magnitude in one dimension')
    parser.add_argument('-v', dest='verbose', action='store_true', help='verbose')
    parser.add_argument('-nx', default=128, help='number of gridpoints in x-direction')
    parser.add_argument('-ny', default=128, help='number of gridpoints in y-direction')
    parser.add_argument('-nz', default=64, help='number of gridpoints in z-direction')
    parser.add_argument('-3d', dest='d3', action='store_true', help='3D input')
    args = parser.parse_args()

    if (args.d3):
        print('ERROR: Currently there is no 3D support', file=sys.stderr)
        sys.exit()

    if (args.outfile == args.infile):
        print('WARNING: The outfile cannot be the same file as the infile, prepending "converted_"')
        args.outfile = None

    if (not args.outfile):
        in_splitted = args.infile.split('/')
        if len(in_splitted) > 1:
            out = ''
            for elem in in_splitted[0:-1]:
                out = out + elem + '/'

            args.outfile = out + 'converted_' + in_splitted[-1]

        else:
            args.outfile = 'converted_' + args.infile

    start_time = time.time()
    convert_data(args.infile, args.outfile, args.vlim, args.nx, args.ny, args.nz, args.verbose)
    print("INFO: Converting the database took %s seconds" % (time.time() - start_time))

if __name__ == "__main__":
    main()
