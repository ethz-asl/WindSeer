#!/usr/bin/env python

'''
Convert a dataset of csv files to serialized torch tensors.

TODO: Investigate how to speed up the 3D case
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

def convert_data(infile, outfile, vlim, nx, ny, nz, nutlim, d3, boolean_terrain, verbose = False):
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
        nutlim: Maximum allowed turbulence viscosity
        d3: If true the input is assumed to be 3d, else 2d
        boolean_terrain: If true the terrain is represented by a boolean variable, if false by a distance field
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
            try:
                wind_data = pd.read_csv(f, header=0, dtype = types)
            except:
                print('Reading the csv {0} failed'.format(member.name))

            # quick sanity check if the csv has the expected format
            if 'U:0' not in wind_data.keys():
                rejection_counter += 1
                print('U:0 not in {0}'.format(member.name))

            else:
                # check if the wind entries are fine (basically a check if the cfd simulation converged
                max_0 = np.max(wind_data['U:0'])
                min_0 = np.min(wind_data['U:0'])
                max_1 = np.max(wind_data['U:1'])
                min_1 = np.min(wind_data['U:1'])
                max_2 = np.max(wind_data['U:2'])
                min_2 = np.min(wind_data['U:2'])
                max_nut = np.max(wind_data['nut'])
                min_nut = np.min(wind_data['nut'])

                if ((np.abs(max_0) > vlim) or
                    (np.abs(min_0) > vlim) or
                    (np.abs(min_1) > vlim) or
                    (np.abs(max_1) > vlim) or
                    (np.abs(max_2) > vlim) or
                    (np.abs(min_2) > vlim) or
                    (np.abs(max_nut) > nutlim) or
                    (np.abs(min_nut) > nutlim)):

                    rejection_counter += 1
                    if verbose:
                        print('------------------------------------')
                        print('Removing', member.name)
                        print('Statistics: max U0: ', max_0, ', max U1: ', max_1, ', max U2:', max_2)
                        print('            min U0: ', min_0, ', min U1: ', min_1, ', min U2:', min_2)
                        print('            min nut:', min_nut, ', max nut:', max_nut)
                        print('------------------------------------')

                else:
                    channel_shape = [nz, nx]

                    if d3:
                        channel_shape = [nz, ny, nx]

                    # generate the labels
                    u_x_out = wind_data.get('U:0').values.reshape(channel_shape)
                    u_y_out = wind_data.get('U:1').values.reshape(channel_shape)
                    u_z_out = wind_data.get('U:2').values.reshape(channel_shape)
                    turbelence_viscosity_out = wind_data.get('nut').values.reshape(channel_shape)

                    # generate the input
                    is_wind = wind_data.get('vtkValidPointMask').values.reshape(channel_shape).astype(np.float32)

                    if d3:
                        if boolean_terrain:
                            distance_field_in = is_wind
                        else:
                            is_wind = np.insert(is_wind, 0, np.zeros((1, ny, nx)), axis = 0)
                            distance_field_in = ndimage.distance_transform_edt(is_wind).astype(np.float32)
                            distance_field_in = distance_field_in[1:, :, :]

                        u_x_in = np.einsum('kij->ijk',np.tile(u_x_out[:, :, 0], [u_x_out.shape[2], 1, 1]))
                        u_y_in = np.einsum('kij->ijk',np.tile(u_y_out[:, :, 0], [u_y_out.shape[2], 1, 1]))
                        u_z_in = np.einsum('kij->ijk',np.tile(u_z_out[:, :, 0], [u_z_out.shape[2], 1, 1]))

                    else:
                        if boolean_terrain:
                            distance_field_in = is_wind
                        else:
                            is_wind = np.insert(is_wind, 0, np.zeros((1, nx)), axis = 0)
                            distance_field_in = ndimage.distance_transform_edt(is_wind).astype(np.float32)
                            distance_field_in = distance_field_in[1:,:]

                        u_x_in = np.tile(u_x_out[:,0], [u_x_out.shape[1],1]).transpose()
                        u_y_in = np.tile(u_y_out[:,0], [u_y_out.shape[1],1]).transpose()
                        u_z_in = np.tile(u_z_out[:,0], [u_z_out.shape[1],1]).transpose()

                    # store the stacked data
                    out = np.stack([distance_field_in, u_x_in, u_y_in, u_z_in, u_x_out, u_y_out, u_z_out, turbelence_viscosity_out])
                    out_tensor = torch.from_numpy(out)
                    torch.save(out_tensor, 'tmp/' + member.name.replace('.csv','') + '.tp')

                    if d3:
                        # rotate the sample around the z-axis
                        out_rot = np.stack([distance_field_in, -u_y_in, u_x_in, u_z_in, -u_y_out, u_x_out, u_z_out, turbelence_viscosity_out])
                        out_rot = out_rot.swapaxes(-2,-1)[...,::-1].copy()
                        out_tensor = torch.from_numpy(out_rot)
                        torch.save(out_tensor, 'tmp/' + member.name.replace('.csv','') + '_rot.tp')

                        # flip in x direction
                        u_x_out_flipped = np.flip(u_x_out,2) * (-1.0)
                        u_y_out_flipped = np.flip(u_y_out,2)
                        u_z_out_flipped = np.flip(u_z_out,2)
                        turbelence_viscosity_out_flipped = np.flip(turbelence_viscosity_out,2)

                        u_x_in_flipped = np.flip(u_x_in,2) * (-1.0)
                        u_y_in_flipped = np.flip(u_y_in,2)
                        u_z_in_flipped = np.flip(u_z_in,2)
                        distance_field_in_flipped = np.flip(distance_field_in,2)

                        out_flipped = np.stack([distance_field_in_flipped, u_x_in_flipped, u_y_in_flipped, u_z_in_flipped, u_x_out_flipped, u_y_out_flipped, u_z_out_flipped, turbelence_viscosity_out_flipped])
                        out_tensor_flipped = torch.from_numpy(out_flipped)
                        torch.save(out_tensor_flipped, 'tmp/' + member.name.replace('.csv','') + '_flipped_x.tp')

                        # rotate the flipped data
                        out_rot = np.stack([distance_field_in_flipped, -u_y_in_flipped, u_x_in_flipped, u_z_in_flipped, -u_y_out_flipped, u_x_out_flipped, u_z_out_flipped, turbelence_viscosity_out_flipped])
                        out_rot = out_rot.swapaxes(-2,-1)[...,::-1].copy()
                        out_tensor = torch.from_numpy(out_rot)
                        torch.save(out_tensor, 'tmp/' + member.name.replace('.csv','') + '_flipped_x_rot.tp')

                        # flip in y direction
                        u_x_out_flipped = np.flip(u_x_out,1)
                        u_y_out_flipped = np.flip(u_y_out,1) * (-1.0)
                        u_z_out_flipped = np.flip(u_z_out,1)
                        turbelence_viscosity_out_flipped = np.flip(turbelence_viscosity_out,1)

                        u_x_in_flipped = np.flip(u_x_in,1)
                        u_y_in_flipped = np.flip(u_y_in,1) * (-1.0)
                        u_z_in_flipped = np.flip(u_z_in,1)
                        distance_field_in_flipped = np.flip(distance_field_in,1)

                        out_flipped = np.stack([distance_field_in_flipped, u_x_in_flipped, u_y_in_flipped, u_z_in_flipped, u_x_out_flipped, u_y_out_flipped, u_z_out_flipped, turbelence_viscosity_out_flipped])
                        out_tensor_flipped = torch.from_numpy(out_flipped)
                        torch.save(out_tensor_flipped, 'tmp/' + member.name.replace('.csv','') + '_flipped_y.tp')

                        # rotate the flipped data
                        out_rot = np.stack([distance_field_in_flipped, -u_y_in_flipped, u_x_in_flipped, u_z_in_flipped, -u_y_out_flipped, u_x_out_flipped, u_z_out_flipped, turbelence_viscosity_out_flipped])
                        out_rot = out_rot.swapaxes(-2,-1)[...,::-1].copy()
                        out_tensor = torch.from_numpy(out_rot)
                        torch.save(out_tensor, 'tmp/' + member.name.replace('.csv','') + '_flipped_y_rot.tp')

                        # flip in y direction
                        u_x_out_flipped = np.flip(np.flip(u_x_out,1),2) * (-1.0)
                        u_y_out_flipped = np.flip(np.flip(u_y_out,1),2) * (-1.0)
                        u_z_out_flipped = np.flip(np.flip(u_z_out,1),2)
                        turbelence_viscosity_out_flipped = np.flip(np.flip(turbelence_viscosity_out,1),2)

                        u_x_in_flipped = np.flip(np.flip(u_x_in,1),2) * (-1.0)
                        u_y_in_flipped = np.flip(np.flip(u_y_in,1),2) * (-1.0)
                        u_z_in_flipped = np.flip(np.flip(u_z_in,1),2)
                        distance_field_in_flipped = np.flip(np.flip(distance_field_in,1),2)

                        out_flipped = np.stack([distance_field_in_flipped, u_x_in_flipped, u_y_in_flipped, u_z_in_flipped, u_x_out_flipped, u_y_out_flipped, u_z_out_flipped, turbelence_viscosity_out_flipped])
                        out_tensor_flipped = torch.from_numpy(out_flipped)
                        torch.save(out_tensor_flipped, 'tmp/' + member.name.replace('.csv','') + '_flipped_xy.tp')

                        # rotate the flipped data
                        out_rot = np.stack([distance_field_in_flipped, -u_y_in_flipped, u_x_in_flipped, u_z_in_flipped, -u_y_out_flipped, u_x_out_flipped, u_z_out_flipped, turbelence_viscosity_out_flipped])
                        out_rot = out_rot.swapaxes(-2,-1)[...,::-1].copy()
                        out_tensor = torch.from_numpy(out_rot)
                        torch.save(out_tensor, 'tmp/' + member.name.replace('.csv','') + '_flipped_xy_rot.tp')
                    else:
                        # generate the flipped flow
                        u_x_out_flipped = np.flip(u_x_out,1) * (-1.0)
                        u_y_out_flipped = np.flip(u_y_out,1)
                        u_z_out_flipped = np.flip(u_z_out,1)
                        turbelence_viscosity_out_flipped = np.flip(turbelence_viscosity_out,1)

                        u_x_in_flipped = np.flip(u_x_in,1) * (-1.0)
                        u_y_in_flipped = np.flip(u_y_in,1)
                        u_z_in_flipped = np.flip(u_z_in,1)
                        distance_field_in_flipped = np.flip(distance_field_in,1)

                        out_flipped = np.stack([distance_field_in_flipped, u_x_in_flipped, u_y_in_flipped, u_z_in_flipped, u_x_out_flipped, u_y_out_flipped, u_z_out_flipped, turbelence_viscosity_out_flipped])

                        out_tensor_flipped = torch.from_numpy(out_flipped)

                        torch.save(out_tensor_flipped, 'tmp/' + member.name.replace('.csv','') + '_flipped.tp')

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
    parser.add_argument('-nutlim', type=float, default=1000.0, help='limit of the turbulent viscosity')
    parser.add_argument('-v', dest='verbose', action='store_true', help='verbose')
    parser.add_argument('-nx', default=128, help='number of gridpoints in x-direction')
    parser.add_argument('-ny', default=128, help='number of gridpoints in y-direction')
    parser.add_argument('-nz', default=64, help='number of gridpoints in z-direction')
    parser.add_argument('-3d', dest='d3', action='store_true', help='3D input')
    parser.add_argument('-b', dest='boolean_terrain', action='store_true', help='If flag is set the terrain is represented by a boolean variable, else by a distance field.')
    args = parser.parse_args()

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
    convert_data(args.infile, args.outfile, args.vlim, args.nx, args.ny, args.nz, args.nutlim, args.d3, args.boolean_terrain, args.verbose)
    print("INFO: Converting the database took %s seconds" % (time.time() - start_time))

if __name__ == "__main__":
    main()
