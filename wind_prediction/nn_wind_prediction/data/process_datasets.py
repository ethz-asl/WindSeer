#!/usr/bin/env python

from __future__ import print_function

import io
import lz4.frame
from math import trunc
from scipy import ndimage
import numpy as np
import os
import pandas as pd
import shutil
import tarfile
import time
import torch

def save_data(tensor, ds, name, compress):
    if compress:
        bytes = io.BytesIO()
        torch.save((tensor, ds), bytes)
        f = open(name, 'wb')
        f.write(lz4.frame.compress(bytes.getvalue(), compression_level = -20))
        f.close()
    else:
        torch.save((tensor, ds), name)

def compress_dataset(infile, outfile, s_hor, s_ver, input_compressed, compress):
    # open the file
    tar = tarfile.open(infile, 'r')
    num_files = len(tar.getnames())

    # create temp directory to store all serialized arrays
    if (os.path.isdir("/cluster/scratch/")):
        tempfolder = '/scratch/tmp_' + time.strftime("%Y_%m_%d-%H_%M_%S") + '/'
    else:
        tempfolder = 'tmp_' + time.strftime("%Y_%m_%d-%H_%M_%S") + '/'

    os.makedirs(tempfolder)

    print('INFO: Looping through all the files')
    for i, member in enumerate(tar.getmembers()):
        file = tar.extractfile(member)

        if input_compressed:
            data, ds = torch.load(io.BytesIO(lz4.frame.decompress(file.read())))

        else:
            data, ds = torch.load(file)

        if (len(list(data.size())) > 3):
            out = data[:,::s_ver,::s_hor, ::s_hor].clone()
        else:
            out = data[:,::s_ver, ::s_hor].clone()

        save_data(out, (ds[0]*s_hor, ds[1]*s_hor, ds[2]*s_ver), tempfolder + member.name, compress)

        if ((i % np.ceil(num_files/10.0)) == 0.0):
            print(trunc((i+1)/num_files*100), '%')

    print('INFO: Finished compressing all the files')

    # collecting all files in the tmp folder to a tar
    out_tar = tarfile.open(outfile, 'w')
    for filename in os.listdir(tempfolder):
        out_tar.add(tempfolder + filename, arcname = filename)

    # cleaning up
    out_tar.close()
    tar.close()
    shutil.rmtree(tempfolder)

def convert_dataset(infile, outfile, vlim, nx, ny, nz, klim, d3, boolean_terrain, verbose = False, compress = False):
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
        klim: Maximum allowed turbulence viscosity
        d3: If true the input is assumed to be 3d, else 2d
        boolean_terrain: If true the terrain is represented by a boolean variable, if false by a distance field
        verbose: Show the stats of the deleted files
    '''
    # open the file
    tar = tarfile.open(infile, 'r')
    num_files = len(tar.getnames())

    # create temp directory to store all serialized arrays
    if (os.path.isdir("/cluster/scratch/")):
        tempfolder = '/scratch/tmp_' + time.strftime("%Y_%m_%d-%H_%M_%S") + '/'
    else:
        tempfolder = 'tmp_' + time.strftime("%Y_%m_%d-%H_%M_%S") + '/'

    os.makedirs(tempfolder)

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
                max_k = np.max(wind_data['k'])
                min_k = np.min(wind_data['k'])
                x = wind_data['Points:0'][:nx]
                y = wind_data['Points:1'][::nx]
                y = y[:ny]
                z = wind_data['Points:2'][::nx*ny]
                dx = np.mean(np.diff(x.values))
                dy = np.mean(np.diff(y.values))
                dz = np.mean(np.diff(z.values))

                if ((np.abs(max_0) > vlim) or
                    (np.abs(min_0) > vlim) or
                    (np.abs(min_1) > vlim) or
                    (np.abs(max_1) > vlim) or
                    (np.abs(max_2) > vlim) or
                    (np.abs(min_2) > vlim) or
                    (np.abs(max_k) > klim) or
                    (np.abs(min_k) > klim) or
                    (np.std(np.diff(x.values)) > 0.1) or
                    (np.std(np.diff(y.values)) > 0.1) or
                    (np.std(np.diff(z.values)) > 0.1)):

                    rejection_counter += 1
                    if verbose:
                        print('------------------------------------')
                        print('Removing', member.name)
                        print('Statistics: max U0: ', max_0, ', max U1: ', max_1, ', max U2:', max_2)
                        print('            min U0: ', min_0, ', min U1: ', min_1, ', min U2:', min_2)
                        print('            min k:', min_k, ', max k:', max_k)
                        print('            regular grid:', 
                              not ((np.std(np.diff(x.values)) > 0.1) or (np.std(np.diff(y.values)) > 0.1) or (np.std(np.diff(z.values)) > 0.1)))
                        print('------------------------------------')

                else:
                    channel_shape = [nz, nx]

                    if d3:
                        channel_shape = [nz, ny, nx]

                    # generate the labels
                    u_x_out = wind_data.get('U:0').values.reshape(channel_shape)
                    u_y_out = wind_data.get('U:1').values.reshape(channel_shape)
                    u_z_out = wind_data.get('U:2').values.reshape(channel_shape)
                    turbelence_viscosity_out = wind_data.get('k').values.reshape(channel_shape)

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
                    save_data(out_tensor, (dx, dy, dz), tempfolder + member.name.replace('.csv','') + '.tp', compress)

                    if d3:
                        # rotate the sample around the z-axis
                        out_rot = np.stack([distance_field_in, -u_y_in, u_x_in, u_z_in, -u_y_out, u_x_out, u_z_out, turbelence_viscosity_out])
                        out_rot = out_rot.swapaxes(-2,-1)[...,::-1].copy()
                        out_tensor = torch.from_numpy(out_rot)
                        save_data(out_tensor, (dy, dx, dz), tempfolder + member.name.replace('.csv','') + '_rot.tp', compress)

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
                        save_data(out_tensor_flipped, (dx, dy, dz), tempfolder + member.name.replace('.csv','') + '_flipped_x.tp', compress)

                        # rotate the flipped data
                        out_rot = np.stack([distance_field_in_flipped, -u_y_in_flipped, u_x_in_flipped, u_z_in_flipped, -u_y_out_flipped, u_x_out_flipped, u_z_out_flipped, turbelence_viscosity_out_flipped])
                        out_rot = out_rot.swapaxes(-2,-1)[...,::-1].copy()
                        out_tensor = torch.from_numpy(out_rot)
                        save_data(out_tensor, (dy, dx, dz), tempfolder + member.name.replace('.csv','') + '_flipped_x_rot.tp', compress)

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
                        save_data(out_tensor_flipped, (dx, dy, dz), tempfolder + member.name.replace('.csv','') + '_flipped_y.tp', compress)

                        # rotate the flipped data
                        out_rot = np.stack([distance_field_in_flipped, -u_y_in_flipped, u_x_in_flipped, u_z_in_flipped, -u_y_out_flipped, u_x_out_flipped, u_z_out_flipped, turbelence_viscosity_out_flipped])
                        out_rot = out_rot.swapaxes(-2,-1)[...,::-1].copy()
                        out_tensor = torch.from_numpy(out_rot)
                        save_data(out_tensor, (dy, dx, dz), tempfolder + member.name.replace('.csv','') + '_flipped_y_rot.tp', compress)

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
                        save_data(out_tensor_flipped, (dx, dy, dz), tempfolder + member.name.replace('.csv','') + '_flipped_xy.tp', compress)

                        # rotate the flipped data
                        out_rot = np.stack([distance_field_in_flipped, -u_y_in_flipped, u_x_in_flipped, u_z_in_flipped, -u_y_out_flipped, u_x_out_flipped, u_z_out_flipped, turbelence_viscosity_out_flipped])
                        out_rot = out_rot.swapaxes(-2,-1)[...,::-1].copy()
                        out_tensor = torch.from_numpy(out_rot)
                        save_data(out_tensor, (dy, dx, dz), tempfolder + member.name.replace('.csv','') + '_flipped_xy_rot.tp', compress)
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

                        save_data(out_tensor_flipped, (dx, dy, dz), tempfolder + member.name.replace('.csv','') + '_flipped.tp', compress)

        if ((i % np.ceil(num_files/10.0)) == 0.0):
            print(trunc((i+1)/num_files*100), '%')

    print('INFO: Finished parsing all the files, rejecting', rejection_counter, 'out of', num_files)

    # collecting all files in the tmp folder to a tar
    out_tar = tarfile.open(outfile, 'w')
    for filename in os.listdir(tempfolder):
        out_tar.add(tempfolder + filename, arcname = filename)

    # cleaning up
    out_tar.close()
    tar.close()
    shutil.rmtree(tempfolder)

