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
import h5py

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

def convert_dataset(infile, outfile, vlim, klim, boolean_terrain, verbose = True, create_zero_samples = True, compress = False):
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
        klim: Maximum allowed turbulence kinetik energy
        boolean_terrain: If true the terrain is represented by a boolean variable, if false by a distance field
        verbose: Show the stats of the deleted files
        create_zero_samples: Indicates if all zero samples should be created and saved for each new terrain encountered
    '''
    # open the file
    tar = tarfile.open(infile, 'r')
    num_files = len(tar.getnames())

    # define compression
    if compress:
        compression_type = "lzf"
    else:
        compression_type = None

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
    if create_zero_samples:
        print('INFO: Zero sample creation enabled')
    rejection_counter = 0
    zero_samples_created = 0

    # check output file extension
    if not outfile.endswith('.hdf5'):
        outfile = outfile + '.hdf5'

    # create h5 file where all the data wll be stored
    if os.path.exists(outfile):
        os.remove(outfile)
    ouput_file = h5py.File(outfile)

    # create list of channel names
    channels = ['terrain', 'ux', 'uy', 'uz', 'turb', 'p', 'epsilon', 'nut']

    # iterate over the csv files
    for i, member in enumerate(tar.getmembers()):
        f = tar.extractfile(member)

        # detect if a new terrain (a new W01 sample) is encountered
        if '_W01_' in member.name:
            new_terrain = True
        else:
            new_terrain = False

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
                # get dimensions of the grid
                x = np.unique(wind_data['Points:0'])
                y = np.unique(wind_data['Points:1'])
                z = np.unique(wind_data['Points:2'])
                nx = len(x)
                ny = len(y)
                nz = len(z)

                d3 = False
                if (ny > 1):
                    d3 = True

                # check if the wind entries are fine (basically a check if the cfd simulation converged
                max_0 = np.max(wind_data['U:0'])
                min_0 = np.min(wind_data['U:0'])
                max_1 = np.max(wind_data['U:1'])
                min_1 = np.min(wind_data['U:1'])
                max_2 = np.max(wind_data['U:2'])
                min_2 = np.min(wind_data['U:2'])
                max_k = np.max(wind_data['k'])
                min_k = np.min(wind_data['k'])
                dx = np.mean(np.diff(x))
                dy = np.mean(np.diff(y))
                dz = np.mean(np.diff(z))

                if ((np.abs(max_0) > vlim) or
                    (np.abs(min_0) > vlim) or
                    (np.abs(min_1) > vlim) or
                    (np.abs(max_1) > vlim) or
                    (np.abs(max_2) > vlim) or
                    (np.abs(min_2) > vlim) or
                    (np.abs(max_k) > klim) or
                    (np.abs(min_k) > klim) or
                    (np.std(np.diff(x)) > 0.1) or
                    (np.std(np.diff(y)) > 0.1) or
                    (np.std(np.diff(z)) > 0.1) or
                    (nx > 256) or
                    (ny > 256) or
                    (nz > 256)):

                    rejection_counter += 1
                    if verbose:
                        print('------------------------------------')
                        print('Removing', member.name)
                        print('Statistics: max U0: ', max_0, ', max U1: ', max_1, ', max U2:', max_2)
                        print('            min U0: ', min_0, ', min U1: ', min_1, ', min U2:', min_2)
                        print('            min k:', min_k, ', max k:', max_k)
                        print('            regular grid:', 
                              not ((np.std(np.diff(x)) > 0.1) or (np.std(np.diff(y)) > 0.1) or (np.std(np.diff(z)) > 0.1)))
                        print('            nx: ', nx, ', ny: ', ny, ', nz: ', nz)
                        print('------------------------------------')

                else:
                    # create group in hdf5 file for the current sample
                    sample = member.name.replace('.csv', '')
                    ouput_file.create_group(sample)

                    channel_shape = [nz, nx]
                    slice_shape = (1, nx)

                    if d3:
                        channel_shape = [nz, ny, nx]
                        slice_shape = (1, ny, nx)

                    # extract the cfd properties
                    u_x = wind_data.get('U:0').values.reshape(channel_shape)
                    u_y = wind_data.get('U:1').values.reshape(channel_shape)
                    u_z = wind_data.get('U:2').values.reshape(channel_shape)
                    turb = wind_data.get('k').values.reshape(channel_shape)
                    terrain = np.less(wind_data.get('vtkValidPointMask').values.reshape(channel_shape), 0.5).astype(np.float32)
                    p = wind_data.get('p').values.reshape(channel_shape)
                    epsilon = wind_data.get('epsilon').values.reshape(channel_shape)
                    nut = wind_data.get('nut').values.reshape(channel_shape)

                    # filter out bad terrain pixels, paraview sometimes fails to interpolate the flow although it seems to be correct
                    terrain = np.insert(terrain, 0, np.ones(slice_shape), axis = 0)
                    terrain_old = np.zeros(terrain.shape)
                    terrain_original = np.copy(terrain)

                    # identify bad pixels
                    iter = 0
                    while((np.sum(terrain - terrain_old) > 0) and (iter < 256)):
                        terrain_old = np.copy(terrain)
                        terrain[1:,:] *= terrain[:-1,:]
                        iter += 1

                    # set the value of the bad pixels by interpolating the neighbor cells
                    idx_z, idx_y, idx_x = np.nonzero(terrain[1:,:]-terrain_original[1:,:])
                    idx_1D = idx_x + nx * idx_y + nx * ny * idx_z
                    for ix, iy, iz in zip(idx_x, idx_y, idx_z):
                        denom = 6.0
                        mul = np.ones([6,])

                        i1 = [iz, iy, ix]
                        while(i1[0] * nx * ny + i1[1] * nx + i1[2] in idx_1D):
                            i1[0] += 1
                            if (i1[0] >= nz):
                                i1[0] = nz -1
                                mul[0] = 0.0
                                denom -= 1.0
                                break

                        i2 = [iz, iy, ix]
                        while(i2[0] * nx * ny + i2[1] * nx + i2[2] in idx_1D):
                            i2[0] -= 1
                            if (i2[0] < 0):
                                mul[1] = 0.0
                                denom -= 1.0
                                break

                        i3 = [iz, iy, ix]
                        while(i3[0] * nx * ny + i3[1] * nx + i3[2] in idx_1D):
                            i3[1] += 1
                            if (i3[1] >= ny):
                                i3[1] = ny -1
                                mul[2] = 0.0
                                denom -= 1.0
                                break

                        i4 = [iz, iy, ix]
                        while(i4[0] * nx * ny + i4[1] * nx + i4[2] in idx_1D):
                            i4[1] -= 1
                            if (i4[1] < 0):
                                mul[3] = 0.0
                                denom -= 1.0
                                break

                        i5 = [iz, iy, ix]
                        while(i5[0] * nx * ny + i5[1] * nx + i5[2] in idx_1D):
                            i5[2] += 1
                            if (i5[2] >= nx):
                                i5[2] = nx -1
                                mul[4] = 0.0
                                denom -= 1.0
                                break

                        i6 = [iz, iy, ix]
                        while(i6[0] * nx * ny + i6[1] * nx + i6[2] in idx_1D):
                            i6[2] -= 1
                            if (i6[2] < 0):
                                mul[5] = 0.0
                                denom -= 1.0
                                break

                        i1 = tuple(i1)
                        i2 = tuple(i2)
                        i3 = tuple(i3)
                        i4 = tuple(i4)
                        i5 = tuple(i5)
                        i6 = tuple(i6)

                        u_x[iz, iy, ix] = (mul[0]*u_x[i1] + mul[1]*u_x[i2] + mul[2]*u_x[i3] + mul[3]*u_x[i4] + mul[4]*u_x[i5] + mul[5]*u_x[i6]) / denom
                        u_y[iz, iy, ix] = (mul[0]*u_y[i1] + mul[1]*u_y[i2] + mul[2]*u_y[i3] + mul[3]*u_y[i4] + mul[4]*u_y[i5] + mul[5]*u_y[i6]) / denom
                        u_z[iz, iy, ix] = (mul[0]*u_z[i1] + mul[1]*u_z[i2] + mul[2]*u_z[i3] + mul[3]*u_z[i4] + mul[4]*u_z[i5] + mul[5]*u_z[i6]) / denom
                        turb[iz, iy, ix] = (mul[0]*turb[i1] + mul[1]*turb[i2] + mul[2]*turb[i3] + mul[3]*turb[i4] + mul[4]*turb[i5] + mul[5]*turb[i6]) / denom
                        p[iz, iy, ix] = (mul[0]*p[i1] + mul[1]*p[i2] + mul[2]*p[i3] + mul[3]*p[i4] + mul[4]*p[i5] + mul[5]*p[i6]) / denom
                        epsilon[iz, iy, ix] = (mul[0]*epsilon[i1] + mul[1]*epsilon[i2] + mul[2]*epsilon[i3] + mul[3]*epsilon[i4] + mul[4]*epsilon[i5] + mul[5]*epsilon[i6]) / denom
                        nut[iz, iy, ix] = (mul[0]*nut[i1] + mul[1]*nut[i2] + mul[2]*nut[i3] + mul[3]*nut[i4] + mul[4]*nut[i5] + mul[5]*nut[i6]) / denom

                    if verbose:
                        print('------------------------------------')
                        print('File ', member.name, ' is ok, contains ', len(idx_1D), ' bad pixels.')

                    is_wind = np.less(terrain, 0.5).astype(np.float32)

                    # if the terrain channel should be a binary class or a distance field
                    if boolean_terrain:
                        distance_field_in = is_wind[1:,:]
                    else:
                        distance_field_in = ndimage.distance_transform_edt(is_wind).astype(np.float32)
                        distance_field_in = distance_field_in[1:, :]

                    # stack the data for easy iteration
                    out = np.stack([distance_field_in, u_x, u_y, u_z, turb, p, epsilon, nut])

                    # add each channel to the hdf5 file for the current sample
                    for k, channel in enumerate(channels):
                        ouput_file[sample].create_dataset(channel, data=out[k,:,:,:], compression= compression_type)

                    # add the grid size to the hdf5 file for the current sample
                    ouput_file[sample].create_dataset('ds', data=(dx, dy, dz))

                    # add the zero sample if new terrain and creating zero samples is enabled
                    if create_zero_samples and new_terrain:
                        zero_sample = sample.replace('_W01_', '_W00_')

                        # add zero sample to hdf5 file
                        ouput_file.create_group(zero_sample)

                        # add all the channels to the zero sample
                        for k, channel in enumerate(channels):
                            if k==0:
                                # add terrain to the hdf5 file for the zero sample
                                ouput_file[zero_sample].create_dataset(channel, data=out[k,:,:,:], compression=compression_type)
                            else:
                                # add terrain to the hdf5 file for the zero sample
                                ouput_file[zero_sample].create_dataset(channel, data= np.zeros_like(out[k,:,:,:]), compression=compression_type)

                        # add the grid size to the hdf5 file for the zero sample
                        ouput_file[zero_sample].create_dataset('ds', data=(dx, dy, dz))
                        zero_samples_created += 1

        if ((i % np.ceil(num_files/10.0)) == 0.0):
            print(trunc((i+1)/num_files*100), '%')

    print('INFO: Finished parsing all the files, rejecting', rejection_counter, 'out of', num_files)
    if create_zero_samples:
        print('INFO: Created ', zero_samples_created, 'zero sample(s)')

    # close hdf5 file
    ouput_file.close()

def sample_dataset(dbloader, outfile, n_sampling_rounds, compress, input_channels, label_channels):
    # define compression
    if compress:
        compression_type = "lzf"
    else:
        compression_type = None

    # check output file extension
    if not outfile.endswith('.hdf5'):
        outfile = outfile + '.hdf5'

    # create h5 file where all the data wll be stored
    if os.path.exists(outfile):
        print('Removing old file')
        os.remove(outfile)
    ouput_file = h5py.File(outfile)

    # get the channels list (terrain + label channels)
    channels = [input_channels[0]] + label_channels

    for j in range(n_sampling_rounds):
        for i, data in enumerate(dbloader):
            input, label, ds, name = data

            resampled_name = name + '_' + str(j)

            # create the dataset group
            ouput_file.create_group(resampled_name)

            # combine the data
            out = torch.cat([input[0].unsqueeze(0), label], dim=0).numpy()

            # add each channel to the hdf5 file for the current sample
            for k, channel in enumerate(channels):
                ouput_file[resampled_name].create_dataset(channel, data=out[k,:,:,:], compression=compression_type)

            # add the grid size to the hdf5 file for the current sample
            ouput_file[resampled_name].create_dataset('ds', data=ds.numpy())

    # close hdf5 file
    ouput_file.close()
