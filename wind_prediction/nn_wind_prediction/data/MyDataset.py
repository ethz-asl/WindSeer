from __future__ import print_function

from nn_wind_prediction.utils.interpolation import DataInterpolation

from io import BytesIO
import lz4.frame
import numpy as np
import random
import sys
import tarfile
import torch
from torch.utils.data.dataset import Dataset

class MyDataset(Dataset):
    '''
    Class to handle the dataset with containing velocities and the turbulent kinetic energy (k).

    The dataset is a single tar file containing all the samples stored as 4D-pytorch tensors, possibly compressed with LZ4.
    The four dimensions are: [channels, z, y, x].
    The channel ordering is: [terrain, ux, uy, uz, k].

    The raw data is split up to an input tensor and output tensor. The input tensor contains the velocities and the terrain
    information in the following order: [terrain, ux_in, *uy_in, uz_in], where uy_in is only present in the 3D case.
    The number of output channels is configurable but always contains at least the velocities:
    [ux_out, *uy_out, uz_out, *k, *dx, *dy, *dz].
    uy_out is only present in the 3D case and it can be chosen if also the turbulent kinetic energy (k) and the grid sizes
    (dx, dy, dz) are contained in the output tensor.

    TODO:
    - Check if it is feasible also to store the filedescriptors or how much faster it will make the dataloading (using Lock when accessing the file descriptors
    - Reimplement the 2D data handling
    '''

    def __init__(self, device, filename, nx, ny, nz, input_mode = 0, subsample = False, augmentation = False,
                 stride_hor = 1, stride_vert = 1, turbulence_label = False,
                 scaling_uhor = 1.0, scaling_uz = 1.0, scaling_k = 1.0,
                 compressed = True, use_grid_size = False, return_grid_size = False, return_name = False):
        '''
        Params:
            device:
                Device (CPU or GPU) on which the tensor operations are executed
            filename:
                The name of the tar file containing the dataset
            nx:
                Number of grid points in x-direction of the output
            ny:
                Number of grid points in y-direction of the output
            nz:
                Number of grid points in z-direction of the output
            input_mode:
                Indicates how the input is constructed. The following modes are currently implemented:
                    0: The inflow condition is copied over the full domain
                    1: The vertical edges are interpolated over the full domain
            subsample:
                If true a region with the size of (nx, ny, nz) is sampled from the input data
            augmentation:
                If true the data is augmented using flipping in x/y direction and rotation aroung z
            stride_hor:
                Horizontal stride, used to reduce the size of the output tensors
            stride_vert:
                Vertical stride, used to reduce the size of the output tensors
            turbulence_label:
                Specifies if the turbulent kinetic energy is contained in the output
            scaling_uhor:
                Scaling factor for the horizontal velocity components
            scaling_uz:
                Scaling factor for the vertical velocity component
            scaling_k:
                Scaling factor for the turbulent kinetic energy
            compressed:
                Specifies if the input tensors are compressed using LZ4
            use_grid_size:
                Specifies if the turbulent kinetic energy is contained in the output
            return_grid_size:
                If true a tuple of the grid size is returned in addition to the input and output tensors
            return_name:
                Return the filename of the sample
        '''
        try:
            tar = tarfile.open(filename, 'r')
        except IOError as e:
            print('I/O error({0}): {1}: {2}'.format(e.errno, e.strerror, filename))
            sys.exit()

        self.__device = device

        self.__filename = filename
        self.__num_files = len(tar.getnames())
        self.__memberslist = tar.getmembers()

        self.__nx = nx
        self.__ny = ny
        self.__nz = nz

        self.__input_mode = input_mode
        self.__subsample = subsample
        self.__augmentation = augmentation

        self.__turbulence_label = turbulence_label
        self.__scaling_uhor = scaling_uhor
        self.__scaling_uz = scaling_uz
        self.__scaling_k = scaling_k

        self.__stride_hor = stride_hor
        self.__stride_vert = stride_vert

        self.__compressed = compressed

        self.__return_grid_size = return_grid_size
        self.__use_grid_size = use_grid_size

        self.__return_name = return_name

        self.__rand = random.SystemRandom()

        # interpolator for the three input velocities
        self.__interpolator = DataInterpolation(device, 3, nx, ny, nz)

        print('MyDataset: ' + filename + ' contains {} samples'.format(self.__num_files))

    def __getitem__(self, index):
        tar = tarfile.open(self.__filename, 'r')
        file = tar.extractfile(self.__memberslist[index])

        # load the data
        if self.__compressed:
            data, ds = torch.load(BytesIO(lz4.frame.decompress(file.read())))

        else:
            data, ds = torch.load(file)

        data = data.to(self.__device)

        data_shape = data[0,:].shape
        if (len(data_shape) == 3):
            # scale the data according to the specifications
            data[1, :, :, :] /= self.__scaling_uhor # in u_x
            data[2, :, :, :] /= self.__scaling_uhor # in u_y
            data[3, :, :, :] /= self.__scaling_uz # in u_z
            data[4, :, :, :] /= self.__scaling_k # label turbulence

            # downscale if requested
            data = data[:,::self.__stride_vert,::self.__stride_hor, ::self.__stride_hor]

            # determine the region for the output
            if self.__subsample:
                start_x = self.__rand.randint(0,data_shape[2]-self.__nx)
                start_y = self.__rand.randint(0,data_shape[1]-self.__ny)
                start_z = self.__rand.randint(0,data_shape[0]-self.__nz)

                data = data[:, start_z:start_z+self.__nz,  start_y:start_y+self.__ny,  start_x:start_x+self.__nx]
            else:
                data = data[:,:self.__nz, :self.__ny, :self.__nx]

            # generate the input channels
            if (self.__input_mode == 0):
                # copy the inflow condition across the full domain
                input = torch.cat([data[0,:,:,:].unsqueeze(0), data[1:4,:,:,0].unsqueeze(-1).expand(-1,-1,-1,self.__nx)])

            elif (self.__input_mode == 1):
                # interpolating the vertical edges
                input = torch.cat([data[0,:,:,:].unsqueeze(0), self.__interpolator.edge_interpolation(data[1:4,:,:,:])])

            else:
                print('MyDataset Error: Input mode ', self.__input_mode, ' is not supported')
                sys.exit()

            # append the grid size
            if self.__use_grid_size:
                dx = torch.full((self.__nz, self.__ny, self.__nx), float(ds[0])).unsqueeze(0).to(self.__device)
                dy = torch.full((self.__nz, self.__ny, self.__nx), float(ds[1])).unsqueeze(0).to(self.__device)
                dz = torch.full((self.__nz, self.__ny, self.__nx), float(ds[2])).unsqueeze(0).to(self.__device)
                input = torch.cat([input, dx, dy, dz])

            if self.__turbulence_label:
                output = data[1:,:,:,:]
            else:
                output = data[1:4,:,:,:]

            # data augmentation
            if self.__augmentation:
                # flip in x-direction
                if (self.__rand.randint(0,1)):
                    output = torch.from_numpy(np.flip(output.cpu().numpy(), 3).copy())
                    input = torch.from_numpy(np.flip(input.cpu().numpy(), 3).copy())
                    output[0,:,:,:] *= -1.0
                    input[1,:,:,:] *= -1.0

                # flip in y-direction
                if (self.__rand.randint(0,1)):
                    output = torch.from_numpy(np.flip(output.cpu().numpy(), 2).copy())
                    input = torch.from_numpy(np.flip(input.cpu().numpy(), 2).copy())
                    output[1,:,:,:] *= -1.0
                    input[2,:,:,:] *= -1.0

                # rotate 90 degrees
                if (self.__rand.randint(0,1)):
                    output = torch.from_numpy(output.cpu().numpy().swapaxes(-2,-1)[...,::-1].copy())
                    output = torch.cat([-output[1,:,:,:].unsqueeze(0), output[0,:,:,:].unsqueeze(0), output[2:,:,:,:]])
                    input = torch.from_numpy(input.cpu().numpy().swapaxes(-2,-1)[...,::-1].copy())
                    input = torch.cat([input[0,:,:,:].unsqueeze(0), -input[2,:,:,:].unsqueeze(0), input[1,:,:,:].unsqueeze(0), input[3:,:,:,:]])
                    ds = (ds[1], ds[0], ds[2])

            if self.__return_grid_size:
                if self.__return_name:
                    return input, output, ds, self.__memberslist[index].name
                else:
                    return input, output, ds
            else:
                if self.__return_name:
                    return input, output, self.__memberslist[index].name
                else:
                    return input, output

        elif (len(data_shape) == 2):
            print('MyDataset Error: 2D data handling is not implemented yet')
            sys.exit()
        else:
            print('MyDataset Error: Data dimension of ', len(data_shape), ' is not supported')
            sys.exit()

    def __len__(self):
        return self.__num_files
