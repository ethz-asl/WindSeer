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

    __default_device = 'cpu'
    __default_nx = 64
    __default_ny = 64
    __default_nz = 64
    __default_input_mode = 0
    __default_subsample = False
    __default_augmentation = False
    __default_stride_hor = 1
    __default_stride_vert = 1
    __default_turbulence_label = True
    __default_scaling_uhor = 1.0
    __default_scaling_uz = 1.0
    __default_scaling_turb = 1.0
    __default_scaling_terrain = 1.0
    __default_compressed = False
    __default_use_grid_size = False
    __default_return_grid_size = False
    __default_return_name = False
    __default_autoscale = False
    __default_normalize_terrain = False

    def __init__(self, filename, **kwargs):
        '''
        Params:
            device:
                Device (CPU or GPU) on which the tensor operations are executed, default 'cpu'
            filename (required):
                The name of the tar file containing the dataset
            nx:
                Number of grid points in x-direction of the output, default 64
            ny:
                Number of grid points in y-direction of the output, default 64
            nz:
                Number of grid points in z-direction of the output, default 64
            input_mode:
                Indicates how the input is constructed. The following modes are currently implemented:
                    0: The inflow condition is copied over the full domain
                    1: The vertical edges are interpolated over the full domain, default 0
            subsample:
                If true a region with the size of (nx, ny, nz) is sampled from the input data, default False
            augmentation:
                If true the data is augmented using flipping in x/y direction and rotation aroung z, default False
            stride_hor:
                Horizontal stride, used to reduce the size of the output tensors, default 1
            stride_vert:
                Vertical stride, used to reduce the size of the output tensors, default 1
            turbulence_label:
                Specifies if the turbulent kinetic energy is contained in the output, default True
            scaling_uhor:
                Scaling factor for the horizontal velocity components, default 1.0
            scaling_uz:
                Scaling factor for the vertical velocity component, default 1.0
            scaling_turb:
                Scaling factor for the turbulent kinetic energy, default 1.0
            compressed:
                Specifies if the input tensors are compressed using LZ4, default False
            use_grid_size:
                Specifies if the turbulent kinetic energy is contained in the output, default False
            return_grid_size:
                If true a tuple of the grid size is returned in addition to the input and output tensors, default False
            return_name:
                Return the filename of the sample, default False
            autoscale:
                Automatically scale the input and return the scale, default False
            normalize_terrain:
                Normalize the values of the terrain output channel, default False
        '''
        self.__filename = filename

        try:
            tar = tarfile.open(filename, 'r')
        except IOError as e:
            print('I/O error({0}): {1}: {2}'.format(e.errno, e.strerror, filename))
            sys.exit()

        try:
            verbose = kwargs['verbose']
        except KeyError:
            verbose = False

        try:
            self.__device = kwargs['device']
        except KeyError:
            self.__device = self.__default_device
            if verbose:
                print('MyDataset: device not present in kwargs, using default value:', self.__default_device)

        try:
            self.__nx = kwargs['nx']
        except KeyError:
            self.__nx = self.__default_nx
            if verbose:
                print('MyDataset: nx not present in kwargs, using default value:', self.__default_nx)

        try:
            self.__ny = kwargs['ny']
        except KeyError:
            self.__ny = self.__default_ny
            if verbose:
                print('MyDataset: ny not present in kwargs, using default value:', self.__default_ny)

        try:
            self.__nz = kwargs['nz']
        except KeyError:
            self.__nz = self.__default_nz
            if verbose:
                print('MyDataset: nz not present in kwargs, using default value:', self.__default_nz)

        try:
            self.__input_mode = kwargs['input_mode']
        except KeyError:
            self.__input_mode = self.__default_input_mode
            if verbose:
                print('MyDataset: input_mode not present in kwargs, using default value:', self.__default_input_mode)

        try:
            self.__subsample = kwargs['subsample']
        except KeyError:
            self.__subsample = self.__default_subsample
            if verbose:
                print('MyDataset: subsample not present in kwargs, using default value:', self.subsample)

        try:
            self.__augmentation = kwargs['augmentation']
        except KeyError:
            self.__augmentation = self.__default_augmentation
            if verbose:
                print('MyDataset: augmentation not present in kwargs, using default value:', self.__default_augmentation)

        try:
            self.__turbulence_label = kwargs['turbulence_label']
        except KeyError:
            self.__turbulence_label = self.__default_turbulence_label
            if verbose:
                print('MyDataset: turbulence_label not present in kwargs, using default value:', self.__default_turbulence_label)

        try:
            self.__scaling_uhor = kwargs['scaling_uhor']
        except KeyError:
            self.__scaling_uhor = self.__default_scaling_uhor
            if verbose:
                print('MyDataset: scaling_uhor not present in kwargs, using default value:', self.__default_scaling_uhor)

        try:
            self.__scaling_uz = kwargs['scaling_uz']
        except KeyError:
            self.__scaling_uz = self.__default_scaling_uz
            if verbose:
                print('MyDataset: scaling_uz not present in kwargs, using default value:', self.__default_scaling_uz)

        try:
            self.__scaling_turb = kwargs['scaling_turb']
        except KeyError:
            self.__scaling_turb = self.__default_scaling_turb
            if verbose:
                print('MyDataset: scaling_turb not present in kwargs, using default value:', self.__default_scaling_turb)

        try:
            self.__scaling_terrain = kwargs['scaling_terrain']
        except KeyError:
            self.__scaling_terrain = self.__default_scaling_terrain
            if verbose:
                print('MyDataset: scaling_terrain not present in kwargs, using default value:', self.__default_scaling_terrain)

        try:
            self.__stride_hor = kwargs['stride_hor']
        except KeyError:
            self.__stride_hor = self.__default_stride_hor
            if verbose:
                print('MyDataset: stride_hor not present in kwargs, using default value:', self.__default_stride_hor)

        try:
            self.__stride_vert = kwargs['stride_vert']
        except KeyError:
            self.__stride_vert = self.__default_stride_vert
            if verbose:
                print('MyDataset: stride_vert not present in kwargs, using default value:', self.__default_stride_vert)

        try:
            self.__compressed = kwargs['compressed']
        except KeyError:
            self.__compressed = self.__default_compressed
            if verbose:
                print('MyDataset: compressed not present in kwargs, using default value:', self.__default_compressed)

        try:
            self.__return_grid_size = kwargs['return_grid_size']
        except KeyError:
            self.__return_grid_size = self.__default_return_grid_size
            if verbose:
                print('MyDataset: return_grid_size not present in kwargs, using default value:', self.__default_return_grid_size)

        try:
            self.__use_grid_size = kwargs['use_grid_size']
        except KeyError:
            self.__use_grid_size = self.__default_use_grid_size
            if verbose:
                print('MyDataset: use_grid_size not present in kwargs, using default value:', self.__default_use_grid_size)

        try:
            self.__return_name = kwargs['return_name']
        except KeyError:
            self.__return_name = self.__default_return_name
            if verbose:
                print('MyDataset: return_name not present in kwargs, using default value:', self.__default_return_name)

        try:
            self.__autoscale = kwargs['autoscale']
        except KeyError:
            self.__autoscale = self.__default_autoscale
            if verbose:
                print('MyDataset: autoscale not present in kwargs, using default value:', self.__default_autoscale)

        try:
            self.__normalize_terrain = kwargs['normalize_terrain']
        except KeyError:
            self.__normalize_terrain = self.__default_normalize_terrain
            if verbose:
                print('MyDataset: normalize_terrain not present in kwargs, using default value:', self.__default_normalize_terrain)

        # extract data from the tar file
        self.__num_files = len(tar.getnames())
        self.__memberslist = tar.getmembers()

        # initialize random number generator used for the subsampling
        self.__rand = random.SystemRandom()

        # interpolator for the three input velocities
        self.__interpolator = DataInterpolation(self.__device, 3, self.__nx, self.__ny, self.__nz)

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
            if self.__autoscale:
                scale = self.__get_scale(data[1:4, :, :, :])
            else:
                scale = 1.0

            data[0, :, :, :] /= self.__scaling_terrain # terrain
            data[1, :, :, :] /= scale * self.__scaling_uhor # in u_x
            data[2, :, :, :] /= scale * self.__scaling_uhor # in u_y
            data[3, :, :, :] /= scale * self.__scaling_uz # in u_z
            data[4, :, :, :] /= scale * scale * self.__scaling_turb # label turbulence

            if self.__normalize_terrain:
                data[0, :, :, :] /= data[0, :, :, :].max()

            # downscale if requested
            data = data[:,::self.__stride_vert,::self.__stride_hor, ::self.__stride_hor]

            # determine the region for the output
            if self.__subsample:
                start_x = self.__rand.randint(0,data_shape[2]-self.__nx)
                start_y = self.__rand.randint(0,data_shape[1]-self.__ny)
                # gauss distribution
#                 start_z = int(min(np.abs(self.__rand.gauss(0.0,(data_shape[0]-self.__nz)/3.0)), (data_shape[0]-self.__nz)))
                # triangle distribution
                start_z = int(self.__rand.triangular(0,(data_shape[0]-self.__nz),0))

                data = data[:, start_z:start_z+self.__nz,  start_y:start_y+self.__ny,  start_x:start_x+self.__nx]
            else:
                data = data[:,:self.__nz, :self.__ny, :self.__nx]

            # generate the input channels
            if (self.__input_mode == 0):
                # copy the inflow condition across the full domain
                input = torch.cat([data[0,:,:,:].unsqueeze(0), data[1:4,:,:,0].unsqueeze(-1).expand(-1,-1,-1,self.__nx)])

            elif (self.__input_mode == 1):
                # This interpolation is slower (at least on a cpu)
#                 input = torch.cat([data[0,:,:,:].unsqueeze(0),
#                                    self.__interpolator.edge_interpolation_batch(data[1:4,:,:,:].unsqueeze(0)).squeeze()])

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
                input_permute = [0,2,1,3,4,5,6]
            else:
                input_permute = [0,2,1,3]

            if self.__turbulence_label:
                output = data[1:,:,:,:]
            else:
                output = data[1:4,:,:,:]

            # data augmentation
            if self.__augmentation:
                # flip in x-direction
                if (self.__rand.randint(0,1)):
                    output = output.flip(3)
                    input = input.flip(3)

                    output[0,:,:,:] *= -1.0
                    input[1,:,:,:] *= -1.0

                # flip in y-direction
                if (self.__rand.randint(0,1)):
                    output = output.flip(2)
                    input = input.flip(2)

                    output[1,:,:,:] *= -1.0
                    input[2,:,:,:] *= -1.0

                # rotate 90 degrees
                if (self.__rand.randint(0,1)):
                    output = output.transpose(2,3).flip(3)
                    output = output[[1,0,2,3]]
                    output[0,:,:,:] *= -1.0

                    input = input.transpose(2,3).flip(3)
                    input = input[input_permute]
                    input[1,:,:,:] *= -1.0

                    ds = (ds[1], ds[0], ds[2])

            out = [input, output]

            if self.__autoscale:
                out.append(scale)

            if self.__return_grid_size:
                out.append(ds)

            if self.__return_name:
                out.append(self.__memberslist[index].name)

            return out

        elif (len(data_shape) == 2):
            print('MyDataset Error: 2D data handling is not implemented yet')
            sys.exit()
        else:
            print('MyDataset Error: Data dimension of ', len(data_shape), ' is not supported')
            sys.exit()

    def get_name(self, index):
        return self.__memberslist[index].name

    def __len__(self):
        return self.__num_files

    def __get_scale(self, x):
        shape = x.shape

        corners = torch.index_select(x, 2, torch.tensor([0,shape[2]-1]))
        corners = torch.index_select(corners, 3, torch.tensor([0,shape[3]-1]))

        return corners.norm(dim=0).mean(dim=0).max()
