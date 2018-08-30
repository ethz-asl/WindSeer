from io import BytesIO
import lz4.frame
import numpy as np
import sys
import tarfile
import torch
from torch.utils.data.dataset import Dataset

'''
TODO: try if it is feasible also to store the filedescriptors or how much faster it will make the dataloading (using Lock when accessing the file descriptors
'''
class MyDataset(Dataset):
    def __init__(self, filename, stride_hor = 1, stride_vert = 1, turbulence_label = False, scaling_uhor = 1.0, scaling_uz = 1.0, scaling_nut = 1.0, compressed = True):
        try:
            tar = tarfile.open(filename, 'r')
        except IOError as e:
            print('I/O error({0}): {1}: {2}'.format(e.errno, e.strerror, filename))
            sys.exit()

        self.__filename = filename
        self.__num_files = len(tar.getnames())
        self.__memberslist = tar.getmembers()

        self.__turbulence_label = turbulence_label
        self.__scaling_uhor = scaling_uhor
        self.__scaling_uz = scaling_uz
        self.__scaling_nut = scaling_nut

        self.__stride_hor = stride_hor
        self.__stride_vert = stride_vert

        self.__compressed = compressed

        print('MyDataset: ' + filename + ' contains {} samples'.format(self.__num_files))

    def __getitem__(self, index):
        tar = tarfile.open(self.__filename, 'r')
        file = tar.extractfile(self.__memberslist[index])
        print(self.__memberslist[index].name)

        if self.__compressed:
            data = torch.load(BytesIO(lz4.frame.decompress(file.read())))

        else:
            data = torch.load(file)

        tar.close()
        del tar
        del file

        if (len(list(data.size())) > 3):
            data[1, :, :, :] /= self.__scaling_uhor # in u_x
            data[2, :, :, :] /= self.__scaling_uhor # in u_y
            data[3, :, :, :] /= self.__scaling_uz # in u_z
            data[4, :, :, :] /= self.__scaling_uhor # label u_x
            data[5, :, :, :] /= self.__scaling_uhor # label u_y
            data[6, :, :, :] /= self.__scaling_uz # label u_z
            data[7, :, :, :] /= self.__scaling_nut # label turbulence

            if self.__turbulence_label:
                return data[:4,::self.__stride_vert,::self.__stride_hor, ::self.__stride_hor], data[4:,::self.__stride_vert,::self.__stride_hor, ::self.__stride_hor]
            else:
                return data[:4,::self.__stride_vert,::self.__stride_hor, ::self.__stride_hor], data[4:7,::self.__stride_vert,::self.__stride_hor, ::self.__stride_hor]

        else:
            # 2D data
            input = data[[0,1,3], :, :]

            if self.__turbulence_label:
                output = data[[4,6,7], :, :]
                output[2, :, :] /= self.__scaling_nut

            else:
                output = data[[4,6], :, :]

            del data

            # apply scaling
            input[1, :, :] /= self.__scaling_uhor
            input[2, :, :] /= self.__scaling_uz
            output[0, :, :] /= self.__scaling_uhor
            output[1, :, :] /= self.__scaling_uz

            # scale the output if necessary
            input = input[:,::self.__stride_vert, ::self.__stride_hor]
            output = output[:,::self.__stride_vert, ::self.__stride_hor]

            return input, output

    def __len__(self):
        return self.__num_files
