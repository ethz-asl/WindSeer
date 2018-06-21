from io import BytesIO
import numpy as np
import sys
import tarfile
import torch
from torch.utils.data.dataset import Dataset

'''
TODO: try if it is feasable also to store the filedescriptors or how much faster it will make the dataloading (using Lock when accessing the file descriptors
'''
class MyDataset(Dataset):
    def __init__(self, filename, scaling_ux = 1.0, scaling_uz = 1.0, scaling_nut = 1.0):
        try:
            tar = tarfile.open(filename, 'r')
        except IOError as e:
            print('I/O error({0}): {1}: {2}'.format(e.errno, e.strerror, filename))
            sys.exit()

        self.__filename = filename
        self.__num_files = len(tar.getnames())
        self.__memberslist = tar.getmembers()

        self.__scaling_ux = scaling_ux
        self.__scaling_uz = scaling_uz
        self.__scaling_nut = scaling_nut

    def __getitem__(self, index):
        tar = tarfile.open(self.__filename, 'r')
        file = tar.extractfile(self.__memberslist[index])
        data = torch.load(file)

        # split into input output
        input = data[:3, :, :]
        output = data[3:5, :, :]

        # apply scaling
        input[1, :, :] /= self.__scaling_ux
        input[2, :, :] /= self.__scaling_uz
        output[0, :, :] /= self.__scaling_ux
        output[1, :, :] /= self.__scaling_uz
        #output[2, :, :] /= self.__scaling_nut

        return input, output

    def __len__(self):
        return self.__num_files
