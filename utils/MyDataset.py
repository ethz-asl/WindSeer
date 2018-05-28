from io import BytesIO
import numpy as np
import sys
import tarfile
import torch

'''
TODO: try if it is feasable also to store the filedescriptors or how much faster it will make the dataloading
'''
class MyDataset():
    def __init__(self, filename, scaling_ux = 1.0, scaling_uz = 1.0, scaling_nut = 1.0):
        try:
            self.__tar = tarfile.open(filename, 'r')
        except IOError as e:
            print('I/O error({0}): {1}: {2}'.format(e.errno, e.strerror, filename))
            sys.exit()

        self.__num_files = len(self.__tar.getnames())
        self.__memberslist = self.__tar.getmembers()

        self.__scaling_ux = scaling_ux
        self.__scaling_uz = scaling_uz
        self.__scaling_nut = scaling_nut

#         self.__fileslist = []
#         for i in range(self.__num_files):
#             self.__fileslist.append(self.__tar.extractfile(self.__memberslist[i]))

    def __del__(self):
        try:
            self.__tar.close()
        except:
            pass

    def __getitem__(self, index):
#         self.__fileslist[index].seek(0)
#         data = torch.load(self.__fileslist[index])
#         self.__fileslist[index].seek(0)

        file = self.__tar.extractfile(self.__memberslist[index])
        data = torch.load(file)
#         file.seek(0)

        # split into input output
        input = data[:3, :, :]
        output = data[3:, :, :]

        # apply scaling
        input[1, :, :] /= self.__scaling_ux
        input[2, :, :] /= self.__scaling_uz
        output[0, :, :] /= self.__scaling_ux
        output[1, :, :] /= self.__scaling_uz
        output[2, :, :] /= self.__scaling_nut

        return input, output

    def __len__(self):
        return self.__num_files
