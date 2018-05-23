import numpy as np
import pandas as pd
import torch
import zipfile

class MyDataset():
    def __init__(self, filename, nx = 128, nz = 64, scaling_ux = 1.0, scaling_uz = 1.0, scaling_nut = 1.0):
        self.__filename = filename

        with zipfile.ZipFile(self.__filename, "r") as zip:
            self.__nameslist = zip.namelist()

        self.__types = {"p": np.float32,
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
        
        self.__nx = nx
        self.__nz = nz
        self.__scaling_ux = scaling_ux
        self.__scaling_uz = scaling_uz
        self.__scaling_nut = scaling_nut


    def __getitem__(self, index):
        with zipfile.ZipFile(self.__filename, "r") as zip:
            f = zip.open(self.__nameslist[index])
            wind_data = pd.read_csv(f, header=0, dtype = self.__types)

            if 'U:0' not in wind_data.keys():
                print('U:0 not in {0}'.format(self.__nameslist[index]))
                raise IOError

            # generate the labels
            u_x_out = torch.from_numpy(wind_data.get('U:0').values.reshape([self.__nz, self.__nx])).unsqueeze(0) / self.__scaling_ux
            u_z_out = torch.from_numpy(wind_data.get('U:2').values.reshape([self.__nz, self.__nx])).unsqueeze(0) / self.__scaling_uz
            turbelence_viscosity_out = torch.from_numpy(wind_data.get('nut').values.reshape([self.__nz, self.__nx])).unsqueeze(0) / self.__scaling_nut

            label = torch.cat((u_x_out, u_z_out, turbelence_viscosity_out), 0)

            # generate the input
            is_wind_in = torch.from_numpy(wind_data.get('vtkValidPointMask').values.reshape([self.__nz, self.__nx]).astype(np.float32)).unsqueeze(0)

            u_x_in = u_x_out[:,:,0].unsqueeze(-1).repeat(1, 1, u_x_out.size()[2])
            u_z_in = u_z_out[:,:,0].unsqueeze(-1).repeat(1, 1, u_z_out.size()[2])

            # alternative method, needs to be evaluated on the gpu which one is faster
#             u_x_in = u_x_out.clone()
#             u_z_in = u_z_out.clone()
#              
#             for i in range(1, self.__nx):
#                u_x_in[:,:,i] = u_x_in[:,:,0] 
#                u_z_in[:,:,i] = u_z_in[:,:,0] 

            input = torch.cat((is_wind_in, u_x_in, u_z_in), 0)

            return input, label

    def __len__(self):
        return len(self.__nameslist)