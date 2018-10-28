import numpy as np
import torch

class DataInterpolation:
    def __init__(self, device, num_channels, nx, ny, nz):
        self.__ny = ny

        self.__fac_1_x = torch.from_numpy(np.linspace(1.0, 0.0, nx, dtype=np.float32)).unsqueeze(0).unsqueeze(0).expand(num_channels, -1, -1).to(device)
        self.__fac_2_x = (1.0 - self.__fac_1_x).to(device)

        self.__fac_1_y = torch.from_numpy(
            np.linspace(1.0, 0.0, ny, dtype=np.float32)).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(num_channels, nz, -1, nx).to(device)
        self.__fac_2_y = (1.0 - self.__fac_1_y).to(device)

    def edge_interpolation(self, input):
        '''
        Interpolate the four vertical edges over the full domain
        '''
        if (len(input.shape) > 4):
            print('Error: Edge interpolation does not support batch operations.')

        face1 = torch.bmm(input[:,:,0,0].unsqueeze(-1), self.__fac_1_x) + torch.bmm(input[:,:,0,-1].unsqueeze(-1), self.__fac_2_x)
        face2 = torch.bmm(input[:,:,-1,0].unsqueeze(-1), self.__fac_1_x) + torch.bmm(input[:,:,-1,-1].unsqueeze(-1), self.__fac_2_x)

        face1 = face1.unsqueeze(2).expand(-1,-1,self.__ny,-1)
        face2 = face2.unsqueeze(2).expand(-1,-1,self.__ny,-1)

        return self.__fac_1_y * face1 + self.__fac_2_y * face2
