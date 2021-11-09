import numpy as np
import torch
import torch.nn.functional as F
import sys
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial.qhull import QhullError

class DataInterpolation:
    def __init__(self, device, num_channels, nx, ny, nz):
        self.__ny = ny

        self.__fac_1_x = torch.from_numpy(np.linspace(1.0, 0.0, nx, dtype=np.float32)).unsqueeze(0).unsqueeze(0).expand(num_channels, -1, -1).to(device)
        self.__fac_2_x = (1.0 - self.__fac_1_x).to(device)

        self.__fac_1_y = torch.from_numpy(
            np.linspace(1.0, 0.0, ny, dtype=np.float32)).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(num_channels, nz, -1, nx).to(device)
        self.__fac_2_y = (1.0 - self.__fac_1_y).to(device)

        self.num_channels = num_channels

    def edge_interpolation(self, input):
        '''
        Interpolate the four vertical edges over the full domain
        '''
        if (len(input.shape) > 4):
            print('Error: Edge interpolation does not support batch operations.')
            sys.exit()

        face1 = torch.bmm(input[:,:,0,0].unsqueeze(-1), self.__fac_1_x) + torch.bmm(input[:,:,0,-1].unsqueeze(-1), self.__fac_2_x)
        face2 = torch.bmm(input[:,:,-1,0].unsqueeze(-1), self.__fac_1_x) + torch.bmm(input[:,:,-1,-1].unsqueeze(-1), self.__fac_2_x)

        face1 = face1.unsqueeze(2).expand(-1,-1,self.__ny,-1)
        face2 = face2.unsqueeze(2).expand(-1,-1,self.__ny,-1)

        return self.__fac_1_y * face1 + self.__fac_2_y * face2

    def edge_interpolation_batch(self, input):
        '''
        Interpolate the four vertical edges over the full domain
        For a single batch interpolation use the non-batch version as it is faster, at least on the cpu.
        '''
        shape = input.shape

        if (len(shape) != 5):
            print('Error: Edge interpolation batch expects a 5D input.')
            sys.exit()

        corners = input.index_select(3, torch.tensor([0,shape[3]-1])).index_select(4, torch.tensor([0,shape[4]-1]))

        return F.interpolate(corners, size=shape[2:], mode='trilinear', align_corners=True)

def interpolate_sparse_data(data, mask, grid_dimensions, linear=True):
    if len(data.squeeze().shape) != 4:
        raise ValueError('The prediction is assumed to be a 4D tensor (channels, z, y, x)')

    indices = torch.nonzero(mask)
    points = indices.float()
    points[:, 0] *= grid_dimensions[2]
    points[:, 1] *= grid_dimensions[1]
    points[:, 2] *= grid_dimensions[0]

    Z, Y, X = np.meshgrid(np.linspace(0, mask.shape[0]-1, mask.shape[0])*grid_dimensions[2].item(),
                          np.linspace(0, mask.shape[1]-1, mask.shape[1])*grid_dimensions[1].item(),
                          np.linspace(0, mask.shape[2]-1, mask.shape[2])*grid_dimensions[0].item(), indexing='ij')

    data_interpolated = torch.zeros_like(data)

    # loop over channels
    for i in range(data.shape[0]):
        inter_nearest = NearestNDInterpolator(points, data[i, indices[:,0], indices[:,1], indices[:,2]].detach().cpu().numpy())

        if linear:
            try:
                inter_linear = LinearNDInterpolator(points, data[i, indices[:,0], indices[:,1], indices[:,2]].detach().cpu().numpy())
                vals = inter_linear(Z, Y, X)
                vals_nearest = inter_nearest(Z, Y, X)
                mask = np.isnan(vals)
                vals[mask] = vals_nearest[mask]
            except QhullError:
                # if there is no convex hull only nearest interpolation is possible
                vals = inter_nearest(Z, Y, X)
        else:
            vals = inter_nearest(Z, Y, X)

        data_interpolated[i] = torch.from_numpy(vals)

    return data_interpolated
