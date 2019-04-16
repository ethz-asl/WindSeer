import tarfile
import torch

def get_grid_size(dataset_filename):
    '''
    This function gets the spacing of the grid [in meters] of the dataset provided. Assumes that the grid size is
    provided in the second component of the data, and is equal between samples.

    Input params:
        dataset_filename: the name of the dataset to get the grid spacing from.

    Output:
        grid_size: list containing the grid spacing in directions X, Y and Z of the dataset. [m]
    '''

    # read the dataset tar file
    tar = tarfile.open(dataset_filename, 'r')
    memberslist = tar.getmembers()
    file = tar.extractfile(memberslist[0])

    # extract the spacing of the grid which is contained in the second component of the dataset
    grid_size = torch.load(file)[1]

    # change type to list of floats
    grid_size = torch.as_tensor(grid_size, dtype=torch.float).tolist()
    return grid_size
