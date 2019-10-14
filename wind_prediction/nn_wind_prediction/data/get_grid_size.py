import h5py
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
    h5_file = h5py.File(dataset_filename, 'r', swmr=True)
    sample = h5_file[list(h5_file.keys())[0]]

    # extract the spacing of the grid which is contained in the second component of the dataset
    grid_size = torch.from_numpy(sample['ds'][...]).tolist()

    return grid_size
