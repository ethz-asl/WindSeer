import tarfile
import torch

def get_grid_size(dataset_filename):
    tar = tarfile.open(dataset_filename, 'r')
    memberslist = tar.getmembers()
    file = tar.extractfile(memberslist[0])

    ds = torch.load(file)[1]
    return torch.tensor(ds).tolist()
