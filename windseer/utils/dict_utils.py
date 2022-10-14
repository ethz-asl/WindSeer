from collections import Mapping
import copy
import torch


def dict_update(d, u):
    """
    Update the fields for nested dictionaries.

    Parameters
    ----------
    d : dict
        Dictionary to be updated
    u : dict
        Dictionary with the updated values

    Returns
    -------
    out : dict
        Updated dictionary
    """
    out = copy.deepcopy(d)
    for k, v in u.items():
        if isinstance(v, Mapping):
            if type(out.get(k, {})) is dict:
                out[k] = dict_update(out.get(k, {}), v)
            else:
                out[k] = dict_update({}, v)
        else:
            out[k] = v
    return out


def data_to_device(data, device):
    """
    Move all tensors in the dictionary to the requested device.

    Parameters
    ----------
    data : dict
        Dictionary containing torch.Tensors
    device : torch.Device
        Device where the tensors should be moved

    Returns
    -------
    data : dict
        Output dictionary
    """
    for key in data.keys():
        if type(data[key]) is torch.Tensor:
            data[key] = data[key].to(device)
        elif type(data[key]) is dict:
            data[key] = data_to_device(data[key], device)
    return data


def tensors_to_dtype(data, dtype):
    """
    Cast all tensors in the dictionary to the requested dtype.

    Parameters
    ----------
    data : dict
        Dictionary containing torch.Tensors
    dtype : torch.dtype
        Type of the tensor values

    Returns
    -------
    data : dict
        Output dictionary
    """
    for key in data.keys():
        if type(data[key]) is torch.Tensor:
            data[key] = data[key].to(dtype)
        elif type(data[key]) is dict:
            data[key] = tensors_to_dtype(data[key], dtype)
    return data


def data_unsqueeze(data, dim):
    """
    Unsqueeze all tensors in the dictionary in the requested dimension.

    Parameters
    ----------
    data : dict
        Dictionary containing torch.Tensors
    dim : int
        Axis index where to add the extra dimension

    Returns
    -------
    data : dict
        Output dictionary
    """
    for key in data.keys():
        if type(data[key]) is torch.Tensor:
            data[key] = data[key].unsqueeze(dim)
        elif type(data[key]) is dict:
            data[key] = data_unsqueeze(data[key], dim)
    return data
