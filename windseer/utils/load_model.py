#!/usr/bin/env python

import windseer.utils as utils
import windseer.nn.models as models

import os
import torch


def load_model(directory, version, dataset, device, eval=True):
    '''
    Load a neural network together with the corresponding weights.

    Parameters
    ----------
    directory : str
        Model directory
    version : str
        Version of the model to load (filename without the .model ending)
    dataset : str
        Filename of the prediction dataset
    device : torch.Device
        Device where the model should be stored
    eval : bool, default : True
        Put the model in eval mode

    Returns
    -------
    net : torch.Module
        Neural network with the loaded weights
    params : WindseerParams
        Model parameter dictionary
    '''
    params = utils.WindseerParams(os.path.join(directory, 'params.yaml'))

    # get grid size of test dataset if potential flow is used
    if params.model_kwargs()['potential_flow']:
        grid_size = nn_data.get_grid_size(dataset)
        params.model_kwargs()['grid_size'] = grid_size

    NetworkType = getattr(models, params.model['model_type'])
    net = NetworkType(**params.model_kwargs())

    state_dict = torch.load(
        os.path.join(directory, version + '.model'),
        map_location=lambda storage, loc: storage
        )

    # fix keys for legacy models
    keys = list(state_dict.keys())
    for key in keys:
        legacy_string = '_' + params.model['model_type'] + '_'
        if legacy_string in key:
            state_dict[key.replace(legacy_string, '')] = state_dict.pop(key)

    # load params
    net.load_state_dict(state_dict)
    net.to(device)

    if eval:
        net.eval()

    return net, params
