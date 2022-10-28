#!/usr/bin/env python


def scale_tensor(tensor, tensor_channels, scale, params):
    '''
    Scale a tensor according to the scaling parameter in the params dict.
    Assumes the tensor has either 5 [batch, channels, x, y, z]
    or 4 [channels, x, y, z] dimensions.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor
    tensor_channels : list of str
        Channel names of the input tensor
    scale : float or None
        Scale of the sample, if None the scale is set to 1.0
    params : WindseerParams
        Parameter class

    Returns
    -------
    out_tensor : torch.Tensor
        Rescaled tensor
    '''
    if len(tensor.shape) == 5:
        out_tensor = tensor.clone()

    elif len(tensor.shape) == 4:
        out_tensor = tensor.clone().unsqueeze(0)

    else:
        raise ValueError('A 4D or 5D input tensor is expected')

    if scale is None:
        scale = 1.0

    # make sure the channels to predict exist and are properly ordered
    default_channels = ['terrain', 'ux', 'uy', 'uz', 'turb', 'p', 'epsilon', 'nut', 'mask']
    for channel in tensor_channels:
        if channel not in default_channels:
            raise ValueError(
                'Incorrect label_channel detected: \'{}\', '
                'correct channels are {}'.format(channel, default_channels)
                )
    tensor_channels = [x for x in default_channels if x in tensor_channels]

    # rescale the labels and predictions
    for i, channel in enumerate(tensor_channels):
        if channel == 'terrain':
            out_tensor[:, i] /= params.data[channel + '_scaling']
        elif channel.startswith('u') or channel == 'nut':
            out_tensor[:, i] /= scale * params.data[channel + '_scaling']
        elif channel == 'p' or channel == 'turb':
            out_tensor[:, i] /= scale * scale * params.data[channel + '_scaling']
        elif channel == 'epsilon':
            out_tensor[:,
                       i] /= scale * scale * scale * params.data[channel + '_scaling']

    if len(tensor.shape) == 4:
        return out_tensor[0]
    else:
        return out_tensor


def rescale_tensor(tensor, tensor_channels, scale, params):
    '''
    Rescale a tensor according to the scaling parameter in the params dict.
    Assumes the tensor has either 5 [batch, channels, x, y, z]
    or 4 [channels, x, y, z] dimensions.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor
    tensor_channels : list of str
        Channel names of the input tensor
    scale : float or None
        Scale of the sample, if None the scale is set to 1.0
    params : WindseerParams
        Parameter class

    Returns
    -------
    out_tensor : torch.Tensor
        Rescaled tensor
    '''
    if len(tensor.shape) == 5:
        out_tensor = tensor.clone()

    elif len(tensor.shape) == 4:
        out_tensor = tensor.clone().unsqueeze(0)

    else:
        raise ValueError('A 4D or 5D input tensor is expected')

    if scale is None:
        scale = 1.0

    # make sure the channels to predict exist and are properly ordered
    default_channels = ['terrain', 'ux', 'uy', 'uz', 'turb', 'p', 'epsilon', 'nut', 'mask']
    for channel in tensor_channels:
        if channel not in default_channels:
            raise ValueError(
                'Incorrect label_channel detected: \'{}\', '
                'correct channels are {}'.format(channel, default_channels)
                )
    tensor_channels = [x for x in default_channels if x in tensor_channels]

    # rescale the labels and predictions
    for i, channel in enumerate(tensor_channels):
        if channel == 'terrain':
            out_tensor[:, i] *= params.data[channel + '_scaling']
        elif channel.startswith('u') or channel == 'nut':
            out_tensor[:, i] *= scale * params.data[channel + '_scaling']
        elif channel == 'p' or channel == 'turb':
            out_tensor[:, i] *= scale * scale * params.data[channel + '_scaling']
        elif channel == 'epsilon':
            out_tensor[:,
                       i] *= scale * scale * scale * params.data[channel + '_scaling']

    if len(tensor.shape) == 4:
        return out_tensor[0]
    else:
        return out_tensor
