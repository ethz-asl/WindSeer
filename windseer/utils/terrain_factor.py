def compute_terrain_factor(samples, terrain):
    '''
    This function computes the ratio of the amount of fluid data in a sample or a batch of samples vs the total amount
    of data in the samples. This factor is used to correct the loss, where samples with more terrain have a loss that is
    underestimated (data is 0 in the terrain). Thus samples with more terrain must have a higher terrain factor.

    Parameters
    ----------
    samples : torch.Tensor
        5D tensor [batch, input channels, Z, Y, X]
    terrain : torch.Tensor
        5D tensor [batch, terrain, Z, Y, X]. Must be for the same samples as input.

    Returns
    -------
    terrain_factors : torch.Tensor
        The ratio of the amount of data in the samples terrain to the total amount of data for each sample.
    '''
    if (len(samples.shape) != 5) or (len(terrain.shape) != 5):
        raise ValueError(
            'compute_terrain_factor: only defined for 5D data. Unsqueeze single samples!'
            )

    if (samples.shape[0] != terrain.shape[0]):
        raise ValueError(
            'compute_terrain_factor: batchsize of samples({}) and terrain({}) are different.'
            .format(samples.shape[0], terrain.shape[0])
            )

    if (samples.shape[-3:] != terrain.shape[-3:]):
        raise ValueError(
            'compute_terrain_factor: samples and terrain must have same domain shape and resolution.'
            )

    # get amount of data that is not in the terrain for each sample
    fluid_elements_in_samples = terrain.sign_().sum(-1).sum(-1).sum(-1
                                                                    ) * samples.shape[1]

    # get total amount of data of 1 sample
    total_elements_in_samples = samples[0, :].numel()

    # compute the terrain factor for each sample
    terrain_factors = total_elements_in_samples / fluid_elements_in_samples

    # return squeezed version of the terrain factors vector
    return terrain_factors.squeeze()
