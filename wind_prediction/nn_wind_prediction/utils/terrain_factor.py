def compute_terrain_factor(samples, terrain):
    '''
    This function computes the ratio of the amount of fluid data in a sample or a batch of samples vs the total amount
    of data in the samples. This factor is used to correct the loss, where samples with more terrain have a loss that is
    underestimated (data is 0 in the terrain). Thus samples with more terrain have a higher terrain factor.

    Input params:
        net_output: 5D tensor [samples, input channels, Z, Y, X]
        terrain: 5D tensor [samples, terrain, Z, Y, X]. Must be the same size and for the same samples as input.

    Output:
        terrain_factor: ratio of the amount of data in the samples terrain to the total amount of data.
    '''
    if (samples.shape[0]!=terrain.shape[0]):
        raise ValueError('Terrain factor: samples and terrain must concern the same batch.')

    if (samples.shape[-3:]!=terrain.shape[-3:]):
        raise ValueError('Terrain factor: samples and terrain must have same domain shape and resolution.')

    elements_in_fluid = terrain.sign_().sum().item()*samples.shape[1]
    total_elements = samples.numel()
    terrain_factor = total_elements/elements_in_fluid

    return terrain_factor