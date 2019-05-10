def remove_terrain_data(input_tensor, terrain):
    '''
    This function returns a reduced version of input where data that is in the terrain is removed. Used to compute loss
    of batches while excluding data in the terrain.

    Input params:
        input: 5D tensor [samples, input channels, Z, Y, X]
        terrain: 5D tensor [samples, terrain, Z, Y, X]. Must be the same size and for the same samples as input.

    Output:
        fluid_cells: 1D tensor containing input data of all cells located in the fluid for all samples.
    '''
    if len(input_tensor.shape) != 5 or len(terrain.shape) != 5:
        raise ValueError('Inputs must be 5D. Unsqueeze single samples!')

    # create mask
    is_fluid = terrain.sign_().expand(-1, input_tensor.shape[1], -1, -1, -1)

    # keep only data that is not in the terrain
    fluid_cells = input_tensor[is_fluid.byte()]
    return fluid_cells
