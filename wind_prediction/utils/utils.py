def edge_interpolation(input):
    '''
    Interpolate the four vertical edges over the full domain
    '''
    edge1 = input[:,:,:,0,0]
    edge2 = input[:,:,:,0,-1]
    edge3 = input[:,:,:,-1,0]
    edge4 = input[:,:,:,-1,-1]

    output = torch.zeros(input.size())
    output[:,:,:,0,0] = edge1
    output[:,:,:,0,-1] = edge2
    output[:,:,:,-1,0] = edge3
    output[:,:,:,-1,-1] = edge4

    for i in range(64):
        output[:,:,:,0,i] = (64.0 - float(i)) / 64.0 * output[:,:,:,0,0] + float(i) / 64.0 * output[:,:,:,0,-1]
        output[:,:,:,-1,i] = (64.0 - float(i)) / 64.0 * output[:,:,:,-1,0] + float(i) / 64.0 * output[:,:,:,-1,-1]

    for i in range(64):
       output[:,:,:,i,:] = (64.0 - float(i)) / 64.0 * output[:,:,:,0,:] + float(i) / 64.0 * output[:,:,:,-1,:]

    return output
