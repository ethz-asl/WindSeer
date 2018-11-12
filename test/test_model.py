import nn_wind_prediction.models as models
import torch
import argparse
from termcolor import colored

'''
Uncomment the @profile if it is run using mprof: mprof run --interval 0.001 test_model.py
'''
memory_profiler_available = True
try:
    import memory_profiler
    from memory_profiler import memory_usage
except:
    print("Running script without checking RAM, install the memory_profiler to also log the RAM usage:")
    print("pip3 install -U memory_profiler")
    memory_profiler_available = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#@profile
def test_ModelEDNN2D(skipping, batchsize, error_counter = 0, test_counter = 0):
    test_counter += 1
    loss_fn = torch.nn.MSELoss()

    input = torch.randn(batchsize, 3, 64, 128).to(device)
    labels = torch.randn(batchsize, 3, 64, 128).to(device)

    try:
        net = models.ModelEDNN2D(3, 'bilinear', True, skipping, True).to(device)
        net.init_params()

    except:
        print('\tTest #{} (skip: {}, batchsize: {}):'.format(test_counter, skipping, batchsize) + colored(' init failed', 'red'))
        error_counter += 1
        return error_counter, test_counter

    try:
        output = net(input)

    except:
        print('\tTest #{} (skip: {}, batchsize: {}):'.format(test_counter, skipping, batchsize) + colored(' forward failed', 'red'))
        error_counter += 1
        return error_counter, test_counter

    try:
        loss = loss_fn(output, labels)
        loss.backward()

    except:
        print('\tTest #{} (skip: {}, batchsize: {}):'.format(test_counter, skipping, batchsize) + colored(' backward failed', 'red'))
        error_counter += 1
        return error_counter, test_counter

    print('\tTest #{} (skip: {}, batchsize: {}):'.format(test_counter, skipping, batchsize) + colored(' passed', 'green'))

    return error_counter, test_counter

#@profile
def test_ModelEDNN3D(batch_size, n_input_layers, n_output_layers, n_x, n_y, n_z, n_downsample_layers,
                     interpolation_mode, align_corners, skipping, use_terrain_mask, pooling_method,
                     use_mapping_layer, use_fc_layers, fc_scaling, potential_flow, error_counter = 0, test_counter = 0):
    test_counter += 1
    loss_fn = torch.nn.MSELoss()

    input = torch.randn(batch_size, n_input_layers, n_z, n_y, n_x).to(device)
    labels = torch.randn(batch_size, n_output_layers, n_z, n_y, n_x).to(device)
    print('\tTest #{}'.format(test_counter))
    print('\t\tConfig:')
    print('\t\t\tn_input_layers: {}, n_output_layers: {}'.format(n_input_layers, n_output_layers))
    print('\t\t\tn_x: {}, n_y: {}, n_z: {}, n_downsample_layers: {}'.format(n_x, n_y, n_z, n_downsample_layers))
    print('\t\t\tinterpolation_mode: {}, align_corners: {}, skipping: {}'.format(interpolation_mode, align_corners, skipping))
    print('\t\t\tuse_terrain_mask: {}, pooling_method: {}, batchsize: {}'.format(use_terrain_mask, pooling_method, batch_size))
    print('\t\t\tuse_mapping_layer: {}, use_fc_layers: {}, fc_scaling: {}'.format(use_mapping_layer, use_fc_layers, fc_scaling))
    print('\t\t\tpotential_flow: {}'.format(potential_flow))
    print('\t\tResult:')

    try:
        net = models.ModelEDNN3D(n_input_layers, n_output_layers, n_x, n_y, n_z, n_downsample_layers, interpolation_mode,
                                 align_corners, skipping, use_terrain_mask, pooling_method, use_mapping_layer, use_fc_layers,
                                 fc_scaling, potential_flow).to(device)
        net.init_params()

    except:
        print(colored('\t\t\tinit failed', 'red'))
        error_counter += 1
        return error_counter, test_counter
    output = net(input)

    try:
        output = net(input)

    except:
        print(colored('\t\t\tforward failed', 'red'))
        error_counter += 1
        return error_counter, test_counter

    try:
        loss = loss_fn(output, labels)
        loss.backward()

    except:
        print(colored('\t\t\tbackward failed', 'red'))
        error_counter += 1
        return error_counter, test_counter

    print(colored('\t\t\tpassed', 'green'))
    return error_counter, test_counter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to test the models')
    parser.add_argument('-m', dest='mode', type = int, required=True, help='mode in which the script should be run, 0: ram profiling, 1: integration test')
    args = parser.parse_args()

    error_counter = 0
    test_counter = 0

    if args.mode == 0:
        if memory_profiler_available:
            print("--------------------------------------------------------")
            print("ModelEDNN2D tests")
            ram = memory_usage((test_ModelEDNN2D, (False,1, error_counter, test_counter)), interval=0.001)
            test_counter += 1
            print('\t\tmax ram: {} MB'.format(max(ram)))
            ram = memory_usage((test_ModelEDNN2D, (True,1, error_counter, test_counter)), interval=0.001)
            test_counter += 1
            print('\t\tmax ram: {} MB'.format(max(ram)))
            ram = memory_usage((test_ModelEDNN2D, (False,32, error_counter, test_counter)), interval=0.001)
            test_counter += 1
            print('\t\tmax ram: {} MB'.format(max(ram)))
            ram = memory_usage((test_ModelEDNN2D, (True,32, error_counter, test_counter)), interval=0.001)
            test_counter += 1
            print('\t\tmax ram: {} MB'.format(max(ram)))

            print("--------------------------------------------------------")
            print("ModelEDNN3D 64*128*128 tests")
            ram = memory_usage((test_ModelEDNN3D, (1, 4, 4, 64, 64, 64, 4, 'nearest', False, True, True, 'maxpool', True, True, 2, False, error_counter, test_counter)), interval=0.001)
            test_counter += 1
            print('\t\tmax ram: {} MB'.format(max(ram)))
            ram = memory_usage((test_ModelEDNN3D, (16, 4, 4, 64, 64, 64, 4, 'nearest', False, True, True, 'averagepool', True, True, 2, False, error_counter, test_counter)), interval=0.001)
            test_counter += 1
            print('\t\tmax ram: {} MB'.format(max(ram)))
            ram = memory_usage((test_ModelEDNN3D, (1, 4, 4, 64, 64, 64, 4, 'nearest', False, True, True, 'striding', True, True, 2, False, error_counter, test_counter)), interval=0.001)
            test_counter += 1
            print('\t\tmax ram: {} MB'.format(max(ram)))
            ram = memory_usage((test_ModelEDNN3D, (8, 4, 4, 128, 128, 64, 4, 'nearest', False, True, True, 'maxpool', True, True, 2, False, error_counter, test_counter)), interval=0.001)
            test_counter += 1
            print('\t\tmax ram: {} MB'.format(max(ram)))
            ram = memory_usage((test_ModelEDNN3D, (8, 4, 4, 128, 128, 64, 4, 'nearest', False, True, True, 'maxpool', True, True, 2, True, error_counter, test_counter)), interval=0.001)
            test_counter += 1
            print('\t\tmax ram: {} MB'.format(max(ram)))

        else:
            print("--------------------------------------------------------")
            print("ModelEDNN2D 64*128 tests")
            error_counter, test_counter = test_ModelEDNN2D(False, 1, error_counter, test_counter)
            error_counter, test_counter = test_ModelEDNN2D(True, 1, error_counter, test_counter)
            error_counter, test_counter = test_ModelEDNN2D(False, 32, error_counter, test_counter)
            error_counter, test_counter = test_ModelEDNN2D(True, 32, error_counter, test_counter)

            print("--------------------------------------------------------")
            print("ModelEDNN3D tests")
            test_ModelEDNN3D(1, 4, 4, 64, 64, 64, 4, 'nearest', False, True, True, 'maxpool', True, True, 2, False, error_counter, test_counter)
            test_ModelEDNN3D(64, 4, 4, 64, 64, 64, 4, 'nearest', False, True, True, 'maxpool', True, True, 2, False, error_counter, test_counter)
            test_ModelEDNN3D(1, 4, 4, 128, 128, 64, 4, 'nearest', False, True, True, 'maxpool', True, True, 2, False, error_counter, test_counter)
            test_ModelEDNN3D(8, 4, 4, 128, 128, 64, 4, 'nearest', False, True, True, 'maxpool', True, True, 2, False, error_counter, test_counter)
            test_ModelEDNN3D(8, 4, 4, 128, 128, 64, 4, 'nearest', False, True, True, 'maxpool', True, True, 2, True, error_counter, test_counter)

    elif args.mode == 1:
        print("--------------------------------------------------------")
        print("ModelEDNN2D 64*128 tests")
        error_counter, test_counter = test_ModelEDNN2D(False, 1, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN2D(True, 1, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN2D(False, 32, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN2D(True, 32, error_counter, test_counter)

        print("--------------------------------------------------------")
        print("ModelEDNN3D tests")
        error_counter, test_counter = test_ModelEDNN3D(1, 4, 4, 8, 8, 8, 0, 'nearest', False, True, True, 'maxpool', True, True, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(1, 4, 4, 8, 8, 8, 0, 'nearest', False, True, True, 'maxpool', True, True, 2, True, error_counter, test_counter)
#         error_counter, test_counter = test_ModelEDNN3D(1, 4, 4, 8, 8, 8, 0, 'nearest', False, True, True, 'maxpool', False, False, 2, error_counter, test_counter) #Backwards is expected to fail here because
        error_counter, test_counter = test_ModelEDNN3D(1, 4, 4, 8, 8, 8, 0, 'nearest', False, True, True, 'maxpool', True, False, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(1, 4, 4, 8, 8, 8, 0, 'nearest', False, True, True, 'maxpool', False, True, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(1, 4, 4, 8, 8, 8, 0, 'nearest', False, True, True, 'maxpool', False, True, 1, False, error_counter, test_counter)

        error_counter, test_counter = test_ModelEDNN3D(1, 4, 4, 8, 8, 8, 2, 'nearest', False, True, True, 'maxpool', True, True, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(1, 4, 4, 8, 8, 8, 2, 'nearest', False, False, True, 'maxpool', True, True, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(1, 4, 4, 8, 8, 8, 2, 'nearest', False, True, False, 'maxpool', True, True, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(1, 4, 4, 8, 8, 8, 2, 'nearest', False, False, False, 'maxpool', True, True, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(64, 4, 4, 8, 8, 8, 2, 'nearest', False, True, True, 'maxpool', True, True, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(64, 4, 4, 8, 8, 8, 2, 'nearest', False, False, True, 'maxpool', True, True, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(64, 4, 4, 8, 8, 8, 2, 'nearest', False, True, False, 'maxpool', True, True, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(64, 4, 4, 8, 8, 8, 2, 'nearest', False, False, False, 'maxpool', True, True, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(1, 4, 4, 8, 8, 8, 2, 'nearest', False, True, True, 'averagepool', True, True, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(1, 4, 4, 8, 8, 8, 2, 'nearest', False, True, True, 'striding', True, True, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(1, 4, 4, 8, 8, 8, 2, 'trilinear', False, True, True, 'striding', True, True, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(1, 4, 4, 8, 8, 8, 2, 'trilinear', True, True, True, 'striding', True, True, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(1, 4, 4, 8, 8, 8, 2, 'trilinear', True, False, False, 'striding', True, True, 2, False, error_counter, test_counter)

        error_counter, test_counter = test_ModelEDNN3D(1, 4, 4, 8, 8, 8, 2, 'nearest', False, True, True, 'maxpool', True, True, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(1, 1, 1, 8, 8, 8, 2, 'nearest', False, True, True, 'maxpool', True, True, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(1, 4, 1, 8, 8, 8, 2, 'nearest', False, True, True, 'maxpool', True, True, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(1, 1, 4, 8, 8, 8, 2, 'nearest', False, True, True, 'maxpool', True, True, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(16, 10, 10, 8, 8, 8, 2, 'nearest', False, True, True, 'maxpool', True, True, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(16, 10, 10, 8, 8, 8, 2, 'nearest', False, True, True, 'maxpool', True, True, 1, False, error_counter, test_counter)

        error_counter, test_counter = test_ModelEDNN3D(1, 4, 4, 64, 64, 32, 5, 'nearest', False, False, False, 'maxpool', True, True, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(1, 4, 4, 64, 32, 64, 5, 'nearest', False, False, False, 'maxpool', True, True, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(1, 4, 4, 32, 64, 64, 5, 'nearest', False, False, False, 'maxpool', True, True, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(1, 4, 4, 32, 64, 64, 5, 'nearest', False, False, False, 'maxpool', True, True, 2, True, error_counter, test_counter)

        error_counter, test_counter = test_ModelEDNN3D(1, 4, 4, 64, 64, 32, 5, 'nearest', False, True, True, 'maxpool', True, True, 2, False, error_counter, test_counter)
        error_counter, test_counter = test_ModelEDNN3D(1, 4, 4, 64, 64, 64, 4, 'nearest', False, True, True, 'maxpool', True, True, 2, True, error_counter, test_counter)

        if (error_counter == 0):
            print(colored('{} out of {} test failed'.format(error_counter, test_counter), 'green'))
        else:
            print(colored('{} out of {} test failed'.format(error_counter, test_counter), 'red'))

    else:
        raise ValueError('Unknown mode input, only 0, 1 currently supported: {}'.format(args.mode) )
