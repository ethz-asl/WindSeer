import models
import torch

'''
Uncomment the @profile if it is run using mprof: mprof run --interval 0.001 test_model.py
'''

profile_ram = True
try:
    import memory_profiler
    from memory_profiler import memory_usage
except:
    print("Running script without checking RAM, install the memory_profiler to also log the RAM usage:")
    print("pip3 install -U memory_profiler")
    profile_ram = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#@profile
def test_ModelEDNN2D(skipping, batchsize):
    loss_fn = torch.nn.MSELoss()

    input = torch.randn(batchsize, 3, 128, 64).to(device)
    labels = torch.randn(batchsize, 2, 128, 64).to(device)

    try:
        net = models.ModelEDNN2D(3, 'bilinear', True, skipping).to(device)
        net.init_params()

    except:
        print('\tTest (skip: {}, batchsize: {}): init failed'.format(skipping, batchsize))
        return False

    try:
        output = net(input)

    except:
        print('\tTest (skip: {}, batchsize: {}): forwad failed'.format(skipping, batchsize))
        return False

    try:
        loss = loss_fn(output, labels)
        loss.backward()

    except:
        print('\tTest (skip: {}, batchsize: {}): backward failed'.format(skipping, batchsize))
        return False

    print('\tTest (skip: {}, batchsize: {}): passed'.format(skipping, batchsize))

    return True

#@profile
def test_ModelEDNN3D(skipping, batchsize, high_resolution):
    loss_fn = torch.nn.MSELoss()

    if (high_resolution):
        input = torch.randn(batchsize, 3, 128, 128, 128).to(device)
        labels = torch.randn(batchsize, 2, 128, 128, 128).to(device)
    else:
        input = torch.randn(batchsize, 3, 64, 64, 64).to(device)
        labels = torch.randn(batchsize, 2, 64, 64, 64).to(device)

    try:
        net = models.ModelEDNN3D(3, 'trilinear', True, skipping, high_resolution).to(device)
        net.init_params()

    except:
        print('\tTest (skip: {}, batchsize: {}, hr: {}): init failed'.format(skipping, batchsize, high_resolution))
        return False

    try:
        output = net(input)

    except:
        print('\tTest (skip: {}, batchsize: {}, hr: {}): forward failed'.format(skipping, batchsize, high_resolution))
        return False

    try:
        loss = loss_fn(output, labels)
        loss.backward()

    except:
        print('\tTest (skip: {}, batchsize: {}, hr: {}): backward failed'.format(skipping, batchsize, high_resolution))
        return False

    print('\tTest (skip: {}, batchsize: {}, hr: {}): passed'.format(skipping, batchsize, high_resolution))
    return True

if __name__ == "__main__":
    if profile_ram:
        print("--------------------------------------------------------")
        print("ModelEDNN2D tests")
        ram = memory_usage((test_ModelEDNN2D, (False,1)), interval=0.001)
        print('\t\tmax ram: {} MB'.format(max(ram)))
        ram = memory_usage((test_ModelEDNN2D, (True,1)), interval=0.001)
        print('\t\tmax ram: {} MB'.format(max(ram)))
        ram = memory_usage((test_ModelEDNN2D, (False,32)), interval=0.001)
        print('\t\tmax ram: {} MB'.format(max(ram)))
        ram = memory_usage((test_ModelEDNN2D, (True,32)), interval=0.001)
        print('\t\tmax ram: {} MB'.format(max(ram)))

        print("--------------------------------------------------------")
        print("ModelEDNN3D 64*64*64 tests")
        ram = memory_usage((test_ModelEDNN3D, (False,1, False)), interval=0.001)
        print('\t\tmax ram: {} MB'.format(max(ram)))
        ram = memory_usage((test_ModelEDNN3D, (True,1, False)), interval=0.001)
        print('\t\tmax ram: {} MB'.format(max(ram)))
        ram = memory_usage((test_ModelEDNN3D, (False,32, False)), interval=0.001)
        print('\t\tmax ram: {} MB'.format(max(ram)))

        print("--------------------------------------------------------")
        print("ModelEDNN3D 128*128*128 tests")
        ram = memory_usage((test_ModelEDNN3D, (False,1, True)), interval=0.001)
        print('\t\tmax ram: {} MB'.format(max(ram)))
        ram = memory_usage((test_ModelEDNN3D, (True,1, True)), interval=0.001)
        print('\t\tmax ram: {} MB'.format(max(ram)))

    else:
        print("--------------------------------------------------------")
        print("ModelEDNN2D tests")
        test_ModelEDNN2D(False, 1)
        test_ModelEDNN2D(True, 1)
        test_ModelEDNN2D(False, 32)
        test_ModelEDNN2D(True, 32)

        print("--------------------------------------------------------")
        print("ModelEDNN3D 64*64*64 tests")
        test_ModelEDNN3D(False, 1, False)
        test_ModelEDNN3D(True, 1, False)
        test_ModelEDNN3D(False, 32, False)
#         test_ModelEDNN3D(True, 32, False) # uses >20 GB Ram

        print("--------------------------------------------------------")
        print("ModelEDNN3D 128*128*128 tests")
        test_ModelEDNN3D(False, 1, True)
        test_ModelEDNN3D(True, 1, True)
#         test_ModelEDNN3D(False, 1, True) # uses >20 GB Ram
#         test_ModelEDNN3D(True, 1, True) # uses >20 GB Ram
