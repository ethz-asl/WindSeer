import nn_wind_prediction.nn as nn_custom
import time
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\t\t\t\t\t\tTESTING LOSS FUNCTIONS')
print('Using device: ', device)
print('------------------------------------------------------------------')

# test combined loss
input = torch.rand(20,4,64,64,64, requires_grad=False)
label = torch.rand(20,4,64,64,64, requires_grad=False)
output = torch.rand(20,4,64,64,64, requires_grad=True)
input[:,0,:10,:,:] = 0.0 # generate some terrain

label, output, input = label.to(device), output.to(device), input.to(device)

#------------------------------------------ ADD CONFIGS TO TEST HERE ---------------------------------------------------
configs = []

# single loss testing : LPLoss  (only up to order 9, but can go beyond)
for p in range(1,10):
    name = 'L{}Loss'.format(p)
    configs.append({'loss_components': [name],
                    'learn_scaling': False,
                    name+'_kwargs':{'exclude_terrain': True}})
    configs.append({'loss_components': [name],
                    'learn_scaling': False,
                    name + '_kwargs': {'exclude_terrain': False}})

# single loss testing : Scaled loss
configs.append({'loss_components': ['ScaledLoss'],
                'learn_scaling': False,
                'ScaledLoss_kwargs':{'exclude_terrain': True, 'no_scaling': True}})

configs.append({'loss_components': ['ScaledLoss'],
                'learn_scaling': False,
                'ScaledLoss_kwargs':{'exclude_terrain': True, 'no_scaling': False}})

configs.append({'loss_components': ['ScaledLoss'],
                'learn_scaling': False,
                'ScaledLoss_kwargs':{'exclude_terrain': False, 'no_scaling': True}})

configs.append({'loss_components': ['ScaledLoss'],
                'learn_scaling': False,
                'ScaledLoss_kwargs':{'exclude_terrain': False, 'no_scaling': True}})

# single loss testing : DFL
configs.append({'loss_components': ['DivergenceFreeLoss'],
                'learn_scaling': False,
                'DivergenceFreeLoss_kwargs':{'exclude_terrain': True, 'loss_type': 'L1'}})

configs.append({'loss_components': ['DivergenceFreeLoss'],
                'learn_scaling': False,
                'DivergenceFreeLoss_kwargs':{'exclude_terrain': False, 'loss_type': 'L1'}})

configs.append({'loss_components': ['DivergenceFreeLoss'],
                'learn_scaling': False,
                'DivergenceFreeLoss_kwargs':{'exclude_terrain': True, 'loss_type': 'MSE'}})

configs.append({'loss_components': ['DivergenceFreeLoss'],
                'learn_scaling': False,
                'DivergenceFreeLoss_kwargs':{'exclude_terrain': False, 'loss_type': 'MSE'}})

# single loss testing : VGL
configs.append({'loss_components': ['VelocityGradientLoss'],
                'learn_scaling': False,
                'VelocityGradientLoss_kwargs':{'exclude_terrain': True, 'loss_type': 'L1'}})

configs.append({'loss_components': ['VelocityGradientLoss'],
                'learn_scaling': False,
                'VelocityGradientLoss_kwargs':{'exclude_terrain': False, 'loss_type': 'L1'}})

configs.append({'loss_components': ['VelocityGradientLoss'],
                'learn_scaling': False,
                'VelocityGradientLoss_kwargs':{'exclude_terrain': True, 'loss_type': 'MSE'}})

configs.append({'loss_components': ['VelocityGradientLoss'],
                'learn_scaling': False,
                'VelocityGradientLoss_kwargs':{'exclude_terrain': False, 'loss_type': 'MSE'}})

# single loss testing : GLL
configs.append({'loss_components': ['GaussianLogLikelihoodLoss'],
                'learn_scaling': False,
                'GaussianLogLikelihoodLoss_kwargs':{'exclude_terrain': True, 'uncertainty_loss_eps': 1e-8}})

configs.append({'loss_components': ['GaussianLogLikelihoodLoss'],
                'learn_scaling': False,
                'GaussianLogLikelihoodLoss_kwargs':{'exclude_terrain': True, 'uncertainty_loss_eps': 1e-8}})

# multiple combined loss testing
configs.append({'loss_components': ['L2Loss', 'L1Loss'],
                'learn_scaling': False,
                'L2Loss_kwargs':{'exclude_terrain': True, 'loss_factor_init': 1.0},
                'L1Loss_kwargs':{'exclude_terrain': True, 'loss_factor_init': 1.0}})

configs.append({'loss_components': ['DivergenceFreeLoss', 'L1Loss'],
                'learn_scaling': True,
                'DivergenceFreeLoss_kwargs':{'exclude_terrain': True, 'loss_type': 'L1', 'loss_factor_init': 1.0},
                'L1Loss_kwargs':{'exclude_terrain': True, 'loss_factor_init': 1.0}})

configs.append({'loss_components': ['VelocityGradientLoss', 'L2Loss'],
                'learn_scaling': True,
                'VelocityGradientLoss_kwargs':{'exclude_terrain': True, 'loss_type': 'L1', 'loss_factor_init': 1.0},
                'L2Loss_kwargs':{'exclude_terrain': True, 'loss_factor_init': 1.0}})

configs.append({'loss_components': ['VelocityGradientLoss','DivergenceFreeLoss', 'L2Loss', 'L1Loss'],
                'learn_scaling': True,
                'VelocityGradientLoss_kwargs':{'exclude_terrain': True, 'loss_type': 'L1', 'loss_factor_init': 1.0},
                'L1Loss_kwargs':{'exclude_terrain': True, 'loss_factor_init': 1.0},
                'L2Loss_kwargs': {'exclude_terrain': True, 'loss_factor_init': 1.0},
                'DivergenceFreeLoss_kwargs':{'exclude_terrain': True, 'loss_type': 'L1', 'loss_factor_init': 1.0},})
#-----------------------------------------------------------------------------------------------------------------------

# custom loss testing for loop
for k, config in enumerate(configs):
    k+= 1
    print('\t', 'Test {}: CombinedLoss w/ component(s) {} \n'.format(k,config['loss_components']))
    loss_fn = nn_custom.CombinedLoss(**config)
    param_list = []
    if loss_fn.learn_scaling:
        print('Learning the scaling!')
        param_list.append({'params': loss_fn.parameters()})
        # for i, param in enumerate(loss_fn.parameters()):
        #     print('PARAMS: ', param)
        optimizer = torch.optim.Adam(params=param_list, lr=1e-1)
        optimizer.zero_grad()

    start_time = time.time()
    if 'GaussianLogLikelihoodLoss' in config['loss_components']:
        start_time = time.time()
        loss = loss_fn(torch.cat((output, output), 1), label, input)
    else:
        start_time = time.time()
        loss = loss_fn(output, label, input)
    print('[{}] '.format(k),'Forward took', (time.time() - start_time), 'seconds')
    print('[{}] '.format(k), 'Computed Loss:', loss.item())

    start_time = time.time()
    loss.backward()
    print('[{}] '.format(k),'Backward took', (time.time() - start_time), 'seconds')
    if loss_fn.learn_scaling:
        optimizer.step()
        # for i, param in enumerate(loss_fn.parameters()):
            # print('GRADIENTS: ', param.grad)
            # print('PARAMS: ', param)

        if 'GaussianLogLikelihoodLoss' in config['loss_components']:
            start_time = time.time()
            loss = loss_fn(torch.cat((output,output),1), label, input)
        else:
            start_time = time.time()
            loss = loss_fn(output, label, input)
        print('[{}] '.format(k), 'Forward pass #2 took', (time.time() - start_time), 'seconds')
        print('[{}] '.format(k), 'Computed Loss #2:', loss.item())
    print('------------------------------------------------------------------')

