import windseer.nn as nn_custom

import torch
import unittest

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#------------------------------------------ ADD CONFIGS TO TEST HERE ---------------------------------------------------
configs = []

# single loss testing : LPLoss  (only up to order 9, but can go beyond)
for p in range(1, 10):
    name = 'L{}Loss'.format(p)
    configs.append({
        'loss_components': [name],
        'learn_scaling': False,
        'auto_channel_scaling': False,
        'eps_scaling': 0.01,
        name + '_kwargs': {
            'exclude_terrain': True
            }
        })
    configs.append({
        'loss_components': [name],
        'learn_scaling': False,
        'auto_channel_scaling': False,
        'eps_scaling': 0.01,
        name + '_kwargs': {
            'exclude_terrain': False
            }
        })

# single loss testing : Scaled loss
configs.append({
    'loss_components': ['ScaledLoss'],
    'learn_scaling': False,
    'auto_channel_scaling': False,
    'eps_scaling': 0.01,
    'ScaledLoss_kwargs': {
        'exclude_terrain': True,
        'no_scaling': True,
        'loss_type': 'MSE',
        'norm_threshold': 0.5,
        'max_scale': 4.0
        }
    })

configs.append({
    'loss_components': ['ScaledLoss'],
    'learn_scaling': False,
    'auto_channel_scaling': False,
    'eps_scaling': 0.01,
    'ScaledLoss_kwargs': {
        'exclude_terrain': True,
        'no_scaling': False,
        'loss_type': 'MSE',
        'norm_threshold': 0.5,
        'max_scale': 4.0
        }
    })

configs.append({
    'loss_components': ['ScaledLoss'],
    'learn_scaling': False,
    'auto_channel_scaling': False,
    'eps_scaling': 0.01,
    'ScaledLoss_kwargs': {
        'exclude_terrain': False,
        'no_scaling': True,
        'loss_type': 'MSE',
        'norm_threshold': 0.5,
        'max_scale': 4.0
        }
    })

configs.append({
    'loss_components': ['ScaledLoss'],
    'learn_scaling': False,
    'auto_channel_scaling': False,
    'eps_scaling': 0.01,
    'ScaledLoss_kwargs': {
        'exclude_terrain': False,
        'no_scaling': True,
        'loss_type': 'MSE',
        'norm_threshold': 0.5,
        'max_scale': 4.0
        }
    })

# single loss testing : DFL
configs.append({
    'loss_components': ['DivergenceFreeLoss'],
    'learn_scaling': False,
    'auto_channel_scaling': False,
    'eps_scaling': 0.01,
    'DivergenceFreeLoss_kwargs': {
        'exclude_terrain': True,
        'loss_type': 'L1',
        'grid_size': [1, 1, 1]
        }
    })

configs.append({
    'loss_components': ['DivergenceFreeLoss'],
    'learn_scaling': False,
    'auto_channel_scaling': False,
    'eps_scaling': 0.01,
    'DivergenceFreeLoss_kwargs': {
        'exclude_terrain': False,
        'loss_type': 'L1',
        'grid_size': [1, 1, 1]
        }
    })

configs.append({
    'loss_components': ['DivergenceFreeLoss'],
    'learn_scaling': False,
    'auto_channel_scaling': False,
    'eps_scaling': 0.01,
    'DivergenceFreeLoss_kwargs': {
        'exclude_terrain': True,
        'loss_type': 'MSE',
        'grid_size': [1, 1, 1]
        }
    })

configs.append({
    'loss_components': ['DivergenceFreeLoss'],
    'learn_scaling': False,
    'auto_channel_scaling': False,
    'eps_scaling': 0.01,
    'DivergenceFreeLoss_kwargs': {
        'exclude_terrain': False,
        'loss_type': 'MSE',
        'grid_size': [1, 1, 1]
        }
    })

# single loss testing : VGL
configs.append({
    'loss_components': ['VelocityGradientLoss'],
    'learn_scaling': False,
    'auto_channel_scaling': False,
    'eps_scaling': 0.01,
    'VelocityGradientLoss_kwargs': {
        'exclude_terrain': True,
        'loss_type': 'L1',
        'grid_size': [1, 1, 1]
        }
    })

configs.append({
    'loss_components': ['VelocityGradientLoss'],
    'learn_scaling': False,
    'auto_channel_scaling': False,
    'eps_scaling': 0.01,
    'VelocityGradientLoss_kwargs': {
        'exclude_terrain': False,
        'loss_type': 'L1',
        'grid_size': [1, 1, 1]
        }
    })

configs.append({
    'loss_components': ['VelocityGradientLoss'],
    'learn_scaling': False,
    'auto_channel_scaling': False,
    'eps_scaling': 0.01,
    'VelocityGradientLoss_kwargs': {
        'exclude_terrain': True,
        'loss_type': 'MSE',
        'grid_size': [1, 1, 1]
        }
    })

configs.append({
    'loss_components': ['VelocityGradientLoss'],
    'learn_scaling': False,
    'auto_channel_scaling': False,
    'eps_scaling': 0.01,
    'VelocityGradientLoss_kwargs': {
        'exclude_terrain': False,
        'loss_type': 'MSE',
        'grid_size': [1, 1, 1]
        }
    })

# single loss testing : KLDiv
configs.append({
    'loss_components': ['KLDivLoss'],
    'learn_scaling': False,
    'auto_channel_scaling': False,
    'eps_scaling': 0.01,
    'KLDivLoss_kwargs': {}
    })

# single loss testing : GLL
configs.append({
    'loss_components': ['GaussianLogLikelihoodLoss'],
    'learn_scaling': False,
    'auto_channel_scaling': False,
    'eps_scaling': 0.01,
    'GaussianLogLikelihoodLoss_kwargs': {
        'exclude_terrain': True,
        'uncertainty_loss_eps': 1e-8
        }
    })

configs.append({
    'loss_components': ['GaussianLogLikelihoodLoss'],
    'learn_scaling': False,
    'auto_channel_scaling': False,
    'eps_scaling': 0.01,
    'GaussianLogLikelihoodLoss_kwargs': {
        'exclude_terrain': True,
        'uncertainty_loss_eps': 1e-8
        }
    })

# multiple combined loss testing
configs.append({
    'loss_components': ['L2Loss', 'L1Loss', 'KLDivLoss'],
    'learn_scaling': False,
    'auto_channel_scaling': True,
    'eps_scaling': 0.01,
    'eps_scheduling_kwargs': {},
    'L2Loss_kwargs': {
        'exclude_terrain': True,
        'loss_factor_init': 1.0
        },
    'L1Loss_kwargs': {
        'exclude_terrain': True,
        'loss_factor_init': 1.0
        },
    'KLDivLoss_kwargs': {
        'loss_factor_init': 1.0
        }
    })

configs.append({
    'loss_components': ['L2Loss', 'L1Loss', 'KLDivLoss'],
    'learn_scaling': False,
    'auto_channel_scaling': False,
    'eps_scaling': 0.01,
    'L2Loss_kwargs': {
        'exclude_terrain': True,
        'loss_factor_init': 1.0
        },
    'L1Loss_kwargs': {
        'exclude_terrain': True,
        'loss_factor_init': 1.0
        },
    'KLDivLoss_kwargs': {
        'loss_factor_init': 1.0
        }
    })

configs.append({
    'loss_components': ['DivergenceFreeLoss', 'L1Loss'],
    'learn_scaling': True,
    'auto_channel_scaling': False,
    'eps_scaling': 0.01,
    'DivergenceFreeLoss_kwargs': {
        'exclude_terrain': True,
        'loss_type': 'L1',
        'grid_size': [1, 1, 1],
        'loss_factor_init': 1.0
        },
    'L1Loss_kwargs': {
        'exclude_terrain': True,
        'loss_factor_init': 1.0
        }
    })

configs.append({
    'loss_components': ['VelocityGradientLoss', 'L2Loss'],
    'learn_scaling': True,
    'auto_channel_scaling': False,
    'eps_scaling': 0.01,
    'VelocityGradientLoss_kwargs': {
        'exclude_terrain': True,
        'loss_type': 'L1',
        'grid_size': [1, 1, 1],
        'loss_factor_init': 1.0
        },
    'L2Loss_kwargs': {
        'exclude_terrain': True,
        'loss_factor_init': 1.0
        }
    })

configs.append({
    'loss_components': [
        'VelocityGradientLoss', 'DivergenceFreeLoss', 'L2Loss', 'L1Loss'
        ],
    'learn_scaling': True,
    'auto_channel_scaling': False,
    'eps_scaling': 0.01,
    'VelocityGradientLoss_kwargs': {
        'exclude_terrain': True,
        'loss_type': 'L1',
        'grid_size': [1, 1, 1],
        'loss_factor_init': 1.0
        },
    'L1Loss_kwargs': {
        'exclude_terrain': True,
        'loss_factor_init': 1.0
        },
    'L2Loss_kwargs': {
        'exclude_terrain': True,
        'loss_factor_init': 1.0
        },
    'DivergenceFreeLoss_kwargs': {
        'exclude_terrain': True,
        'loss_type': 'L1',
        'grid_size': [1, 1, 1],
        'loss_factor_init': 1.0
        },
    })
#-----------------------------------------------------------------------------------------------------------------------


class TestLosses(unittest.TestCase):

    def run_test(self, input, label, W, output, config):
        loss_fn = nn_custom.CombinedLoss(**config)

        param_list = []
        if loss_fn.learn_scaling:
            param_list.append({'params': loss_fn.parameters()})
            optimizer = torch.optim.Adam(params=param_list, lr=1e-1)
            optimizer.zero_grad()

        loss = loss_fn(output, label, input, W)

        loss.backward()

        if loss_fn.learn_scaling:
            optimizer.step()

        return True

    def test_losses(self):
        input = torch.rand(10, 4, 64, 64, 64, requires_grad=False)
        label = torch.rand(10, 4, 64, 64, 64, requires_grad=False)
        input[:, 0, :10, :, :] = 0.0  # generate some terrain
        W = torch.ones(10, 1, 64, 64, 64)  # generate a weighting function for the loss

        label, input, W = label.to(device), input.to(device), W.to(device)

        for cfg in configs:
            output = {
                'pred': torch.rand(10, 4, 64, 64, 64, requires_grad=True),
                'logvar': torch.rand(10, 4, 64, 64, 64, requires_grad=True),
                'distribution_mean': torch.rand(10, 128, requires_grad=True),
                'distribution_logvar': torch.rand(10, 128, requires_grad=True),
                }
            for key in output.keys():
                output[key] = output[key].to(device)
            self.assertTrue(self.run_test(input, label, W, output, cfg))


if __name__ == '__main__':
    unittest.main()
