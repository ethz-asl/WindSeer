import nn_wind_prediction.utils as utils
from wind_optimiser import WindOptimiser, OptTest, SimpleStepOptimiser
import torch
import matplotlib.pyplot as plt

# Create WindOptimiser object using yaml config
wind_opt = WindOptimiser('config/optim_config.yaml')

# Testing a range of different optimisers from torch.optim, and a basic gradient step (SimpleStepOptimiser)
optimisers = [OptTest(SimpleStepOptimiser, {'lr': 10.0, 'lr_decay': 0.01}),
              OptTest(torch.optim.Adadelta, {'lr': 1.0}),
              OptTest(torch.optim.Adagrad, {'lr': 1.0, 'lr_decay': 0.1}),
              OptTest(torch.optim.Adam, {'lr': 1.0, 'betas': (.9, .999)}),
              OptTest(torch.optim.Adamax, {'lr': 1.0, 'betas': (.9, .999)}),
              OptTest(torch.optim.ASGD, {'lr': 5.0, 'lambd': 1e-3}),
              OptTest(torch.optim.SGD, {'lr': 5.0, 'momentum': 0.5, 'nesterov': True}),
              ]

# Try each optimisation method
n_steps = 200
losses, grads, names = [], [], []
for i, o in enumerate(optimisers):
    loss, grad, no = wind_opt.optimise_rotation_scale(o.opt, n=n_steps, opt_kwargs=o.kwargs)
    losses.append(loss)
    grads.append(grad)
    names.append(no)
    wind_opt.reset_rotation_scale()

# Plot results for all optimisers
fig, ax = plt.subplots(1, 2)
loss_lines, grad_lines = [], []
for l, g in zip(losses, grads):
    loss_lines.append(ax[0].plot(range(len(l)), l)[0])
    grad_lines.append(ax[1].plot(range(len(g)), g)[0])
ax[0].legend(loss_lines, names)
ax[0].set_yscale('log')
ax[0].set_xlabel('Optimisation steps')
ax[0].set_ylabel('Loss ({0})'.format(wind_opt._loss_fn))
ax[1].set_xlabel('Optimisation steps')
ax[1].set_ylabel('Max. loss gradient')
# utils.plot_sample(output.detach(), test.wind_blocks, test.terrain.network_terrain.squeeze(0))
plt.show(block=False)