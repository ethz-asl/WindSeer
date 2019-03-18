from wind_optimiser import WindOptimiser, OptTest, SimpleStepOptimiser
import torch
import matplotlib.pyplot as plt
import numpy as np
import nn_wind_prediction.utils as utils
import argparse


def angle_wrap(angles):
    # Wrap angles to [-pi, pi)
    return (angles + np.pi) % (2 * np.pi) - np.pi


parser = argparse.ArgumentParser(description='Optimise wind speed and direction from COSMO data using observations')
parser.add_argument('input_yaml', help='Input yaml config')
parser.add_argument('-n', '--n_steps', type=int, default=200, help='Number of optimisation steps')
parser.add_argument('-r', '--rotation', type=float, default=0.0, help='Initial rotation (rad)')
parser.add_argument('-s', '--scale', type=float, default=1.0, help='Initial scale')
args = parser.parse_args()

# Create WindOptimiser object using yaml config
wind_opt = WindOptimiser(args.input_yaml)

# Testing a range of different optimisers from torch.optim, and a basic gradient step (SimpleStepOptimiser)
optimisers = [OptTest(SimpleStepOptimiser, {'lr': 5.0, 'lr_decay': 0.01}),
              OptTest(torch.optim.Adadelta, {'lr': 1.0}),
              OptTest(torch.optim.Adagrad, {'lr': 1.0, 'lr_decay': 0.1}),
              OptTest(torch.optim.Adam, {'lr': 1.0, 'betas': (.9, .999)}),
              OptTest(torch.optim.Adamax, {'lr': 1.0, 'betas': (.9, .999)}),
              OptTest(torch.optim.ASGD, {'lr': 2.0, 'lambd': 1e-3}),
              OptTest(torch.optim.SGD, {'lr': 2.0, 'momentum': 0.5, 'nesterov': True}),
              ]

# Try each optimisation method
all_rs, losses, grads = [], [], []
for i, o in enumerate(optimisers):
    wind_opt.reset_rotation_scale(args.rotation, args.scale)
    rs, loss, grad = wind_opt.optimise_rotation_scale(o.opt, n=args.n_steps, opt_kwargs=o.kwargs, verbose=False)
    all_rs.append(rs)
    losses.append(loss)
    grads.append(grad)

# Plot results for all optimisers
fig, ax = plt.subplots(1, 2)
names = [o.opt.__name__ for o in optimisers]
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

# Plot final values and associated losses
fig2, ax2 = plt.subplots()
for rs, loss, l in zip(all_rs, losses, ax[0].lines):
    neg_scale = rs[:,1] < 0
    rs[neg_scale, 0] += np.pi
    rs[neg_scale, 1] *= -1
    rs[:,0] = angle_wrap(rs[:,0])
    ax2.plot(rs[:,0]*180.0/np.pi, rs[:,1], color=l.get_color())
    ax2.scatter(rs[-1, 0] * 180.0 / np.pi, rs[-1, 1], c=l.get_color())
    ax2.text(rs[-1,0]*180.0/np.pi, rs[-1,1], "{0:0.3e}".format(loss[-1]))
ax2.legend(ax2.lines, names)
ax2.set_xlabel('Rotation (deg)')
ax2.set_ylabel('Scale')

# Plot best wind estimate
best_method_index = np.argmin([l[-1] for l in losses])
best_rs = all_rs[best_method_index]
wind_opt.reset_rotation_scale(rot=best_rs[-1,0], scale=best_rs[-1,1])
print('Plotting for optimal method {0}, rotation = {1:0.3f} deg, scale = {2:0.3f}'.format(names[best_method_index],
      best_rs[-1, 0]*180.0/np.pi, best_rs[-1, 1]))
utils.plot_prediction_observations(wind_opt.get_prediction().detach(), wind_opt._wind_blocks, wind_opt.terrain.network_terrain.squeeze(0))

plt.show(block=False)