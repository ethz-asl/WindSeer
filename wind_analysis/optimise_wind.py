import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from scipy.interpolate import RegularGridInterpolator
from wind_optimiser import WindOptimiser, OptTest, SimpleStepOptimiser
from analysis_utils.plotting_analysis import plot_prediction_observations, plot_wind_estimates

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

# Extract best wind estimate
best_method_index = np.argmin([l[-1] for l in losses])
best_rs = all_rs[best_method_index]
wind_opt.reset_rotation_scale(rot=best_rs[-1,0], scale=best_rs[-1,1])
wind_prediction = wind_opt.get_prediction().detach()

# Plot wind over time
w_vanes = np.array([wind_opt._ulog_data['we'], wind_opt._ulog_data['wn'], wind_opt._ulog_data['wd']])
w_ekfest = np.array([wind_opt._ulog_data['we_east'], wind_opt._ulog_data['we_north'], wind_opt._ulog_data['we_down']])
all_winds = [w_vanes, w_ekfest]
plot_time = (wind_opt._ulog_data['gp_time'] - wind_opt._ulog_data['gp_time'][0]) * 1e-6
fig3, ax3 = plot_wind_estimates(plot_time, all_winds, ['Raw vane estimates', 'On-board EKF estimate'], polar=False)

x_terr2 = np.linspace(wind_opt.terrain.x_terr[0], wind_opt.terrain.x_terr[-1], wind_prediction.shape[-1])
y_terr2 = np.linspace(wind_opt.terrain.y_terr[0], wind_opt.terrain.y_terr[-1], wind_prediction.shape[-2])
z_terr2 = np.linspace(wind_opt.terrain.z_terr[0], wind_opt.terrain.z_terr[-1], wind_prediction.shape[-3])
prediction_interp = []
for pred_dim in wind_prediction:
    prediction_interp.append(RegularGridInterpolator((z_terr2, y_terr2, x_terr2), pred_dim))

# Get all the in bounds points
inbounds = np.ones(wind_opt._ulog_data['x'].shape, dtype='bool')
inbounds = np.logical_and.reduce([wind_opt._ulog_data['x'] > x_terr2[0], wind_opt._ulog_data['x'] < x_terr2[-1], inbounds])
inbounds = np.logical_and.reduce([wind_opt._ulog_data['y'] > y_terr2[0], wind_opt._ulog_data['y'] < y_terr2[-1], inbounds])
inbounds = np.logical_and.reduce([wind_opt._ulog_data['alt'] > z_terr2[0], wind_opt._ulog_data['alt'] < z_terr2[-1], inbounds])

pred_t = (wind_opt._ulog_data['gp_time'][inbounds] - wind_opt._ulog_data['gp_time'][0])*1e-6
points = np.array([wind_opt._ulog_data['alt'][inbounds], wind_opt._ulog_data['y'][inbounds], wind_opt._ulog_data['x'][inbounds]]).T
pred_wind = [prediction_interp[0](points), prediction_interp[1](points), prediction_interp[2](points)]

wind_opt.reset_rotation_scale(rot=0.0, scale=1.0)
orig_wind_prediction = wind_opt.get_prediction().detach()
orig_prediction_interp = []
for pred_dim in orig_wind_prediction:
    orig_prediction_interp.append(RegularGridInterpolator((z_terr2, y_terr2, x_terr2), pred_dim))
orig_pred_wind = [orig_prediction_interp[0](points), orig_prediction_interp[1](points), orig_prediction_interp[2](points)]

ax3[0].plot(pred_t, orig_pred_wind[0], 'g.', ms=3)
ax3[1].plot(pred_t, orig_pred_wind[1], 'g.', ms=3)
ax3[2].plot(pred_t, orig_pred_wind[2], 'g.', ms=3)
ax3[0].plot(pred_t, pred_wind[0], 'r.', ms=3)
ax3[1].plot(pred_t, pred_wind[1], 'r.', ms=3)
ax3[2].plot(pred_t, pred_wind[2], 'r.', ms=3)
ax3[0].legend(['Raw vane estimates', 'On-board EKF estimate', 'Pre-optimisation network estimate', 'Post-optimisation network estimate'])

# Plot best wind estimate
print('Plotting for optimal method {0}, rotation = {1:0.3f} deg, scale = {2:0.3f}'.format(names[best_method_index],
      best_rs[-1, 0]*180.0/np.pi, best_rs[-1, 1]))
plot_prediction_observations(wind_prediction, wind_opt._wind_blocks, wind_opt.terrain.network_terrain.squeeze(0))

plt.show()
