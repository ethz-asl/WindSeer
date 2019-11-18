import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import RegularGridInterpolator
from analysis_utils.plotting_analysis import plot_prediction_observations, plot_wind_estimates


def angle_wrap(angles):
    # Wrap angles to [-pi, pi)
    return (angles + np.pi) % (2 * np.pi) - np.pi


class WindOptimiserOutput:
    def __init__(self, wind_opt, opt, all_rs, losses, grads):
        self.wind_opt = wind_opt
        self._optimisers = opt
        self._all_rs = all_rs
        self._losses = losses
        self._grads = grads
        self._names = self.get_names()
        self._wind_prediction, self._best_method_index, self._best_rs = self.get_best_wind_estimate()

    def get_names(self):
        # Optimisers names
        names = [o.opt.__name__ for o in self._optimisers]
        return names

    def get_best_wind_estimate(self):
        # Extract best wind estimate
        best_method_index = np.argmin([l[-1] for l in self._losses])
        best_rs = self._all_rs[best_method_index]
        self.wind_opt.reset_rotation_scale(rot=best_rs[-1, 0], scale=best_rs[-1, 1])
        wind_prediction = self.wind_opt.get_prediction().detach()
        return wind_prediction, best_method_index, best_rs

    def plot_opt_convergence(self):
        # Plot results for all optimisers
        fig, ax = plt.subplots(1, 2)
        names = [o.opt.__name__ for o in self._optimisers]
        loss_lines, grad_lines = [], []
        for l, g in zip(self._losses, self._grads):
            loss_lines.append(ax[0].plot(range(len(l)), l)[0])
            grad_lines.append(ax[1].plot(range(len(g)), g)[0])
        ax[0].legend(loss_lines, names)
        ax[0].set_yscale('log')
        ax[0].set_xlabel('Optimisation steps')
        ax[0].set_ylabel('Loss ({0})'.format(self.wind_opt._loss_fn))
        ax[1].set_xlabel('Optimisation steps')
        ax[1].set_ylabel('Max. loss gradient')

        plt.show()
        #self.pp.savefig(fig)

    def plot_final_values(self):
        # Plot final values and associated losses
        fig, ax = plt.subplots()
        for rs, loss in zip(self._all_rs, self._losses):
            neg_scale = rs[:, 1] < 0
            rs[neg_scale, 0] += np.pi
            rs[neg_scale, 1] *= -1
            rs[:, 0] = angle_wrap(rs[:, 0])
            ax.plot(rs[:, 0] * 180.0 / np.pi, rs[:, 1])
            ax.scatter(rs[-1, 0] * 180.0 / np.pi, rs[-1, 1])
            ax.text(rs[-1, 0] * 180.0 / np.pi, rs[-1, 1], "{0:0.3e}".format(loss[-1]))
        ax.legend(ax.lines, self._names)
        ax.set_xlabel('Rotation (deg)')
        ax.set_ylabel('Scale')

        plt.show()
        #self.pp.savefig(fig)

    def plot_wind_over_time(self):
        # Plot wind over time
        w_vanes = np.array([self.wind_opt._ulog_data['we'], self.wind_opt._ulog_data['wn'], self.wind_opt._ulog_data['wd']])
        w_ekfest = np.array(
            [self.wind_opt._ulog_data['we_east'], self.wind_opt._ulog_data['we_north'], self.wind_opt._ulog_data['we_down']])
        all_winds = [w_vanes, w_ekfest]
        plot_time = (self.wind_opt._ulog_data['gp_time'] - self.wind_opt._ulog_data['gp_time'][0]) * 1e-6
        fig, ax = plot_wind_estimates(plot_time, all_winds, ['Raw vane estimates', 'On-board EKF estimate'],
                                        polar=False)

        x_terr2 = np.linspace(self.wind_opt.terrain.x_terr[0], self.wind_opt.terrain.x_terr[-1], self._wind_prediction.shape[-1])
        y_terr2 = np.linspace(self.wind_opt.terrain.y_terr[0], self.wind_opt.terrain.y_terr[-1], self._wind_prediction.shape[-2])
        z_terr2 = np.linspace(self.wind_opt.terrain.z_terr[0], self.wind_opt.terrain.z_terr[-1], self._wind_prediction.shape[-3])
        prediction_interp = []
        for pred_dim in self._wind_prediction:
            # Convert torch tensor to numpy array to work with RegularGridInterpolator
            pred_dim = pred_dim.detach().cpu().numpy()
            prediction_interp.append(RegularGridInterpolator((z_terr2, y_terr2, x_terr2), pred_dim))

        # Get all the in bounds points
        inbounds = np.ones(self.wind_opt._ulog_data['x'].shape, dtype='bool')
        inbounds = np.logical_and.reduce(
            [self.wind_opt._ulog_data['x'] > x_terr2[0], self.wind_opt._ulog_data['x'] < x_terr2[-1], inbounds])
        inbounds = np.logical_and.reduce(
            [self.wind_opt._ulog_data['y'] > y_terr2[0], self.wind_opt._ulog_data['y'] < y_terr2[-1], inbounds])
        inbounds = np.logical_and.reduce(
            [self.wind_opt._ulog_data['alt'] > z_terr2[0], self.wind_opt._ulog_data['alt'] < z_terr2[-1], inbounds])

        pred_t = (self.wind_opt._ulog_data['gp_time'][inbounds] - self.wind_opt._ulog_data['gp_time'][0]) * 1e-6
        points = np.array([self.wind_opt._ulog_data['alt'][inbounds], self.wind_opt._ulog_data['y'][inbounds],
                           self.wind_opt._ulog_data['x'][inbounds]]).T
        pred_wind = [prediction_interp[0](points), prediction_interp[1](points), prediction_interp[2](points)]

        self.wind_opt.reset_rotation_scale(rot=0.0, scale=1.0)
        orig_wind_prediction = self.wind_opt.get_prediction().detach()
        orig_prediction_interp = []
        for pred_dim in orig_wind_prediction:
            # Convert torch tensor to numpy array to work with RegularGridInterpolator
            pred_dim = pred_dim.detach().cpu().numpy()
            orig_prediction_interp.append(RegularGridInterpolator((z_terr2, y_terr2, x_terr2), pred_dim))
        orig_pred_wind = [orig_prediction_interp[0](points), orig_prediction_interp[1](points),
                          orig_prediction_interp[2](points)]

        ax[0].plot(pred_t, orig_pred_wind[0], 'g.', ms=3)
        ax[1].plot(pred_t, orig_pred_wind[1], 'g.', ms=3)
        ax[2].plot(pred_t, orig_pred_wind[2], 'g.', ms=3)
        ax[0].plot(pred_t, pred_wind[0], 'r.', ms=3)
        ax[1].plot(pred_t, pred_wind[1], 'r.', ms=3)
        ax[2].plot(pred_t, pred_wind[2], 'r.', ms=3)
        ax[0].legend(['Raw vane estimates', 'On-board EKF estimate', 'Pre-optimisation network estimate',
                       'Post-optimisation network estimate'])

        plt.show()
        #self.pp.savefig(fig)

    def plot_best_wind_estimate(self):
        # Plot best wind estimate
        fig = plt.figure()
        print('Plotting for optimal method {0}, rotation = {1:0.3f} deg, scale = {2:0.3f}'.format(
            self._names[self._best_method_index],
            self._best_rs[-1, 0] * 180.0 / np.pi, self._best_rs[-1, 1]))
        plot_prediction_observations(self._wind_prediction, self.wind_opt._wind_blocks,
                                     self.wind_opt.terrain.network_terrain.squeeze(0))

        plt.show()
        #self.pp.savefig(fig)

    def print_losses(self):
        # Get minimum losses
        min_losses = []
        for loss in self._losses:
            min_losses.append(loss.min())
        min_losses_indices = np.argsort(min_losses)
        min_loss = min(min_losses)

        print("Minimum losses for each optimizer are: ", min_losses)
        print("Indices of the minimum losses are: ", min_losses_indices)
        print("Mimimum loss is: ", min_loss)

    def close(self):
            self.pp.close()

    def plot(self):
        #self.pp = PdfPages(self.basePath + str(self.c * self.th) + "_" + str(self.mTh) + "_" + str(self.r) + '.pdf')
        self.plot_opt_convergence()
        self.plot_final_values()
        self.plot_wind_over_time()
        self.plot_best_wind_estimate()
        #self.close()

