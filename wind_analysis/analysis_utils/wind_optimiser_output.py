import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator
from analysis_utils.plotting_analysis import plot_prediction_observations, plot_wind_estimates, plot_wind_3d
import datetime
import torch


def angle_wrap(angles):
    # Wrap angles to [-pi, pi)
    return (angles + np.pi) % (2 * np.pi) - np.pi


class WindOptimiserOutput:
    def __init__(self, wind_opt, wind_predictions, losses, inputs):
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.wind_opt = wind_opt
        # self._optimisers = opt
        self._wind_predictions = wind_predictions
        self._wind_prediction = self.get_best_wind_estimate()
        self._losses = losses
        self._inputs = inputs
        self._masked_input = self.get_masked_input()
        # self._grads = grads
        # self._names = self.get_names()
        # self._wind_prediction, self._best_method_index, self._best_ov = self.get_best_wind_estimate()
        self._save_output = False
        self._add_sparse_mask_row = True
        self._base_path = "analysis_output/"
        self._current_time = str(datetime.datetime.now().time())

    def get_names(self):
        # Optimisers names
        names = [o.opt.__name__ for o in self._optimisers]
        return names

    def get_best_wind_estimate(self):
        # # Extract best wind estimate
        # best_method_index = np.argmin([l[-1] for l in self._losses])
        # best_ov = self._wind_predictions[best_method_index]
        # best_opt_var = [best_ov[-1, 0], best_ov[-1, 1], best_ov[-1, 2], best_ov[-1, 3]]
        # self.wind_opt.reset_optimisation_variables(best_opt_var)
        # wind_prediction = self.wind_opt.get_prediction().detach()
        # return wind_prediction, best_method_index, best_ov
        # if len(self._wind_predictions) > 3:
        wind_prediction = self._wind_predictions[-1]
        # else:
        #     wind_prediction = self._wind_predictions
        return wind_prediction

    def get_masked_input(self):
        input = self._inputs[-1]
        sparse_mask = input[4, :].unsqueeze(0).clone()
        masked_input = sparse_mask.repeat(3, 1, 1, 1) * input[1:4, :, :]
        return masked_input

    def plot_wind_profile(self):
        fig, ax = plt.subplots(4, 4)
        optimized_corners = 4
        input_ = self.wind_opt.generate_wind_input()
        output_ = self.wind_opt.get_prediction()
        for i in range(optimized_corners):
            terrain_corners = input_[0, i//2, (i+2) % 2, :]
            wind_corners = input_[1:-1, i//2, (i + 2) % 2, :]
        hor_wind_speed = []
        heights = []
        if self._save_output:
            self.pp.savefig(fig)
        else:
            plt.show()

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

        if self._save_output:
            self.pp.savefig(fig)
        else:
            plt.show()

    def plot_final_values(self):
        # Plot final values and associated losses
        fig, ax = plt.subplots()
        for ov, loss in zip(self._wind_predictions, self._losses):
            neg_scale = ov[:, 1] < 0
            ov[neg_scale, 0] += np.pi
            ov[neg_scale, 1] *= -1
            ov[:, 0] = angle_wrap(ov[:, 0])
            ax.plot(ov[:, 0] * 180.0 / np.pi, ov[:, 1])
            ax.scatter(ov[-1, 0] * 180.0 / np.pi, ov[-1, 1])
            ax.text(ov[-1, 0] * 180.0 / np.pi, ov[-1, 1], "{0:0.3e}".format(loss[-1]))
        ax.legend(ax.lines, self._names)
        ax.set_xlabel('Rotation (deg)')
        ax.set_ylabel('Scale')

        if self._save_output:
            self.pp.savefig(fig)
        else:
            plt.show()

    def plot_wind_over_time(self):
        # Plot wind over time
        w_vanes = np.array([self.wind_opt._flight_data['we'], self.wind_opt._flight_data['wn'], self.wind_opt._flight_data['wd']])
        w_ekfest = np.array(
            [self.wind_opt._flight_data['we_east'], self.wind_opt._flight_data['we_north'], self.wind_opt._flight_data['we_down']])
        all_winds = [w_vanes, w_ekfest]
        plot_time = (self.wind_opt._flight_data['gp_time'] - self.wind_opt._flight_data['gp_time'][0]) * 1e-6
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
        inbounds = np.ones(self.wind_opt._flight_data['x'].shape, dtype='bool')
        inbounds = np.logical_and.reduce(
            [self.wind_opt._flight_data['x'] > x_terr2[0], self.wind_opt._flight_data['x'] < x_terr2[-1], inbounds])
        inbounds = np.logical_and.reduce(
            [self.wind_opt._flight_data['y'] > y_terr2[0], self.wind_opt._flight_data['y'] < y_terr2[-1], inbounds])
        inbounds = np.logical_and.reduce(
            [self.wind_opt._flight_data['alt'] > z_terr2[0], self.wind_opt._flight_data['alt'] < z_terr2[-1], inbounds])

        pred_t = (self.wind_opt._flight_data['gp_time'][inbounds] - self.wind_opt._flight_data['gp_time'][0]) * 1e-6
        points = np.array([self.wind_opt._flight_data['alt'][inbounds], self.wind_opt._flight_data['y'][inbounds],
                           self.wind_opt._flight_data['x'][inbounds]]).T
        pred_wind = [prediction_interp[0](points), prediction_interp[1](points), prediction_interp[2](points)]

        opt_var, opt_var_names = self.wind_opt.get_optimisation_variables()
        self.wind_opt.reset_optimisation_variables(opt_var)
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

        if self._save_output:
            self.pp.savefig(fig)
        else:
            plt.show()

    def plot_best_wind_estimate(self):
        # Plot best wind estimate
        fig, ax = plot_prediction_observations(self._wind_prediction, self.wind_opt.labels.to(self._device),
                                               self.wind_opt.terrain.network_terrain.squeeze(0),
                                               self._save_output, self._add_sparse_mask_row, self._masked_input)

        if self._save_output:
            self.pp.savefig(fig)
        else:
            plt.show()

    def print_losses(self):
        # Get minimum losses
        min_losses = []
        for o, loss, ov in zip(self._optimisers, self._losses, self._wind_predictions):
            min_loss = loss.min()
            idx = np.argmin(loss)
            opt_var = ov[idx, :]
            name = o.opt.__name__
            if self._save_output:
                file = open(self._base_path + self._current_time + ".txt", "a")
                file.write("{0}: minimum loss = {1}, rotation = {2} deg, scale = {3}, shear = {4}, exp = {5}".format(
                    name, min_loss,
                    opt_var[0] * 180.0 / np.pi,
                    opt_var[1], opt_var[2]* 180.0 / (np.pi*10000),
                    opt_var[3]/10 )
                           + "\n\n")
                if o == self._optimisers[-1]:
                    file.write("Best optimization method: {0}".format(
                        self._names[self._best_method_index]))
                file.close()
            else:
                print("{0}: minimum loss = {1}, rotation = {2} deg, scale = {3}, shear = {4}, exp = {5}".format(
                    name, min_loss,
                    opt_var[0] * 180.0 / np.pi,
                    opt_var[1], opt_var[2]* 180.0 / (np.pi*10000),
                    opt_var[3]/10 )
                           + "\n\n")
                if o == self._optimisers[-1]:
                    print("Best optimization method: {0}".format(
                        self._names[self._best_method_index]))

    def plot_trajectory_and_terrain(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # trajectory
        indeces = self._inputs[0][4, :].nonzero()
        wind_indices = np.array(indeces.cpu().detach().numpy())
        zs = wind_indices[:, 0]
        ys = wind_indices[:, 1]
        xs = wind_indices[:, 2]

        # skip values
        wskip = 1
        zs_skip = zs[::wskip]
        ys_skip = ys[::wskip]
        xs_skip = xs[::wskip]

        # plot trajectory
        ax.scatter(xs_skip, ys_skip, zs_skip, label='trajectory curve', color='red')
        ax.plot(xs_skip, ys_skip, zs_skip, label='trajectory curve', color='red')

        # plot wind vectors
        wind = self._inputs[0][1:-1, :].cpu().detach().numpy()
        ax.quiver(xs_skip, ys_skip, zs_skip, wind[0, zs_skip, ys_skip, xs_skip], wind[1, zs_skip, ys_skip, xs_skip],
                  wind[2, zs_skip, ys_skip, xs_skip],
                  length=1)


        # terrain
        h_grid = (self.wind_opt.terrain.z_terr[-1] - self.wind_opt.terrain.z_terr[0])/64
        h_network_terrain = np.floor((self.wind_opt.terrain.h_terr - self.wind_opt.terrain.z_terr[0])/h_grid)
        if 'torch' in str(h_network_terrain.dtype):
            h_network_terrain = h_network_terrain.detach().cpu().numpy()
        nx, ny = h_network_terrain.shape
        x = np.arange(0, nx, 1)
        y = np.arange(0, ny, 1)
        X, Y = np.meshgrid(x, y)
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, h_network_terrain, rstride=1, cstride=1, cmap=plt.cm.gray,
                        linewidth=0)
        # X, Y = np.meshgrid(self.wind_opt.terrain.x_terr/self.wind_opt.grid_size[0], self.wind_opt.terrain.y_terr//self.wind_opt.grid_size[0])
        # ax.plot_surface(X, Y, self.wind_opt.terrain.h_terr.detach().cpu().numpy()/self.wind_opt.grid_size[2], cmap=plt.cm.gray)

        # elev = 10
        # azim = -50
        # plt.gca().view_init(elev, azim)

        if self._save_output:
            self.pp.savefig(fig)
        else:
            plt.show()

    def close(self):
        self.pp.close()

    def plot(self):
        if self._save_output:
            self.pp = PdfPages(self._base_path + self._current_time + '.pdf')

        # self.plot_wind_profile()
        # self.plot_opt_convergence()
        # self.plot_final_values()
        # self.plot_wind_over_time()
        self.plot_trajectory_and_terrain()
        self.plot_best_wind_estimate()


        if self._save_output:
            self.close()

