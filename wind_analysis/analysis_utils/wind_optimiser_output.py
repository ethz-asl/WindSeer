import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.transforms as transforms
from scipy.interpolate import RegularGridInterpolator
from analysis_utils.plotting_analysis import plot_prediction_observations, plot_wind_estimates
import pandas as pd
import datetime
import torch
from mayavi import mlab
from scipy import signal


def angle_wrap(angles):
    # Wrap angles to [-pi, pi)
    return (angles + np.pi) % (2 * np.pi) - np.pi


class WindOptimiserOutput:
    def __init__(self, wind_opt, wind_predictions, losses, inputs, losses_dict=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.wind_opt = wind_opt
        # self._optimisers = opt
        self._wind_predictions = wind_predictions
        self._losses = losses
        self._inputs = inputs
        self._losses_dict = losses_dict
        self._wind_prediction, self._input, self._loss = self.get_last_wind_estimate()
        self._masked_input = self.get_masked_input()
        # self._grads = grads
        # self._names = self.get_names()
        self._save_output = False
        self._add_sparse_mask_row = True
        self._base_path = "analysis_output/"
        self._current_time = str(datetime.datetime.now().time())

    def get_names(self):
        # Optimisers names
        names = [o.opt.__name__ for o in self._optimisers]
        return names

    def get_last_wind_estimate(self):
        # Extract last wind estimate
        if not self._wind_predictions:
            wind_prediction = []
        else:
            wind_prediction = self._wind_predictions[-1]
        if not self._inputs:
            input = []
        else:
            input = self._inputs[-1]
        if not self._losses:
            loss = []
        else:
            loss = self._losses[-1]
        return wind_prediction, input, loss

    def get_masked_input(self):
        input = self._input
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
                    opt_var[1], opt_var[2] * 180.0 / (np.pi*10000),
                    opt_var[3]/10 )
                           + "\n\n")
                if o == self._optimisers[-1]:
                    print("Best optimization method: {0}".format(
                        self._names[self._best_method_index]))

    def plot_high_res_version(self):
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        mlab.options.backend = 'envisage'
        mlab.figure(size=(3840, 2160), bgcolor=(1, 1, 1))

        # terrain
        h_grid = self.wind_opt.terrain.z_terr[1] - self.wind_opt.terrain.z_terr[0]
        # h_network_terrain = np.floor((self.wind_opt.terrain.h_terr - self.wind_opt.terrain.z_terr[0])/h_grid)
        h_network_terrain = (self.wind_opt.terrain.h_terr - self.wind_opt.terrain.z_terr[0]) / h_grid
        ny, nx = h_network_terrain.shape
        # size = 1
        # output_shape = (ny*size, nx*size)
        # h_network_terrain = skimage.transform.resize(h_network_terrain, output_shape)
        ny, nx = h_network_terrain.shape
        x = np.arange(0, nx, 1)
        y = np.arange(0, ny, 1)
        # X, Y = np.meshgrid(x, y)
        # smooth out terrain
        # h_network_terrain = scipy.ndimage.filters.gaussian_filter(h_network_terrain, sigma=1)

        # mlab plot
        terrain_surf = mlab.surf(x, y, h_network_terrain.transpose(), colormap='Greys')
        # fill area beneath surface plot
        # outline
        cat_extent = (0, 64, 0, 64, 0, 64)
        mlab.outline(terrain_surf, color=(.7, .7, .7), extent=cat_extent)

        # # wind field
        wind_prediction = self._wind_prediction.detach().cpu()
        #rotation
        # wind_prediction[0, :] = wind_prediction[0, :] * np.sin(np.pi/3) - wind_prediction[1, :] * np.cos(np.pi/3)
        # wind_prediction[1, :] = wind_prediction[0, :] * np.sin(np.pi /3) + wind_prediction[0, :] * np.cos(np.pi /3)
        channels, nz, ny, nx = wind_prediction.shape

        # skip values
        wskip = 63
        xf_skip = np.arange(0, nx, 1)[::wskip]
        yf_skip = np.arange(0, ny, 1)[::wskip]
        zf_skip = np.arange(0, nz, 1)
        xv_skip, yv_skip, zv_skip = np.meshgrid(xf_skip, yf_skip, zf_skip)
        # mlab.quiver3d(xv_skip, yv_skip, zv_skip, wind_prediction[0, zv_skip, yv_skip, xv_skip],
                  # wind_prediction[1, zv_skip, yv_skip, xv_skip], wind_prediction[2, zv_skip, yv_skip, xv_skip], line_width=0.1)

        # voxel rendering
        wind_prediction_render = np.linalg.norm(wind_prediction.detach().cpu().numpy(), axis=0).transpose()
        min_v = wind_prediction_render.min()
        max_v = wind_prediction_render.max()
        # mlab.pipeline.volume(mlab.pipeline.scalar_field(wind_prediction_render), vmin=min_v + 0.35 * (max_v - min_v),
        #                      vmax=min_v + 0.9 * (max_v - min_v))
        mlab.view(220, 80, 200)

        if self._save_output:
            mlab.savefig(self._base_path + self._current_time + '.png')
        else:
            mlab.show()

    def plot_vertical_wind_profile(self):
        fig, ax = plt.subplots()

        if ax is None or fig is None:
            fig, ax = plt.subplots()
        vv = np.sqrt((self.wind_opt._base_cosmo_corners[0, :, 0, 0] ** 2 + self.wind_opt._base_cosmo_corners[1, :, 0, 0] ** 2).detach().cpu().numpy())
        ht, = ax.plot(vv[:], self.wind_opt.terrain.z_terr[:])
        v3 = np.sqrt(self.wind_opt._flight_data['wn'] ** 2 + self.wind_opt._flight_data['we'] ** 2)
        alt = self.wind_opt._flight_data['alt']
        t = (self.wind_opt._flight_data['time_microsec']-self.wind_opt._flight_data['time_microsec'][0])*1e-6
        h2 = ax.scatter(v3, alt, c=t, s=10)
        ax.grid(b=True, which='both')
        ax.set_xlabel('Wind speed (m/s)')
        ax.set_ylabel('Alt, m')
        ax.set_ylim(np.floor(alt.min() / 100) * 100, np.ceil(alt.max() / 100) * 100)
        hl = ax.legend([ht, h2], ['COSMO profile', 'UAV data'])
        hc = fig.colorbar(h2, ax=ax)
        hc.set_label('Mission time (s)')

        if self._save_output:
            self.pp.savefig(fig)
        else:
            plt.show()

    def plot_fft_analysis(self):
        fig, ax = plt.subplots()

        # bach test
        test_set_range = 1
        if self.wind_opt.flag.flight_batch_test and self.wind_opt.flag.test_flight_data:
            test_set_range = len(self.wind_opt._flight_args.params['files'])

        for i in range(test_set_range):
            if self.wind_opt.flag.test_flight_data and self.wind_opt.flag.flight_batch_test:
                flight_data = self.wind_opt.load_flight_data(i)
            elif self.wind_opt.flag.test_flight_data and not self.wind_opt.flag.flight_batch_test:
                flight_data = self.wind_opt._flight_data
            if self.wind_opt.flag.test_simulated_data:
                flight_data = self.wind_opt._simulated_flight_data

            # winds in each direction
            wn = flight_data['wn']
            we = flight_data['we']
            wd = flight_data['wd']

            # wind magnitude
            wind_magnitude = np.sqrt(wn**2 + we**2 + wd**2)

            # time
            time = (flight_data['time_microsec'] - flight_data['time_microsec'][0]) / 1e6

            T = time[1] - time[0]
            N = time.size

            f = np.linspace(0, 1/T, N)

            fft = np.fft.fft(wind_magnitude - wind_magnitude.mean())

            if self.wind_opt.flag.test_flight_data and self.wind_opt.flag.flight_batch_test:
                labels = ['13\_02\_02', '13\_33\_46', '14\_15\_04', '14\_47\_05', '14\_47\_05\_filtered']
                # labels = ['10\_36\_34', '11\_11\_57']
                ax.plot(f[:N // 2], np.abs(fft)[:N // 2] * 1 / N, label=labels[i])
                ax.legend(loc='upper right')
                ax.set_title('Riemenstalden')
                # ax.set_title('Fluelen')
            else:
                ax.plot(f[:N // 2], np.abs(fft)[:N // 2] * 1 / N)
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Amplitude')

            # ax.plot(time, wn - wn.mean())
            # ax.set_xlabel('time [s]')
            # ax.set_ylabel('Wind speed [m/s]')

        # make image full screen
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.maximize()

        if self._save_output:
            self.pp.savefig(fig)
        else:
            plt.show()

    def plot_trajectory_wind_vectors(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # terrain
        h_grid = self.wind_opt.terrain.z_terr[1] - self.wind_opt.terrain.z_terr[0]
        h_network_terrain = np.floor((self.wind_opt.terrain.h_terr - self.wind_opt.terrain.z_terr[0])/h_grid)
        ny, nx = h_network_terrain.shape
        x = np.arange(0, nx, 1)
        y = np.arange(0, ny, 1)
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, h_network_terrain, rstride=1, cstride=1, cmap=plt.cm.gray,
                        linewidth=0)

        # trajectory
        indices = self._input[4, :].nonzero()  # the nonzero values in the NN input correspond to the trajectory points
        trajectory_indices = np.asarray(indices.detach().cpu().numpy())
        zt = trajectory_indices[:, 0]
        yt = trajectory_indices[:, 1]
        xt = trajectory_indices[:, 2]

        # skip values
        wskip = 1
        xt_skip = xt[::wskip]
        yt_skip = yt[::wskip]
        zt_skip = zt[::wskip]

        # plot trajectory
        ax.scatter(xt_skip, yt_skip, zt_skip, label='trajectory curve', color='red')
        # ax.plot(xt_skip, yt_skip, zt_skip, label='trajectory curve', color='red')

        # plot wind vectors
        wind = self._input[1:-1, :].detach().cpu().numpy()
        ax.quiver(xt_skip, yt_skip, zt_skip, wind[0, zt_skip, yt_skip, xt_skip],
                  wind[1, zt_skip, yt_skip, xt_skip], wind[2, zt_skip, yt_skip, xt_skip], length=1.0)

        # make image full screen
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.maximize()

        if self._save_output:
            self.pp.savefig(fig)
        else:
            plt.show()

    def plot_wind_field(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # terrain
        h_grid = self.wind_opt.terrain.z_terr[1] - self.wind_opt.terrain.z_terr[0]
        h_network_terrain = np.floor((self.wind_opt.terrain.h_terr - self.wind_opt.terrain.z_terr[0])/h_grid)
        ny, nx = h_network_terrain.shape
        x = np.arange(0, nx, 1)
        y = np.arange(0, ny, 1)
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, h_network_terrain, rstride=1, cstride=1, cmap=plt.cm.gist_earth,
                        linewidth=0)

        # wind field
        wind_prediction = self._wind_prediction.detach().cpu()
        channels, nz, ny, nx = wind_prediction.shape

        # skip values
        wskip = 8
        xf_skip = np.arange(0, nx, 1)[::wskip]
        yf_skip = np.arange(0, ny, 1)[::wskip]
        zf_skip = np.arange(0, nz, 1)[::wskip]
        xv_skip, yv_skip, zv_skip = np.meshgrid(xf_skip, yf_skip, zf_skip)

        # set color map
        colormap = matplotlib.cm.inferno
        colors = torch.sqrt(wind_prediction[0, zv_skip, yv_skip, xv_skip]**2
                            + wind_prediction[1, zv_skip, yv_skip, xv_skip]**2
                            + wind_prediction[2, zv_skip, yv_skip, xv_skip]**2)
        colors = colors.view(-1)
        # normalize colors
        norm = matplotlib.colors.Normalize()
        norm.autoscale(colors)

        ax.quiver(xv_skip, yv_skip, zv_skip, wind_prediction[0, zv_skip, yv_skip, xv_skip],
                  wind_prediction[1, zv_skip, yv_skip, xv_skip], wind_prediction[2, zv_skip, yv_skip, xv_skip],
                  color=colormap(norm(colors)), length=2.0)

        # add color bar
        hc = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax)
        hc.set_label('Wind speed (m/s)')

        # make image full screen
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.maximize()

        if self._save_output:
            self.pp.savefig(fig)
        else:
            plt.show()

    def plot_wind_vectors_angles(self):
        fig, ax = plt.subplots()

        wind_vectors_angles = self.wind_opt.wind_vector_angles
        mean = wind_vectors_angles.mean()
        measurements = np.arange(0, len(wind_vectors_angles), 1)

        ax.bar(measurements, wind_vectors_angles, color='blue')
        ax.set_xlabel('Measurements')
        ax.set_ylabel('Angles between the wind vectors (deg)')
        # add mean line
        ax.axhline(mean, color='red', linewidth=2)
        trans = transforms.blended_transform_factory(
            ax.get_yticklabels()[0].get_transform(), ax.transData)
        ax.text(0, mean, "{:.2f}".format(mean), color='red', transform=trans, ha='right', va='center')

        # make image full screen
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.maximize()

    #     # input
    #     wind_vectors_angles = self.wind_opt.input_angles
    #     mean = wind_vectors_angles.mean()
    #     measurements = np.arange(0, len(wind_vectors_angles), 1)
    #
    #     ax.bar(measurements, wind_vectors_angles, color='blue')
    #     # add mean line
    #     ax.axhline(mean, color='red', linewidth=2)
    #     trans = transforms.blended_transform_factory(
    #         ax.get_yticklabels()[0].get_transform(), ax.transData)
    #     ax.text(0, mean, "{:.2f}".format(mean), color='red', transform=trans, ha='right', va='center')
    #
    #     # output
    #     wind_vectors_angles = self.wind_opt.output_angles
    #     mean = wind_vectors_angles.mean()
    #     measurements = np.arange(0, len(wind_vectors_angles), 1)
    #
    #     ax.bar(measurements, wind_vectors_angles, color='green')
    #     # add mean line
    #     ax.axhline(mean, color='red', linewidth=2)
    #     trans = transforms.blended_transform_factory(
    #         ax.get_yticklabels()[0].get_transform(), ax.transData)
    #     ax.text(0, mean, "{:.2f}".format(mean), color='black', transform=trans, ha='right', va='center')
    #
    #     ax.set_xlabel('Measurements')
    #     ax.set_ylabel('Angles between the wind vectors (deg)')
    #     # add mean line
    #     ax.axhline(mean, color='red', linewidth=2)
    #     trans = transforms.blended_transform_factory(
    #         ax.get_yticklabels()[0].get_transform(), ax.transData)
    #     ax.text(0, mean, "{:.2f}".format(mean), color='red', transform=trans, ha='right', va='center')
    #
    #     # make image full screen
    #     fig_manager = plt.get_current_fig_manager()
    #     fig_manager.window.maximize()
    #
        if self._save_output:
            self.pp.savefig(fig)
        else:
            plt.show()

    def plot_losses(self):
        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()

        samples_number_of_windows = []
        for i in range(len(self._losses_dict)):
            samples_number_of_windows.append(self._losses_dict[i]['Number of windows'])

        # create matrix of losses
        losses_start_position = list(self._losses_dict[0]).index('Zero wind loss mae') + 6
        num_of_losses = int((len(self._losses_dict[0])-losses_start_position)/2)
        num_of_samples = len(self._losses_dict)
        num_of_timesteps = max(samples_number_of_windows)

        mae_matrix = np.empty((num_of_losses, num_of_samples, num_of_timesteps))
        mae_matrix[:] = np.nan
        mse_matrix = np.empty((num_of_losses, num_of_samples, num_of_timesteps))
        mse_matrix[:] = np.nan
        for i in range(num_of_losses):
            for j in range(num_of_samples):
                for k in range(len(self._losses_dict[j]['Average wind loss mae'])):
                    mae_matrix[i][j][k] = list(self._losses_dict[j].values())[losses_start_position-4 + i*2][k]
                    mse_matrix[i][j][k] = list(self._losses_dict[j].values())[losses_start_position-4 + i*2+1][k]

        # --- Plots ---
        colors = ["#f15a24", "#feb306", "#0071bc", "#03a99d", "#8b5ca4", "#f15a24", "#feb306", "#0071bc", "#03a99d",
                  "#8b5ca4"]
        # mae_matrix = np.expand_dims(mae_matrix[:, 7], axis=2)
        # mse_matrix = np.expand_dims(mse_matrix[:, 7], axis=2)

        # --- Average across time across losses ---
        mean_across_time_mae = np.nanmean(mae_matrix, axis=1)
        mean_across_time_mse = np.nanmean(mse_matrix, axis=1)
        var_across_time_mae = np.nanvar(mae_matrix, axis=1)
        var_across_time_mse = np.nanvar(mse_matrix, axis=1)
        std_across_time_mae = np.sqrt(var_across_time_mae)
        std_across_time_mse = np.sqrt(var_across_time_mse)
        time = [60*i for i in range(mean_across_time_mae.shape[1])]
        for i in range(mean_across_time_mae.shape[0]):
            ax.plot(time, mean_across_time_mae[i, :], color=colors[i])
            ax.fill_between(time, mean_across_time_mae[i, :] + std_across_time_mae[i, :], mean_across_time_mae[i, :] - std_across_time_mae[i, :], facecolor=colors[i], alpha=0.5)
            # ax.boxplot(mean_across_time_mae[:, i], showmeans=True)
            # ax.errorbar(time, mean_across_time_mae[:, i], std_across_time_mae[:, i])
        ax.set_xlabel('Time')
        ax.set_ylabel('MAE')
        ax.legend(('AE', 'VAE', 'Average wind', 'Optimized corners spline'))

        # for i in range(mean_across_time_mse.shape[1]):
        #     ax2.plot(time, mean_across_time_mse[:, i])
        # ax2.set_xlabel('Time')
        # ax2.set_ylabel('MSE')
        # ax2.legend(('AE', 'VAE', 'Average wind', 'Optimized corners spline'))

        # --- Average across samples across losses ---
        # for i in range(mae_matrix.shape[1]):
        #     mae_matrix_sample = mae_matrix[:, i]
        #     mean_sample_mae = np.nanmean(mae_matrix, axis=0)
        #     print('Mean sample ' + str(i) + ' mae', mean_sample_mae)
        mean_across_samples_mae = np.nanmean(mae_matrix, axis=(1, 2))
        print('Mean across samples mae', mean_across_samples_mae)
        mean_across_samples_mse = np.nanmean(mse_matrix, axis=(1, 2))
        print('Mean across samples mse', mean_across_samples_mse)
        reshaped_mae = mae_matrix.swapaxes(0, 1).reshape(-1, mae_matrix.shape[0])
        # filtered_reshaped_mae = reshaped_mae[~np.isnan(reshaped_mae)]
        # filter nans
        mask = ~np.isnan(reshaped_mae)
        filtered_data = [d[m] for d, m in zip(reshaped_mae.T, mask.T)]
        ax2.boxplot(filtered_data, positions=np.arange(mae_matrix.shape[0])+1, showmeans=True)
        ax3.violinplot(filtered_data, positions=np.arange(mae_matrix.shape[0]) + 1, showmeans=True)
        ax2.set_xticks(np.arange(mae_matrix.shape[0]) + 1)
        ax2.set_xticklabels(['AE', 'VAE', 'Average wind', 'Optimized corners spline'])
        ax3.set_xticks(np.arange(mae_matrix.shape[0]) + 1)
        ax3.set_xticklabels(['AE', 'VAE', 'Average wind', 'Optimized corners spline'])
        # data_to_plot = np.random.rand(100, 5)
        # positions = np.arange(5) + 1
        # # matplotlib > 1.4
        # ax2.violinplot(data_to_plot, positions=positions, showmeans=True)

        # make image full screen
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.maximize()

        if self._save_output:
            self.pp.savefig(fig)
            self.pp.savefig(fig2)
            self.pp.savefig(fig3)
        else:
            plt.show()

    def plot_best_wind_estimate(self):
        # Plot best wind estimate
        fig, ax = plot_prediction_observations(self._wind_prediction, self.wind_opt.labels.to(self.device),
                                               self.wind_opt.terrain.network_terrain.squeeze(0),
                                               self._save_output, self._add_sparse_mask_row, self._masked_input)

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
        # self.plot_high_res_version()
        # self.plot_vertical_wind_profile()
        # self.plot_fft_analysis()
        # self.plot_trajectory_wind_vectors()
        # self.plot_wind_field()
        # self.plot_wind_vectors_angles()
        self.plot_losses()
        # self.plot_best_wind_estimate()

        if self._save_output:
            self.close()

