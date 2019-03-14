from .divergence import divergence
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.widgets import Slider, RadioButtons
import torch

# try importing mayavi
mayavi_available = False # currently mayavi is disabled
try:
    from mayavi import mlab
    from mayavi.tools.pipeline import streamline
    from tvtk.api import tvtk, write_data
except:
    mayavi_available = False
    print('ImportError: Mayavi not available, not plotting streamlines.')

class PlotUtils():
    '''
    Class providing the tools to plot the input and labels for the 2D and 3D case.
    '''
    def __init__(self, input, label, terrain, design, uncertainty_predicted = False, plot_divergence = False, plot_turbulence = False,
                 ds = None, title_fontsize = 20, label_fontsize = 15, tick_fontsize = 10, cmap=cm.jet, terrain_color='grey'):
        # Input is the prediction, label is CFD
        self.__axis = 'x-z'
        self.__n_slice = 0
        self.__uncertainty_predicted = uncertainty_predicted
        self.__plot_divergence = plot_divergence
        self.__plot_turbulence = plot_turbulence

        self.__title_fontsize = title_fontsize
        self.__label_fontsize = label_fontsize
        self.__tick_fontsize = tick_fontsize

        self.__button = None
        self.__slider = None
        self.__ax_slider = None
        if design == 1:
            self.__slider_location = [0.15, 0.025, 0.77, 0.04]
            self.__button_location = [0.05, 0.01, 0.05, 0.08]

        else:
            self.__slider_location = [0.15, 0.025, 0.77, 0.04]#[0.09, 0.02, 0.82, 0.04]
            self.__button_location = [0.05, 0.01, 0.05, 0.08]#[0.80, 0.16, 0.05, 0.10]

        self.__in_images = []
        self.__out_images = []
        self.__error_images = []
        self.__uncertainty_images = []

        if self.__plot_divergence and ds and (len(list(label.shape)) > 3):
            label = torch.cat([label, torch.tensor(divergence(label.squeeze()[:3], ds, terrain.squeeze())).unsqueeze(0)])
        else:
            self.__plot_divergence = False

        # Set the input to be a masked array so that we specify a terrain colour
        self.__input = np.ma.MaskedArray(np.zeros(input.shape))
        is_terrain = np.logical_not(terrain.cpu().numpy().astype(bool))
        for i, channel in enumerate(input.cpu()):
            self.__input[i] = np.ma.masked_where(is_terrain, channel)

        self.__label = np.ma.MaskedArray(np.zeros(label.shape))
        for i, channel in enumerate(label.cpu()):
            self.__label[i] = np.ma.masked_where(is_terrain, channel)
        self.__uncertainty = None

        self.__cmap = cmap
        self.__cmap.set_bad(terrain_color)

        num_channels = self.__input.shape[0]
        if uncertainty_predicted:
            self.__uncertainty = self.__input[int(num_channels/2):,:]

        if design == 1:
            if uncertainty_predicted:

                self.__error = self.__label - self.__input[:int(num_channels/2),:]
            else:
                self.__error = self.__label - self.__input
        else:
            self.__error = None # in this case the error plot will not be executed anyway

    def update_images(self):
        '''
        Updates the images according to the slice and axis which should be displayed. 
        '''
        if self.__axis == '  y-z':
            for i, im in enumerate(self.__in_images):
                im.set_data(self.__input[i, :, :, self.__n_slice])
                im.set_extent([0, self.__input.shape[2], 0, self.__input.shape[1]])

            for i, im in enumerate(self.__out_images):
                im.set_data(self.__label[i, :, :, self.__n_slice])
                im.set_extent([0, self.__label.shape[2], 0, self.__label.shape[1]])

            for i, im in enumerate(self.__error_images):
                im.set_data(self.__error[i, :, :, self.__n_slice])
                im.set_extent([0, self.__error.shape[2], 0, self.__error.shape[1]])

            for i, im in enumerate(self.__uncertainty_images):
                im.set_data(self.__uncertainty[i, :, :, self.__n_slice])
                im.set_extent([0, self.__uncertainty.shape[2], 0, self.__uncertainty.shape[1]])

        elif self.__axis == '  x-y':
            for i, im in enumerate(self.__in_images):
                im.set_data(self.__input[i, self.__n_slice, :, :])
                im.set_extent([0, self.__input.shape[3], 0, self.__input.shape[2]])

            for i, im in enumerate(self.__out_images):
                im.set_data(self.__label[i, self.__n_slice, :, :])
                im.set_extent([0, self.__label.shape[3], 0, self.__label.shape[2]])

            for i, im in enumerate(self.__error_images):
                im.set_data(self.__error[i, self.__n_slice, :, :])
                im.set_extent([0, self.__error.shape[3], 0, self.__error.shape[2]])

            for i, im in enumerate(self.__uncertainty_images):
                im.set_data(self.__uncertainty[i, self.__n_slice, :, :])
                im.set_extent([0, self.__uncertainty.shape[3], 0, self.__uncertainty.shape[2]])

        else:
            for i, im in enumerate(self.__in_images):
                im.set_data(self.__input[i, :, self.__n_slice, :])
                im.set_extent([0, self.__input.shape[3], 0, self.__input.shape[1]])

            for i, im in enumerate(self.__out_images):
                im.set_data(self.__label[i, :, self.__n_slice, :])
                im.set_extent([0, self.__label.shape[3], 0, self.__label.shape[1]])

            for i, im in enumerate(self.__error_images):
                im.set_data(self.__error[i, :, self.__n_slice, :])
                im.set_extent([0, self.__error.shape[3], 0, self.__error.shape[1]])

            for i, im in enumerate(self.__uncertainty_images):
                im.set_data(self.__uncertainty[i, :, self.__n_slice, :])
                im.set_extent([0, self.__uncertainty.shape[3], 0, self.__uncertainty.shape[1]])

        plt.draw()

    def slider_callback(self, val):
        '''
        Callback for the slider to change the slice to display.
        '''
        self.__n_slice = int(round(val))
        self.update_images()

    def radio_callback(self, label):
        '''
        Callback for the radio button to change the axis along which the slices are made.
        '''
        if label != self.__axis:
            if label == '  y-z':
                max_slice = self.__input.shape[3] - 1
            elif label == '  x-y':
                max_slice = self.__input.shape[1] - 1
            else:
                max_slice = self.__input.shape[2] - 1

            if self.__n_slice > max_slice:
                self.__n_slice = max_slice

            self.__ax_slider.remove()
            self.__ax_slider = plt.axes(self.__slider_location)
            self.__slider = Slider(self.__ax_slider, 'Slice', 0, max_slice, valinit=self.__n_slice, valfmt='%0.0f')
            self.__slider.on_changed(self.slider_callback)
        self.__axis = label
        self.update_images()

    def plot_sample(self):
        '''
        Creates the plots of a single sample according to the input and label data.
        '''
        if (len(list(self.__label.shape)) > 3):
            # 3D data
            fh_in, ah_in = plt.subplots(3, 3, figsize=(16,13))
            fh_in.patch.set_facecolor('white')

            # plot the input data
            self.__in_images.append(ah_in[2][0].imshow(self.__input[0,:,self.__n_slice,:], origin='lower', vmin=self.__input[0,:,:,:].min(), vmax=self.__input[0,:,:,:].max(), aspect = 'auto', cmap=self.__cmap)) #terrain
            self.__in_images.append(ah_in[0][0].imshow(self.__input[1,:,self.__n_slice,:], origin='lower', vmin=self.__label[0,:,:,:].min(), vmax=self.__label[0,:,:,:].max(), aspect = 'auto', cmap=self.__cmap)) #ux
            self.__in_images.append(ah_in[0][1].imshow(self.__input[2,:,self.__n_slice,:], origin='lower', vmin=self.__label[1,:,:,:].min(), vmax=self.__label[1,:,:,:].max(), aspect = 'auto', cmap=self.__cmap)) #uy
            self.__in_images.append(ah_in[0][2].imshow(self.__input[3,:,self.__n_slice,:], origin='lower', vmin=self.__label[2,:,:,:].min(), vmax=self.__label[2,:,:,:].max(), aspect = 'auto', cmap=self.__cmap)) #uz
            ah_in[2][0].set_title('Terrain', fontsize = self.__title_fontsize)
            ah_in[0][0].set_title('Input Vel X', fontsize = self.__title_fontsize)
            ah_in[0][1].set_title('Input Vel Y', fontsize = self.__title_fontsize)
            ah_in[0][2].set_title('Input Vel Z', fontsize = self.__title_fontsize)
            chbar = fh_in.colorbar(self.__in_images[1], ax=ah_in[0][0])
            plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
            chbar = fh_in.colorbar(self.__in_images[2], ax=ah_in[0][1])
            plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
            chbar = fh_in.colorbar(self.__in_images[3], ax=ah_in[0][2])
            plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
            chbar = fh_in.colorbar(self.__in_images[0], ax=ah_in[2][0])
            plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)

            # plot the label data
            self.__out_images.append(ah_in[1][0].imshow(self.__label[0,:,self.__n_slice,:], origin='lower', vmin=self.__label[0,:,:,:].min(), vmax=self.__label[0,:,:,:].max(), aspect = 'auto', cmap=self.__cmap)) #ux
            self.__out_images.append(ah_in[1][1].imshow(self.__label[1,:,self.__n_slice,:], origin='lower', vmin=self.__label[1,:,:,:].min(), vmax=self.__label[1,:,:,:].max(), aspect = 'auto', cmap=self.__cmap)) #uy
            self.__out_images.append(ah_in[1][2].imshow(self.__label[2,:,self.__n_slice,:], origin='lower', vmin=self.__label[2,:,:,:].min(), vmax=self.__label[2,:,:,:].max(), aspect = 'auto', cmap=self.__cmap)) #uz
            if self.__plot_turbulence:
                self.__out_images.append(ah_in[2][1].imshow(self.__label[3,:,self.__n_slice,:], origin='lower', vmin=self.__label[3,:,:,:].min(), vmax=self.__label[3,:,:,:].max(), aspect = 'auto', cmap=self.__cmap)) #turbulence viscosity
                chbar = fh_in.colorbar(self.__out_images[3], ax=ah_in[2][1])
                ah_in[2][1].set_title('Prediction Turbulence', fontsize = self.__title_fontsize)
                plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
                plt.setp(ah_in[2][1].get_xticklabels(), fontsize=self.__tick_fontsize)
                plt.setp(ah_in[2][1].get_yticklabels(), fontsize=self.__tick_fontsize)
                ah_in[2][1].set_xticks([])
                ah_in[2][1].set_yticks([])

            else:
                fh_in.delaxes(ah_in[2][1])

            if self.__plot_divergence:
                if self.__plot_turbulence:
                    idx_div = 4
                else:
                    idx_div = 3

                self.__out_images.append(ah_in[2][2].imshow(self.__label[idx_div,:,self.__n_slice,:], origin='lower', vmin=max(self.__label[idx_div,:,:,:].min(), -0.5), vmax=min(self.__label[idx_div,:,:,:].max(), 0.5), aspect = 'auto', cmap=self.__cmap)) #turbulence viscosity
                chbar = fh_in.colorbar(self.__out_images[-1], ax=ah_in[2][2])
                ah_in[2][2].set_title('Velocity Divergence', fontsize = self.__title_fontsize)
                plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
                plt.setp(ah_in[2][2].get_xticklabels(), fontsize=self.__tick_fontsize)
                plt.setp(ah_in[2][2].get_yticklabels(), fontsize=self.__tick_fontsize)
                ah_in[2][2].set_xticks([])
                ah_in[2][2].set_yticks([])
            else:
                fh_in.delaxes(ah_in[2][2])

            ah_in[1][0].set_title('CFD Vel X', fontsize = self.__title_fontsize)
            ah_in[1][1].set_title('CFD Vel Y', fontsize = self.__title_fontsize)
            ah_in[1][2].set_title('CFD Vel Z', fontsize = self.__title_fontsize)
            chbar = fh_in.colorbar(self.__out_images[0], ax=ah_in[1][0])
            plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
            chbar = fh_in.colorbar(self.__out_images[1], ax=ah_in[1][1])
            plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
            chbar = fh_in.colorbar(self.__out_images[2], ax=ah_in[1][2])
            plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)

            ah_in[0][0].set_xticks([])
            ah_in[0][0].set_yticks([])
            ah_in[0][1].set_xticks([])
            ah_in[0][1].set_yticks([])
            ah_in[0][2].set_xticks([])
            ah_in[0][2].set_yticks([])
            ah_in[1][0].set_xticks([])
            ah_in[1][0].set_yticks([])
            ah_in[1][1].set_xticks([])
            ah_in[1][1].set_yticks([])
            ah_in[1][2].set_xticks([])
            ah_in[1][2].set_yticks([])
            ah_in[2][0].set_xticks([])
            ah_in[2][0].set_yticks([])

            # create slider to select the slice
            self.__ax_slider = plt.axes(self.__slider_location)
            self.__slider = Slider(self.__ax_slider, 'Slice', 0, self.__input.shape[2]-1, valinit=self.__n_slice, valfmt='%0.0f')
            self.__slider.on_changed(self.slider_callback)

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.12)

            # create button to select the axis along which the slices are made
            rax = plt.axes(self.__button_location)
            self.__button = RadioButtons(rax, ('  x-z', '  x-y', '  y-z'), active=0)
            for circle in self.__button.circles:
                circle.set_radius(0.1)
            self.__button.on_clicked(self.radio_callback)
            plt.show(block=False)

            if mayavi_available:
                # 3D plots
                field = mlab.pipeline.vector_field(self.__label[2,:,:,:], self.__label[1,:,:,:], self.__label[0,:,:,:])
                terrain = mlab.pipeline.vector_field(self.__input[0,:,:,:], self.__input[0,:,:,:], self.__input[0,:,:,:])

                magnitude = mlab.pipeline.extract_vector_norm(field)
                magnitude_terrain = mlab.pipeline.extract_vector_norm(terrain)
                contours = mlab.pipeline.iso_surface(magnitude_terrain,
                                                    contours=[0.01, 0.8, 3.8, ],
                                                    transparent=False,
                                                    opacity=1.0,
                                                    colormap='YlGnBu',
                                                    vmin=0, vmax=0)

                field_lines = mlab.pipeline.streamline(magnitude, seedtype='plane',
                                                       seed_scale = 1,
                                                       seed_resolution = 200,
                                                        integration_direction='both',
                                                        colormap='bone')

                field_lines.seed.widget.enabled = False
                field_lines.seed.widget.point1 = [0, 0, 0]
                field_lines.seed.widget.point2 = [0, 10, 10]
                field_lines.seed.widget.resolution = 10

                mlab.show()
        else:
            print('Warning: The 2D plotting has not been used for some time, it might be not working')
            # 2D data
            fh_in, ah_in = plt.subplots(3, 2, figsize=(20,13))
            fh_in.patch.set_facecolor('white')

            h_ux_in = ah_in[0][0].imshow(self.__input[1,:,:], origin='lower', vmin=self.__label[0,:,:].min(), vmax=self.__label[0,:,:].max(), cmap=self.__cmap)
            h_uz_in = ah_in[0][1].imshow(self.__input[2,:,:], origin='lower', vmin=self.__label[1,:,:].min(), vmax=self.__label[1,:,:].max(), cmap=self.__cmap)
            ah_in[0][0].set_title('Input Vel X', fontsize = self.__title_fontsize)
            ah_in[0][1].set_title('Input Vel Z', fontsize = self.__title_fontsize)
            chbar = fh_in.colorbar(h_ux_in, ax=ah_in[0][0])
            chbar.set_label('[m/s]', fontsize = self.__label_fontsize)
            plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
            chbar = fh_in.colorbar(h_uz_in, ax=ah_in[0][1])
            chbar.set_label('[m/s]', fontsize = self.__label_fontsize)
            plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)

            h_ux_in = ah_in[1][0].imshow(self.__label[0,:,:], origin='lower', vmin=self.__label[0,:,:].min(), vmax=self.__label[0,:,:].max(), cmap=self.__cmap)
            h_uz_in = ah_in[1][1].imshow(self.__label[1,:,:], origin='lower', vmin=self.__label[1,:,:].min(), vmax=self.__label[1,:,:].max(), cmap=self.__cmap)
            ah_in[1][0].set_title('CFD Vel X', fontsize = self.__title_fontsize)
            ah_in[1][1].set_title('CFD Vel Z', fontsize = self.__title_fontsize)
            chbar = fh_in.colorbar(h_ux_in, ax=ah_in[1][0])
            chbar.set_label('[m/s]', fontsize = self.__label_fontsize)
            plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
            chbar = fh_in.colorbar(h_uz_in, ax=ah_in[1][1])
            chbar.set_label('[m/s]', fontsize = self.__label_fontsize)
            plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)

            h_ux_in = ah_in[2][0].imshow(self.__input[0,:,:], origin='lower', vmin=self.__input[0,:,:].min(), vmax=self.__input[0,:,:].max(), cmap=self.__cmap)
            try:
                h_uz_in = ah_in[2][1].imshow(self.__label[2,:,:], origin='lower', vmin=self.__label[2,:,:].min(), vmax=self.__label[2,:,:].max(), cmap=self.__cmap)
                ah_in[2][1].set_title('Turbulence viscosity label', fontsize = self.__title_fontsize)
                chbar = fh_in.colorbar(h_uz_in, ax=ah_in[2][1])
                chbar.set_label('[J/kg]', fontsize = self.__label_fontsize)
                plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
                plt.setp(ah_in[2][1].get_xticklabels(), fontsize=self.__tick_fontsize)
                plt.setp(ah_in[2][1].get_yticklabels(), fontsize=self.__tick_fontsize)
                ah_in[2][1].set_xlabel('x', fontsize=self.__label_fontsize)
                ah_in[2][1].set_ylabel('z', fontsize=self.__label_fontsize)

            except:
                fh_in.delaxes(ah_in[2][1])
                print('INFO: Turbulence viscosity not present as a label')

            ah_in[2][0].set_title('Terrain', fontsize = self.__title_fontsize)
            chbar = fh_in.colorbar(h_ux_in, ax=ah_in[2][0])
            chbar.set_label('-', fontsize = self.__label_fontsize)
            plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)

            plt.setp(ah_in[0][0].get_xticklabels(), fontsize=self.__tick_fontsize)
            plt.setp(ah_in[0][1].get_xticklabels(), fontsize=self.__tick_fontsize)
            plt.setp(ah_in[1][0].get_xticklabels(), fontsize=self.__tick_fontsize)
            plt.setp(ah_in[1][1].get_xticklabels(), fontsize=self.__tick_fontsize)
            plt.setp(ah_in[2][0].get_xticklabels(), fontsize=self.__tick_fontsize)
            plt.setp(ah_in[0][0].get_yticklabels(), fontsize=self.__tick_fontsize)
            plt.setp(ah_in[0][1].get_yticklabels(), fontsize=self.__tick_fontsize)
            plt.setp(ah_in[1][0].get_yticklabels(), fontsize=self.__tick_fontsize)
            plt.setp(ah_in[1][1].get_yticklabels(), fontsize=self.__tick_fontsize)
            plt.setp(ah_in[2][0].get_yticklabels(), fontsize=self.__tick_fontsize)
            ah_in[0][0].set_xlabel('x', fontsize=self.__label_fontsize)
            ah_in[0][1].set_xlabel('x', fontsize=self.__label_fontsize)
            ah_in[1][0].set_xlabel('x', fontsize=self.__label_fontsize)
            ah_in[1][1].set_xlabel('x', fontsize=self.__label_fontsize)
            ah_in[2][0].set_xlabel('x', fontsize=self.__label_fontsize)
            ah_in[0][0].set_ylabel('z', fontsize=self.__label_fontsize)
            ah_in[0][1].set_ylabel('z', fontsize=self.__label_fontsize)
            ah_in[1][0].set_ylabel('z', fontsize=self.__label_fontsize)
            ah_in[1][1].set_ylabel('z', fontsize=self.__label_fontsize)
            ah_in[2][0].set_ylabel('z', fontsize=self.__label_fontsize)
            plt.tight_layout()

        plt.show()

    def plot_prediction(self):
        if (len(self.__label.shape) > 3):
            turbulence_predicted = self.__label.shape[0] > 3

            if self.__uncertainty_predicted:
                if turbulence_predicted:
                    fh_in, ah_in = plt.subplots(4, self.__label.shape[0],figsize=(15,12))
                else:
                    fh_in, ah_in = plt.subplots(4, self.__label.shape[0],figsize=(12,12))
            else:
                if turbulence_predicted:
                    fh_in, ah_in = plt.subplots(3, self.__label.shape[0],figsize=(20,12))
                else:
                    fh_in, ah_in = plt.subplots(3, self.__label.shape[0],figsize=(15,12))

            fh_in.patch.set_facecolor('white')

            title = ['Vel X', 'Vel Y','Vel Z', 'Turbulence']
            units = ['[m/s]', '[m/s]', '[m/s]', '[J/kg]']
            for i in range(self.__label.shape[0]):
                self.__out_images.append(ah_in[0][i].imshow(self.__label[i,:,self.__n_slice,:], origin='lower', vmin=self.__label[i,:,:,:].min(), vmax=self.__label[i,:,:,:].max(), aspect = 'auto', cmap=self.__cmap))
                self.__in_images.append(ah_in[1][i].imshow(self.__input[i,:,self.__n_slice,:], origin='lower', vmin=self.__label[i,:,:,:].min(), vmax=self.__label[i,:,:,:].max(), aspect = 'auto', cmap=self.__cmap))
                self.__error_images.append(ah_in[2][i].imshow(self.__error[i,:,self.__n_slice,:], origin='lower', vmin=self.__error[i,:,:,:].min(), vmax=self.__error[i,:,:,:].max(), aspect = 'auto', cmap=self.__cmap))

                ah_in[0][i].set_title(title[i], fontsize = self.__title_fontsize)

                ah_in[0][i].set_xticks([])
                ah_in[0][i].set_yticks([])
                ah_in[1][i].set_xticks([])
                ah_in[1][i].set_yticks([])
                ah_in[2][i].set_xticks([])
                ah_in[2][i].set_yticks([])

                chbar = fh_in.colorbar(self.__out_images[i], ax=ah_in[0][i])
                plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
                chbar = fh_in.colorbar(self.__in_images[i], ax=ah_in[1][i])
                plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
                chbar = fh_in.colorbar(self.__error_images[i], ax=ah_in[2][i])
                plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)

                if self.__uncertainty_predicted:
                    self.__uncertainty_images.append(ah_in[3][i].imshow(self.__uncertainty[i,:,self.__n_slice,:], origin='lower', vmin=self.__uncertainty[i,:,:,:].min(), vmax=self.__uncertainty[i,:,:,:].max(), aspect = 'auto', cmap=self.__cmap))
                    chbar = fh_in.colorbar(self.__uncertainty_images[i], ax=ah_in[3][i])
                    plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
                    plt.setp(ah_in[3][i].get_xticklabels(), fontsize=self.__tick_fontsize)
                    plt.setp(ah_in[3][i].get_yticklabels(), fontsize=self.__tick_fontsize)

                    ah_in[3][i].set_xticks([])
                    ah_in[3][i].set_yticks([])

                if (i == 0):
                    ah_in[0][i].set_ylabel('CFD', fontsize=self.__label_fontsize)
                    ah_in[1][i].set_ylabel('Prediction', fontsize=self.__label_fontsize)
                    ah_in[2][i].set_ylabel('Error', fontsize=self.__label_fontsize)
                    if self.__uncertainty_predicted:
                        ah_in[3][i].set_ylabel('Uncertainty', fontsize=self.__label_fontsize)

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.12)

            # create slider to select the slice
            self.__ax_slider = plt.axes(self.__slider_location)
            self.__slider = Slider(self.__ax_slider, 'Slice', 0, self.__input.shape[2]-1, valinit=self.__n_slice, valfmt='%0.0f')
            self.__slider.on_changed(self.slider_callback)

            # create button to select the axis along which the slices are made
            rax = plt.axes(self.__button_location)
            self.__button = RadioButtons(rax, ('  x-z', '  x-y', '  y-z'), active=0)
            for circle in self.__button.circles:
                circle.set_radius(0.1)
            self.__button.on_clicked(self.radio_callback)

        else:
            print('Warning: The 2D plotting has not been used for some time, it might be not working')
            use_turbulence = self.__label.shape[0] > 2
            fh_in, ah_in = plt.subplots(3, self.__label.shape[0],figsize=(15,10))
            fh_in.patch.set_facecolor('white')

            title = ['Vel X', 'Vel Z', 'Turbulence']
            units = ['[m/s]', '[m/s]', '[J/kg]']
            for i in range(self.__label.shape[0]):
                self.__in_images.append(ah_in[0][i].imshow(self.__label[i,:,:], origin='lower', vmin=self.__label[i,:,:].min(), vmax=self.__label[i,:,:].max(), aspect = 'auto', cmap=self.__cmap))
                self.__out_images.append(ah_in[1][i].imshow(self.__input[i,:,:], origin='lower', vmin=self.__label[i,:,:].min(), vmax=self.__label[i,:,:].max(), aspect = 'auto', cmap=self.__cmap))
                self.__error_images.append(ah_in[2][i].imshow(self.__error[i,:,:], origin='lower', vmin=self.__error[i,:,:].min(), vmax=self.__error[i,:,:].max(), aspect = 'auto', cmap=self.__cmap))
                ah_in[0][i].set_title(title[i] + ' CFD', fontsize = self.__title_fontsize)
                ah_in[1][i].set_title(title[i] + ' Prediction', fontsize = self.__title_fontsize)
                ah_in[2][i].set_title(title[i] + ' Error', fontsize = self.__title_fontsize)
                plt.setp(ah_in[0][i].get_xticklabels(), fontsize=self.__tick_fontsize)
                plt.setp(ah_in[1][i].get_xticklabels(), fontsize=self.__tick_fontsize)
                plt.setp(ah_in[2][i].get_xticklabels(), fontsize=self.__tick_fontsize)
                plt.setp(ah_in[0][i].get_yticklabels(), fontsize=self.__tick_fontsize)
                plt.setp(ah_in[1][i].get_yticklabels(), fontsize=self.__tick_fontsize)
                plt.setp(ah_in[2][i].get_yticklabels(), fontsize=self.__tick_fontsize)
                chbar = fh_in.colorbar(self.__in_images[i], ax=ah_in[0][i])
                chbar.set_label(units[i], fontsize = self.__label_fontsize)
                plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
                chbar = fh_in.colorbar(self.__out_images[i], ax=ah_in[1][i])
                chbar.set_label(units[i], fontsize = self.__label_fontsize)
                plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
                chbar = fh_in.colorbar(self.__error_images[i], ax=ah_in[2][i])
                chbar.set_label(units[i], fontsize = self.__label_fontsize)
                plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
                ah_in[0][i].set_xlabel('x', fontsize=self.__label_fontsize)
                ah_in[1][i].set_xlabel('x', fontsize=self.__label_fontsize)
                ah_in[2][i].set_xlabel('x', fontsize=self.__label_fontsize)
                ah_in[0][i].set_ylabel('z', fontsize=self.__label_fontsize)
                ah_in[1][i].set_ylabel('z', fontsize=self.__label_fontsize)
                ah_in[2][i].set_ylabel('z', fontsize=self.__label_fontsize)
                plt.tight_layout()

        plt.show()

def plot_sample(input, label, terrain, plot_divergence = False, plot_turbulence = False, ds = None):
    '''
    Creates the plots according to the input and label data.
    Can handle 2D as well as 3D input. For the 3D input only slices are shown.
    The axes along which the slices are made as well as the location of the slice
    can be set using sliders and buttons in the figure.
    '''
    instance = PlotUtils(input, label, terrain, 0, False, plot_divergence, plot_turbulence, ds)
    instance.plot_sample()

def plot_prediction(output, label, terrain, uncertainty_predicted):
    instance = PlotUtils(output, label, terrain, 1, uncertainty_predicted)
    instance.plot_prediction()

def violin_plot(labels, data, xlabel, ylabel, ylim=None):
    index = np.arange(len(labels))

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')

    # need to manually set the factor and make sure that it is not too small, otherwise a numerical underflow will happen
    factor = np.power(len(data[0]), -1.0 / (len(data) + 4))
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False, points=300, bw_method=np.max([factor, 0.6]))

    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    quartile1 = []
    medians = []
    quartile3 = []
    for channel in data:
        quartile1_channel, medians_channel, quartile3_channel = np.percentile(channel, [25, 50, 75])
        quartile1.append(quartile1_channel)
        medians.append(medians_channel)
        quartile3.append(quartile3_channel)

    whiskers = np.array([adjacent_values(sorted(sorted_array), q1, q3) for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(inds)
    ax.set_xticklabels(labels)
    if ylim:
        ax.set_ylim(ylim)
    fig.tight_layout()


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value
