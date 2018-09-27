import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

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
    def __init__(self, input, label, design, uncertainty_predicted = False, title_fontsize = 30, label_fontsize = 23, tick_fontsize = 18):
        self.__axis = 'x-z'
        self.__n_slice = 0
        self.__uncertainty_predicted = uncertainty_predicted

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
            self.__slider_location = [0.09, 0.02, 0.82, 0.04]
            self.__button_location = [0.80, 0.16, 0.05, 0.10]

        self.__in_images = []
        self.__out_images = []
        self.__error_images = []

        self.__input = input
        self.__label = label
        if design == 1:
            if uncertainty_predicted:
                self.__error = label - input[:3,:]
            else:
                self.__error = label - input
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
                im.axes.set_xlabel('y', fontsize=self.__label_fontsize)
                im.axes.set_ylabel('z', fontsize=self.__label_fontsize)

            for i, im in enumerate(self.__out_images):
                im.set_data(self.__label[i, :, :, self.__n_slice])
                im.set_extent([0, self.__label.shape[2], 0, self.__label.shape[1]])
                im.axes.set_xlabel('y', fontsize=self.__label_fontsize)
                im.axes.set_ylabel('z', fontsize=self.__label_fontsize)

            for i, im in enumerate(self.__error_images):
                im.set_data(self.__error[i, :, :, self.__n_slice])
                im.set_extent([0, self.__error.shape[2], 0, self.__error.shape[1]])
                im.axes.set_xlabel('y', fontsize=self.__label_fontsize)
                im.axes.set_ylabel('z', fontsize=self.__label_fontsize)

        elif self.__axis == '  x-y':
            for i, im in enumerate(self.__in_images):
                im.set_data(self.__input[i, self.__n_slice, :, :])
                im.set_extent([0, self.__input.shape[3], 0, self.__input.shape[2]])
                im.axes.set_xlabel('x', fontsize=self.__label_fontsize)
                im.axes.set_ylabel('y', fontsize=self.__label_fontsize)

            for i, im in enumerate(self.__out_images):
                im.set_data(self.__label[i, self.__n_slice, :, :])
                im.set_extent([0, self.__label.shape[3], 0, self.__label.shape[2]])
                im.axes.set_xlabel('x', fontsize=self.__label_fontsize)
                im.axes.set_ylabel('y', fontsize=self.__label_fontsize)

            for i, im in enumerate(self.__error_images):
                im.set_data(self.__error[i, self.__n_slice, :, :])
                im.set_extent([0, self.__error.shape[3], 0, self.__error.shape[2]])
                im.axes.set_xlabel('x', fontsize=self.__label_fontsize)
                im.axes.set_ylabel('y', fontsize=self.__label_fontsize)

        else:
            for i, im in enumerate(self.__in_images):
                im.set_data(self.__input[i, :, self.__n_slice, :])
                im.set_extent([0, self.__input.shape[3], 0, self.__input.shape[1]])
                im.axes.set_xlabel('x', fontsize=self.__label_fontsize)
                im.axes.set_ylabel('z', fontsize=self.__label_fontsize)

            for i, im in enumerate(self.__out_images):
                im.set_data(self.__label[i, :, self.__n_slice, :])
                im.set_extent([0, self.__label.shape[3], 0, self.__label.shape[1]])
                im.axes.set_xlabel('x', fontsize=self.__label_fontsize)
                im.axes.set_ylabel('z', fontsize=self.__label_fontsize)

            for i, im in enumerate(self.__error_images):
                im.set_data(self.__error[i, :, self.__n_slice, :])
                im.set_extent([0, self.__error.shape[3], 0, self.__error.shape[1]])
                im.axes.set_xlabel('x', fontsize=self.__label_fontsize)
                im.axes.set_ylabel('z', fontsize=self.__label_fontsize)

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
        if (len(list(self.__label.size())) > 3):
            # 3D data
            fh_in, ah_in = plt.subplots(3, 3, figsize=(20,13))
            fh_in.patch.set_facecolor('white')
            fh_in.delaxes(ah_in[2][2])

            # plot the input data
            self.__in_images.append(ah_in[2][0].imshow(self.__input[0,:,self.__n_slice,:], origin='lower', vmin=self.__input[0,:,:,:].min(), vmax=self.__input[0,:,:,:].max(), aspect = 'auto')) #terrain
            self.__in_images.append(ah_in[0][0].imshow(self.__input[1,:,self.__n_slice,:], origin='lower', vmin=self.__label[0,:,:,:].min(), vmax=self.__label[0,:,:,:].max(), aspect = 'auto')) #ux
            self.__in_images.append(ah_in[0][1].imshow(self.__input[2,:,self.__n_slice,:], origin='lower', vmin=self.__label[1,:,:,:].min(), vmax=self.__label[1,:,:,:].max(), aspect = 'auto')) #uy
            self.__in_images.append(ah_in[0][2].imshow(self.__input[3,:,self.__n_slice,:], origin='lower', vmin=self.__label[2,:,:,:].min(), vmax=self.__label[2,:,:,:].max(), aspect = 'auto')) #uz
            ah_in[2][0].set_title('Terrain', fontsize = self.__title_fontsize)
            ah_in[0][0].set_title('Input Vel X', fontsize = self.__title_fontsize)
            ah_in[0][1].set_title('Input Vel Y', fontsize = self.__title_fontsize)
            ah_in[0][2].set_title('Input Vel Z', fontsize = self.__title_fontsize)
            chbar = fh_in.colorbar(self.__in_images[1], ax=ah_in[0][0])
            chbar.set_label('[m/s]', fontsize = self.__label_fontsize)
            plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
            chbar = fh_in.colorbar(self.__in_images[2], ax=ah_in[0][1])
            chbar.set_label('[m/s]', fontsize = self.__label_fontsize)
            plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
            chbar = fh_in.colorbar(self.__in_images[3], ax=ah_in[0][2])
            chbar.set_label('[m/s]', fontsize = self.__label_fontsize)
            plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
            chbar = fh_in.colorbar(self.__in_images[0], ax=ah_in[2][0])
            chbar.set_label('[-]', fontsize = self.__label_fontsize)
            plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)

            # plot the label data
            self.__out_images.append(ah_in[1][0].imshow(self.__label[0,:,self.__n_slice,:], origin='lower', vmin=self.__label[0,:,:,:].min(), vmax=self.__label[0,:,:,:].max(), aspect = 'auto')) #ux
            self.__out_images.append(ah_in[1][1].imshow(self.__label[1,:,self.__n_slice,:], origin='lower', vmin=self.__label[1,:,:,:].min(), vmax=self.__label[1,:,:,:].max(), aspect = 'auto')) #uy
            self.__out_images.append(ah_in[1][2].imshow(self.__label[2,:,self.__n_slice,:], origin='lower', vmin=self.__label[2,:,:,:].min(), vmax=self.__label[2,:,:,:].max(), aspect = 'auto')) #uz
            try:
                self.__out_images.append(ah_in[2][1].imshow(self.__label[3,:,self.__n_slice,:], origin='lower', vmin=self.__label[3,:,:,:].min(), vmax=self.__label[3,:,:,:].max(), aspect = 'auto')) #turbulence viscosity
                chbar = fh_in.colorbar(self.__out_images[3], ax=ah_in[2][1])
                ah_in[2][1].set_title('Prediction Turbulence', fontsize = self.__title_fontsize)
                chbar.set_label('[J/kg]', fontsize = self.__label_fontsize)
                plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
                plt.setp(ah_in[2][1].get_xticklabels(), fontsize=self.__tick_fontsize)
                plt.setp(ah_in[2][1].get_yticklabels(), fontsize=self.__tick_fontsize)

            except:
                print('INFO: Turbulence viscosity not present as a label')
                fh_in.delaxes(ah_in[2][1])

            ah_in[1][0].set_title('CFD Vel X', fontsize = self.__title_fontsize)
            ah_in[1][1].set_title('CFD Vel Y', fontsize = self.__title_fontsize)
            ah_in[1][2].set_title('CFD Vel Z', fontsize = self.__title_fontsize)
            chbar = fh_in.colorbar(self.__out_images[0], ax=ah_in[1][0])
            chbar.set_label('[m/s]', fontsize = self.__label_fontsize)
            plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
            chbar = fh_in.colorbar(self.__out_images[1], ax=ah_in[1][1])
            chbar.set_label('[m/s]', fontsize = self.__label_fontsize)
            plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
            chbar = fh_in.colorbar(self.__out_images[2], ax=ah_in[1][2])
            chbar.set_label('[m/s]', fontsize = self.__label_fontsize)
            plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)

            plt.setp(ah_in[0][0].get_xticklabels(), fontsize=self.__tick_fontsize)
            plt.setp(ah_in[0][1].get_xticklabels(), fontsize=self.__tick_fontsize)
            plt.setp(ah_in[0][2].get_xticklabels(), fontsize=self.__tick_fontsize)
            plt.setp(ah_in[1][0].get_xticklabels(), fontsize=self.__tick_fontsize)
            plt.setp(ah_in[1][1].get_xticklabels(), fontsize=self.__tick_fontsize)
            plt.setp(ah_in[1][2].get_xticklabels(), fontsize=self.__tick_fontsize)
            plt.setp(ah_in[2][0].get_xticklabels(), fontsize=self.__tick_fontsize)
            plt.setp(ah_in[0][0].get_yticklabels(), fontsize=self.__tick_fontsize)
            plt.setp(ah_in[0][1].get_yticklabels(), fontsize=self.__tick_fontsize)
            plt.setp(ah_in[0][2].get_yticklabels(), fontsize=self.__tick_fontsize)
            plt.setp(ah_in[1][0].get_yticklabels(), fontsize=self.__tick_fontsize)
            plt.setp(ah_in[1][1].get_yticklabels(), fontsize=self.__tick_fontsize)
            plt.setp(ah_in[1][2].get_yticklabels(), fontsize=self.__tick_fontsize)
            plt.setp(ah_in[2][0].get_yticklabels(), fontsize=self.__tick_fontsize)

            ah_in[0][0].set_xlabel('x', fontsize=self.__label_fontsize)
            ah_in[0][1].set_xlabel('x', fontsize=self.__label_fontsize)
            ah_in[0][2].set_xlabel('x', fontsize=self.__label_fontsize)
            ah_in[1][0].set_xlabel('x', fontsize=self.__label_fontsize)
            ah_in[1][1].set_xlabel('x', fontsize=self.__label_fontsize)
            ah_in[1][2].set_xlabel('x', fontsize=self.__label_fontsize)
            ah_in[2][0].set_xlabel('x', fontsize=self.__label_fontsize)
            ah_in[0][0].set_ylabel('z', fontsize=self.__label_fontsize)
            ah_in[0][1].set_ylabel('z', fontsize=self.__label_fontsize)
            ah_in[0][2].set_ylabel('z', fontsize=self.__label_fontsize)
            ah_in[1][0].set_ylabel('z', fontsize=self.__label_fontsize)
            ah_in[1][1].set_ylabel('z', fontsize=self.__label_fontsize)
            ah_in[1][2].set_ylabel('z', fontsize=self.__label_fontsize)
            ah_in[2][0].set_ylabel('z', fontsize=self.__label_fontsize)

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
            # 2D data
            fh_in, ah_in = plt.subplots(3, 2, figsize=(20,13))
            fh_in.patch.set_facecolor('white')

            h_ux_in = ah_in[0][0].imshow(self.__input[1,:,:], origin='lower', vmin=self.__label[0,:,:].min(), vmax=self.__label[0,:,:].max())
            h_uz_in = ah_in[0][1].imshow(self.__input[2,:,:], origin='lower', vmin=self.__label[1,:,:].min(), vmax=self.__label[1,:,:].max())
            ah_in[0][0].set_title('Input Vel X', fontsize = self.__title_fontsize)
            ah_in[0][1].set_title('Input Vel Z', fontsize = self.__title_fontsize)
            chbar = fh_in.colorbar(h_ux_in, ax=ah_in[0][0])
            chbar.set_label('[m/s]', fontsize = self.__label_fontsize)
            plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
            chbar = fh_in.colorbar(h_uz_in, ax=ah_in[0][1])
            chbar.set_label('[m/s]', fontsize = self.__label_fontsize)
            plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)

            h_ux_in = ah_in[1][0].imshow(self.__label[0,:,:], origin='lower', vmin=self.__label[0,:,:].min(), vmax=self.__label[0,:,:].max())
            h_uz_in = ah_in[1][1].imshow(self.__label[1,:,:], origin='lower', vmin=self.__label[1,:,:].min(), vmax=self.__label[1,:,:].max())
            ah_in[1][0].set_title('CFD Vel X', fontsize = self.__title_fontsize)
            ah_in[1][1].set_title('CFD Vel Z', fontsize = self.__title_fontsize)
            chbar = fh_in.colorbar(h_ux_in, ax=ah_in[1][0])
            chbar.set_label('[m/s]', fontsize = self.__label_fontsize)
            plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
            chbar = fh_in.colorbar(h_uz_in, ax=ah_in[1][1])
            chbar.set_label('[m/s]', fontsize = self.__label_fontsize)
            plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)

            h_ux_in = ah_in[2][0].imshow(self.__input[0,:,:], origin='lower', vmin=self.__input[0,:,:].min(), vmax=self.__input[0,:,:].max())
            try:
                h_uz_in = ah_in[2][1].imshow(self.__label[2,:,:], origin='lower', vmin=self.__label[2,:,:].min(), vmax=self.__label[2,:,:].max())
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
        if (len(list(self.__label.size())) > 3):
            if self.__uncertainty_predicted:
                fh_in, ah_in = plt.subplots(3, self.__input.shape[0],figsize=(20,12))
            else:
                fh_in, ah_in = plt.subplots(3, self.__input.shape[0],figsize=(15,12))
            fh_in.patch.set_facecolor('white')

            title = ['Vel X', 'Vel Y','Vel Z', 'Turbulence']
            units = ['[m/s]', '[m/s]', '[m/s]', '[J/kg]']
            for i in range(self.__label.shape[0]):
                self.__out_images.append(ah_in[0][i].imshow(self.__label[i,:,self.__n_slice,:], origin='lower', vmin=self.__label[i,:,:,:].min(), vmax=self.__label[i,:,:,:].max(), aspect = 'auto'))
                self.__in_images.append(ah_in[1][i].imshow(self.__input[i,:,self.__n_slice,:], origin='lower', vmin=self.__label[i,:,:,:].min(), vmax=self.__label[i,:,:,:].max(), aspect = 'auto'))
                self.__error_images.append(ah_in[2][i].imshow(self.__error[i,:,self.__n_slice,:], origin='lower', vmin=self.__error[i,:,:,:].min(), vmax=self.__error[i,:,:,:].max(), aspect = 'auto'))

                ah_in[0][i].set_title(title[i] + ' CFD', fontsize = self.__title_fontsize)
                ah_in[1][i].set_title(title[i] + ' Prediction', fontsize = self.__title_fontsize)
                ah_in[2][i].set_title(title[i] + ' Error', fontsize = self.__title_fontsize)
                plt.setp(ah_in[0][i].get_xticklabels(), fontsize=self.__tick_fontsize)
                plt.setp(ah_in[1][i].get_xticklabels(), fontsize=self.__tick_fontsize)
                plt.setp(ah_in[2][i].get_xticklabels(), fontsize=self.__tick_fontsize)
                plt.setp(ah_in[0][i].get_yticklabels(), fontsize=self.__tick_fontsize)
                plt.setp(ah_in[1][i].get_yticklabels(), fontsize=self.__tick_fontsize)
                plt.setp(ah_in[2][i].get_yticklabels(), fontsize=self.__tick_fontsize)

                chbar = fh_in.colorbar(self.__out_images[i], ax=ah_in[0][i])
                chbar.set_label(units[i], fontsize = self.__label_fontsize)
                plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
                chbar = fh_in.colorbar(self.__error_images[i], ax=ah_in[2][i])
                chbar.set_label(units[i], fontsize = self.__label_fontsize)
                plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
                chbar = fh_in.colorbar(self.__in_images[i], ax=ah_in[1][i])
                chbar.set_label(units[i], fontsize = self.__label_fontsize)
                plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)

                ah_in[0][i].set_xlabel('x', fontsize=self.__label_fontsize)
                ah_in[2][i].set_xlabel('x', fontsize=self.__label_fontsize)
                ah_in[0][i].set_ylabel('z', fontsize=self.__label_fontsize)
                ah_in[2][i].set_ylabel('z', fontsize=self.__label_fontsize)
                ah_in[1][i].set_ylabel('z', fontsize=self.__label_fontsize)
                ah_in[1][i].set_xlabel('x', fontsize=self.__label_fontsize)

            if self.__uncertainty_predicted:
                fh_in.delaxes(ah_in[2][3])
                fh_in.delaxes(ah_in[0][3])
                self.__in_images.append(ah_in[1][3].imshow(self.__input[3,:,self.__n_slice,:], origin='lower', vmin=self.__input[3,:,:,:].min(), vmax=self.__input[3,:,:,:].max(), aspect = 'auto'))
                ah_in[1][3].set_title('log(variance)', fontsize = self.__title_fontsize)
                plt.setp(ah_in[1][3].get_xticklabels(), fontsize=self.__tick_fontsize)
                plt.setp(ah_in[1][3].get_yticklabels(), fontsize=self.__tick_fontsize)
                chbar = fh_in.colorbar(self.__in_images[3], ax=ah_in[1][3])
                chbar.set_label('[-]', fontsize = self.__label_fontsize)
                plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
                ah_in[1][3].set_ylabel('z', fontsize=self.__label_fontsize)
                ah_in[1][3].set_xlabel('x', fontsize=self.__label_fontsize)

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
            use_turbulence = self.__label.shape[0] > 2
            fh_in, ah_in = plt.subplots(3, self.__label.shape[0],figsize=(15,10))
            fh_in.patch.set_facecolor('white')

            title = ['Vel X', 'Vel Z', 'Turbulence']
            units = ['[m/s]', '[m/s]', '[J/kg]']
            for i in range(self.__label.shape[0]):
                self.__in_images.append(ah_in[0][i].imshow(self.__label[i,:,:], origin='lower', vmin=self.__label[i,:,:].min(), vmax=self.__label[i,:,:].max(), aspect = 'auto'))
                self.__out_images.append(ah_in[1][i].imshow(self.__input[i,:,:], origin='lower', vmin=self.__label[i,:,:].min(), vmax=self.__label[i,:,:].max(), aspect = 'auto'))
                self.__error_images.append(ah_in[2][i].imshow(self.__error[i,:,:], origin='lower', vmin=self.__error[i,:,:].min(), vmax=self.__error[i,:,:].max(), aspect = 'auto'))
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

def plot_sample(input, label):
    '''
    Creates the plots according to the input and label data.
    Can handle 2D as well as 3D input. For the 3D input only slices are shown.
    The axes along which the slices are made as well as the location of the slice
    can be set using sliders and buttons in the figure.
    '''
    instance = PlotUtils(input, label, 0)
    instance.plot_sample()

def plot_prediction(output, label, uncertainty_predicted):
    instance = PlotUtils(output, label, 1, uncertainty_predicted)
    instance.plot_prediction()
