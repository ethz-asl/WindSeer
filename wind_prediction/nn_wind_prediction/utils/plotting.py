from .derivation import divergence
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
from matplotlib.widgets import Slider, RadioButtons
import torch
import math

class PlotUtils():
    '''
    Class providing the tools to plot 3D data.

    Params:
        channels_to_plot: Indicates which channels should be plotted, either 'all' or a list of the channels
        provided_input_channels: a list of the channels that were passed as an input to the network
        provided_prediction_channels: a list of the channels that were predicted
        input: 4D tensor with [channels, z, y, x]. Channels must be ordered as in provided_input_channels.
        prediction: 4D tensor with [channels, z, y, x]. Channels must be ordered as in provided_prediction_channels.
        label: 4D tensor with [channels, z, y, x]. Channels must be ordered as in provided_prediction_channels.
        measurements: 4D tensor with [channels, z, y, x]. Channels must be ordered as in provided_prediction_channels.
        uncertainty: 4D tensor with [channels, z, y, x]. Channels must be ordered as in provided_prediction_channels.
        terrain: 3D tensor of terrain data [z, y, x].
        measurements_mask: 3D tensor with [z, y, x].
        ds: grid size of the data, used for plotting the divergence field, if set and the data is available the divergency is computed and plotted
        title_dict: an optional title dict can be passed, if one would like to replace the default titles or plot new channels
        title_fontsize: font size of the title
        label_fontsize: font size of the label
        tick_fontsize: font size of the tick
        cmap: color map to use
        terrain_color: color of the terrain
        invalid_color: color of invalid pixels
    '''
    def __init__(self, channels_to_plot = 'all', provided_input_channels = None, provided_prediction_channels = None,
                 input = None, prediction = None, label = None, terrain = None, input_mask = None, measurements_variance = None,
                 uncertainty = None, measurements = None, measurements_mask = None, ds = None, title_dict = None,
                 title_fontsize = 14, label_fontsize = 12, tick_fontsize = 8, cmap=cm.jet, terrain_color='grey', invalid_color='white'):

        # Input is the prediction, label is CFD
        self.__axis = 'x-z'

        # set the default titles and titles
        default_title_dict = {'terrain': 'Distance field','ux': 'Velocity X [m/s]', 'uy': 'Velocity Y [m/s]',
                              'uz': 'Velocity Z [m/s]', 'turb': 'Turb. kin. energy [m^2/s^2]', 'p': 'Rho-norm. pressure [m^2/s^2]',
                              'epsilon': 'Dissipation [m^2/s^3]', 'nut': 'Turb. viscosity [m^2/s]', 'ux_in': ' Input Velocity X [m/s]',
                              'uy_in': 'Input Velocity Y [m/s]', 'uz_in': 'Input Velocity Z [m/s]', 'ux_cfd': ' CFD Velocity X [m/s]',
                              'uy_cfd': 'CFD Velocity Y [m/s]', 'uz_cfd': 'CFD Velocity Z [m/s]', 'turb_cfd': 'CFD Turb. kin. energy [m^2/s^2]',
                              'p_cfd': 'CFD Rho-norm. pressure [m^2/s^2]', 'epsilon_cfd': 'CFD Dissipation [m^2/s^3]',
                              'nut_cfd': 'CFD Turb. viscosity [m^2/s]', 'div': 'Divergence [1/s]', 'mask': 'Input Mask'}

        # check what can be plotted
        self.__domain_shape = None
        self.__plot_input = False
        if input is not None and provided_input_channels is not None:
            if len(input.shape) != 4:
                raise ValueError('The input tensor must have 4 dimension')

            if input.shape[0] != len(provided_input_channels):
                raise ValueError('PlotUtils: The number of provided input labels must be equal to the number of channels in the input tensor')

            if self.__domain_shape is None:
                self.__domain_shape = input.shape[1:]
            else:
                if self.__domain_shape != input.shape[1:]:
                    raise ValueError('The tensor sizes are inconsistent, the number of cells in x-y-z must be consistent in all tensors')

            self.__plot_input = True

        self.__plot_prediction = False
        if prediction is not None and provided_prediction_channels is not None:
            if len(prediction.shape) != 4:
                raise ValueError('The prediction tensor must have 4 dimension')

            if prediction.shape[0] != len(provided_prediction_channels):
                raise ValueError('PlotUtils: The number of provided prediction channels must be equal to the number of channels in the prediction tensor')

            if self.__domain_shape is None:
                self.__domain_shape = prediction.shape[1:]
            else:
                if self.__domain_shape != prediction.shape[1:]:
                    raise ValueError('The tensor sizes are inconsistent, the number of cells in x-y-z must be consistent in all tensors')

            self.__plot_prediction = True

        self.__plot_label = False
        if label is not None and provided_prediction_channels is not None:
            if len(label.shape) != 4:
                raise ValueError('The label tensor must have 4 dimension')

            if label.shape[0] != len(provided_prediction_channels):
                raise ValueError('PlotUtils: The number of provided prediction channels must be equal to the number of channels in the label tensor')

            if self.__domain_shape is None:
                self.__domain_shape = label.shape[1:]
            else:
                if self.__domain_shape != label.shape[1:]:
                    raise ValueError('The tensor sizes are inconsistent, the number of cells in x-y-z must be consistent in all tensors')

            self.__plot_label = True

        self.__plot_measurements = False
        if measurements is not None and provided_prediction_channels is not None:
            if len(measurements.shape) != 4:
                raise ValueError('The measurements tensor must have 4 dimension')

            if measurements.shape[0] != len(provided_prediction_channels):
                raise ValueError('PlotUtils: The number of provided prediction channels must be equal to the number of channels in the measurements tensor')

            if self.__domain_shape is None:
                self.__domain_shape = measurements.shape[1:]
            else:
                if self.__domain_shape != measurements.shape[1:]:
                    raise ValueError('The tensor sizes are inconsistent, the number of cells in x-y-z must be consistent in all tensors')

            self.__plot_measurements = True

        self.__plot_measurements_variance = False
        if measurements_variance is not None and provided_prediction_channels is not None:
            if len(measurements_variance.shape) != 4:
                raise ValueError('The measurements tensor must have 4 dimension')

            if measurements_variance.shape[0] != len(provided_prediction_channels):
                raise ValueError('PlotUtils: The number of provided prediction channels must be equal to the number of channels in the measurements variance tensor')

            if self.__domain_shape is None:
                self.__domain_shape = measurements_variance.shape[1:]
            else:
                if self.__domain_shape != measurements_variance.shape[1:]:
                    raise ValueError('The tensor sizes are inconsistent, the number of cells in x-y-z must be consistent in all tensors')

            self.__plot_measurements_variance = True

        self.__plot_error = False
        if label is not None and prediction is not None:
            if label.shape != prediction.shape:
                raise ValueError('PlotUtils: The shape of the label and prediction tensor have to be equal')

            self.__plot_error = True
            plot_prediction_error = True
        elif measurements is not None and prediction is not None:
            if measurements.shape != prediction.shape:
                raise ValueError('PlotUtils: The shape of the measurements and prediction tensor have to be equal')

            self.__plot_error = True
            plot_prediction_error = False

        self.__plot_uncertainty = False
        if uncertainty is not None:
            assert uncertainty.shape is 4, `The uncertainty tensor must have 4 dimension'

            assert uncertainty.shape == prediction.shape, 'PlotUtils: The shape of the uncertainty and prediction tensor have to be equal'

            if self.__domain_shape is None:
                self.__domain_shape = uncertainty.shape[1:]
            else:
                if self.__domain_shape != uncertainty.shape[1:]:
                    raise ValueError('The tensor sizes are inconsistent, the number of cells in x-y-z must be consistent in all tensors')

            self.__plot_uncertainty = True

        if terrain is not None:
            if len(terrain.shape) != 3:
                raise ValueError('The terrain tensor must have 3 dimension')

            if self.__domain_shape != terrain.shape:
                raise ValueError('The tensor sizes are inconsistent, the number of cells in x-y-z must be consistent in all tensors')

        if input_mask is not None:
            if len(input_mask.shape) != 3:
                raise ValueError('The input_mask tensor must have 3 dimension')

            if self.__domain_shape != input_mask.shape:
                raise ValueError('The tensor sizes are inconsistent, the number of cells in x-y-z must be consistent in all tensors')

        if measurements_mask is not None:
            if len(measurements_mask.shape) != 3:
                raise ValueError('The measurements_mask tensor must have 3 dimension')

            if self.__domain_shape != measurements_mask.shape:
                raise ValueError('The tensor sizes are inconsistent, the number of cells in x-y-z must be consistent in all tensors')

        if (not self.__plot_input and not self.__plot_label and not self.__plot_prediction and not self.__plot_uncertainty
            and not self.__plot_measurements and not self.__plot_measurements_variance):
            raise ValueError('PlotUtils: Data incomplete, cannot plot anything')

        # set the title dict and check that the titles for all channels are available
        self.__title_dict = default_title_dict
        if title_dict is not None:
            self.__title_dict.update(title_dict)

        if provided_input_channels is not None:
            for channel in provided_input_channels:
                if channel not in self.__title_dict:
                    raise ValueError('PlotUtils Error: Input label not in labels: \'{}\', '
                                     'not in label list {}'.format(channel, default_channels))

        if provided_prediction_channels is not None:
            for channel in provided_prediction_channels:
                if channel not in self.__title_dict:
                    raise ValueError('PlotUtils Error: Prediction label not in labels: \'{}\', '
                                     'not in label list {}'.format(channel, default_channels))

        # copy data to class variables
        self.__provided_input_channels = provided_input_channels
        self.__provided_prediction_channels = provided_prediction_channels

        # plot the divergence if ds is provided
        possible_velocity_channels = [['ux', 'uy', 'uz'],
                                      ['ux_cfd', 'uy_cfd', 'uz_cfd']]
        if ds is not None and terrain is not None:
            if self.__plot_input:
                vel_indices = None
                for vel_labels in possible_velocity_channels:
                    if all(elem in self.__provided_input_channels for elem in vel_labels):
                        vel_indices = torch.LongTensor([self.__provided_input_channels.index(channel) for channel in vel_labels]).to(input.device)

                if vel_indices is not None:
                    self.__provided_input_channels += ['div']

                    div = divergence(input.index_select(0,vel_indices).unsqueeze(0), ds, terrain.unsqueeze(0)).squeeze().unsqueeze(0)
                    input = torch.cat((input, div), 0)

            if self.__plot_prediction or self.__plot_label:
                vel_indices = None
                for vel_labels in possible_velocity_channels:
                    if all(elem in self.__provided_prediction_channels for elem in vel_labels):
                        vel_indices = torch.LongTensor([self.__provided_prediction_channels.index(channel) for channel in vel_labels]).to(input.device)

                if vel_indices is not None:
                    self.__provided_prediction_channels += ['div']

                    if self.__plot_prediction:
                        div = divergence(prediction.index_select(0,vel_indices).unsqueeze(0), ds, terrain.unsqueeze(0)).squeeze().unsqueeze(0)
                        prediction= torch.cat((prediction, div), 0)

                    if self.__plot_label:
                        div = divergence(label.index_select(0,vel_indices).unsqueeze(0), ds, terrain.unsqueeze(0)).squeeze().unsqueeze(0)
                        label= torch.cat((label, div), 0)

        # reduce the tensors to the data that should be plotted
        if channels_to_plot != 'all':
            if type(channels_to_plot) == list:
                # check if all channels are available
                provided_channels = []
                if self.__plot_input:
                    provided_channels += self.__provided_input_channels

                if self.__plot_label or self.__plot_prediction or self.__plot_uncertainty:
                    provided_channels += self.__provided_prediction_channels

                for channel in channels_to_plot:
                    if not (channel in provided_channels):
                        raise ValueError('PlotUtils Error: plotting the {} channel requested but it is not available'.format(channel))

                if self.__plot_input:
                    input_indices = [self.__provided_input_channels.index(channel) for channel in self.channels_to_plot if (channel in self.__provided_input_channels)]

                    input = torch.index_select(input, 0, torch.LongTensor(input_indices).to(input.device))
                    self.__provided_input_channels = [self.__provided_input_channels[i] for i in input_indices]

                if self.__plot_label or self.__plot_prediction or self.__plot_uncertainty:
                    pred_indices = [self.__provided_prediction_channels.index(channel) for channel in self.channels_to_plot if (channel in self.__provided_prediction_channels)]

                    if prediction is not None:
                        prediction = torch.index_select(prediction, 0, torch.LongTensor(pred_indices).to(input.device))

                    if label is not None:
                        label = torch.index_select(label, 0, torch.LongTensor(pred_indices).to(input.device))

                    if uncertainty is not None:
                        uncertainty = torch.index_select(uncertainty, 0, torch.LongTensor(pred_indices).to(input.device))

                    self.__provided_prediction_channels = [self.__provided_prediction_channels[i] for i in pred_indices]

            else:
                raise ValueError('PlotUtils Error: channels_to_plot needs to be either a list or the string: all')

        # create list of buttons and sliders for each figure
        self.__ax_sliders = []
        self.__sliders = []
        self.__buttons = []
        self.__n_slices = []

        # prealocate lists to the correct size
        self.__n_figures = 0
        if self.__plot_input:
            self.__n_figures += 1

        if self.__plot_prediction or self.__plot_label or self.__plot_error or self.__plot_measurements or self.__plot_measurements_variance or self.__plot_uncertainty:
            self.__n_figures += int(math.ceil(len(self.__provided_prediction_channels) / 4.0))

        for j in range(self.__n_figures):
            self.__ax_sliders += [None]
            self.__sliders += [None]
            self.__buttons += [None]
            self.__n_slices += [0]

        # get the fontsizes
        self.__title_fontsize = title_fontsize
        self.__label_fontsize = label_fontsize
        self.__tick_fontsize = tick_fontsize

        # choose design layout
        self.__slider_location = [0.15, 0.025, 0.77, 0.04]
        self.__button_location = [0.05, 0.01, 0.05, 0.08]

        # initialize the images to be displayed
        self.__images = []

        if self.__plot_prediction:
            prediction = prediction.cpu().numpy()

        if self.__plot_label:
            label = label.cpu().numpy()

        if self.__plot_uncertainty:
            uncertainty = uncertainty.cpu().numpy()

        # mask the input by the input mask if one is provided
        if input_mask is not None and self.__plot_input:
            input_tmp = np.ma.MaskedArray(np.zeros(input.shape))
            is_masked = np.logical_not(input_mask.cpu().numpy().astype(bool))
            for i, channel in enumerate(input.cpu()):
                if self.__provided_input_channels[i] != 'terrain':
                    input_tmp[i] = np.ma.masked_where(is_masked, channel)
                else:
                    if terrain is not None:
                        is_terrain = np.logical_not(terrain.cpu().numpy().astype(bool))
                        input_tmp[i] = np.ma.masked_where(is_terrain, channel)
                    else:
                        input_tmp[i] = channel

            input = input_tmp
        elif self.__plot_input:
            input = input.cpu().numpy()

        # mask the measurements by the mask if one is provided
        if measurements_mask is not None and self.__plot_measurements:
            meas_tmp = np.ma.MaskedArray(np.zeros(measurements.shape))
            is_masked = np.logical_not(measurements_mask.cpu().numpy().astype(bool))
            for i, channel in enumerate(measurements.cpu()):
                meas_tmp[i] = np.ma.masked_where(is_masked, channel)

            measurements = meas_tmp
        elif self.__plot_measurements:
            measurements = measurements.cpu().numpy()

        if measurements_mask is not None and self.__plot_measurements_variance:
            meas_var_tmp = np.ma.MaskedArray(np.zeros(measurements_variance.shape))
            is_masked = np.logical_not(measurements_mask.cpu().numpy().astype(bool))
            for i, channel in enumerate(measurements_variance.cpu()):
                meas_var_tmp[i] = np.ma.masked_where(is_masked, channel)

            measurements_variance = meas_var_tmp
        elif self.__plot_measurements_variance:
            measurements_variance = measurements_variance.cpu().numpy()

        # mask the data by the terrain if it is provided
        if terrain is not None:
            no_terrain = np.logical_not(np.logical_not(terrain.cpu().numpy().astype(bool)))
            self.__terrain_mask = np.ma.masked_where(no_terrain, no_terrain)

        else:
            self.__terrain_mask = None


        if self.__plot_input:
            self.__input = input;

        if self.__plot_prediction:
            self.__prediction = prediction

        if self.__plot_label:
            self.__label = label;

        if self.__plot_measurements:
            self.__measurements = measurements

        if self.__plot_measurements_variance:
            self.__measurements_variance = measurements_variance

        if self.__plot_error:
            if plot_prediction_error:
                self.__error = self.__label - self.__prediction
            else:
                self.__error = self.__measurements - self.__prediction

        if self.__plot_uncertainty:
            self.__uncertainty = uncertainty

        # if only the labels and the inputs are plotted combine them and plot in one figure
        if (self.__plot_input and self.__plot_label and not self.__plot_prediction and not self.__plot_measurements
                and not self.__plot_error and not self.__plot_uncertainty and not self.__plot_measurements_variance):
            self.__input = np.ma.concatenate((self.__input, self.__label), 0)
            self.__provided_input_channels += self.__provided_prediction_channels
            self.__provided_prediction_channels = None
            self.__label = None
            self.__plot_label = False

        # color map
        self.__cmap = cmap
        self.__cmap.set_bad(invalid_color)

        # terrain color map
        self.__cmap_terrain = colors.LinearSegmentedColormap.from_list('custom', colors.to_rgba_array([terrain_color, terrain_color]), 2) #cm.binary_r
        self.__cmap_terrain.set_bad('grey', 0.0)

        # the number of already open figures, used in slider and button callbacks
        self.__n_already_open_figures = 0

    def update_images(self):
        '''
        Updates the images according to the slice and axis which should be displayed. 
        '''
        j = plt.gcf().number - 1 - self.__n_already_open_figures
        slice_number = self.__n_slices[j]
        if self.__axis == '  y-z':
            for im in self.__images:
                im['image'].set_data(im['data'][:, :, slice_number])
                im['image'].set_extent([0, im['data'].shape[1], 0, im['data'].shape[0]])

        elif self.__axis == '  x-y':
            for im in self.__images:
                im['image'].set_data(im['data'][slice_number, :, :])
                im['image'].set_extent([0, im['data'].shape[2], 0, im['data'].shape[1]])

        else:
            for im in self.__images:
                im['image'].set_data(im['data'][:, slice_number, :])
                im['image'].set_extent([0, im['data'].shape[2], 0, im['data'].shape[0]])

        plt.draw()

    def slider_callback(self, val):
        '''
        Callback for the slider to change the slice to display.
        '''
        figure = plt.gcf().number - 1 - self.__n_already_open_figures
        self.__n_slices[figure] = int(round(val))
        self.update_images()

    def radio_callback(self, label):
        '''
        Callback for the radio button to change the axis along which the slices are made.
        '''
        figure = plt.gcf().number - 1 - self.__n_already_open_figures
        if label != self.__axis:
            if label == '  y-z':
                max_slice = self.__domain_shape[2] - 1
            elif label == '  x-y':
                max_slice = self.__domain_shape[0] - 1
            else:
                max_slice = self.__domain_shape[1] - 1

            if self.__n_slices[figure] > max_slice:
                self.__n_slices[figure] = max_slice

            self.__ax_sliders[figure].remove()
            self.__ax_sliders[figure] = plt.axes(self.__slider_location)
            self.__sliders[figure] = Slider(self.__ax_sliders[figure], 'Slice', 0, max_slice, valinit=self.__n_slices[figure], valfmt='%0.0f')
            self.__sliders[figure].on_changed(self.slider_callback)
        self.__axis = label
        self.update_images()

    def plot(self):
        '''
        Creates the plots according to the available data
        '''
        # get the number of already open figures, used in slider and button callbacks
        self.__n_already_open_figures = len(list(map(plt.figure, plt.get_fignums())))
        fig_idx = -1

        if self.__plot_input:
            n_columns = int(math.ceil(math.sqrt(len(self.__provided_input_channels))))
            n_rows = int(math.ceil(len(self.__provided_input_channels)/n_columns))

            fig_in, ah_in = plt.subplots(n_rows, n_columns, squeeze=False, figsize=(14.5,12))
            fig_in.patch.set_facecolor('white')
            fig_idx += 1
            slice = self.__n_slices[fig_idx]

            data_index = 0
            for j in range(n_rows):
                n_col = int(min(len(self.__provided_input_channels) - n_columns * (j), n_columns))
                for i in range(n_col):
                    im = {}
                    im['image'] =  ah_in[j][i].imshow(self.__input[data_index, :, slice, :], origin='lower',
                                                      vmin=self.__input[data_index, :, :, :].min(),
                                                      vmax=self.__input[data_index, :, :, :].max(), aspect='equal', cmap=self.__cmap)

                    chbar = fig_in.colorbar(im['image'], ax=ah_in[j][i])
                    plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)

                    im['data'] = self.__input[data_index]
                    self.__images.append(im)


                    if self.__terrain_mask is not None:
                        im = {}
                        im['image'] = ah_in[j][i].imshow(self.__terrain_mask[:, slice, :], cmap=self.__cmap_terrain, aspect = 'equal', origin='lower')
                        im['data'] = self.__terrain_mask
                        self.__images.append(im)

                    ah_in[j][i].set_title(self.__title_dict[self.__provided_input_channels[data_index]],
                                          fontsize=self.__title_fontsize)

                    ah_in[j][i].set_xticks([])
                    ah_in[j][i].set_yticks([])

                    data_index += 1

                # remove the extra empty figures
                if n_col<n_columns and n_rows>1:
                    for i in range(n_col, n_columns):
                        fig_in.delaxes(ah_in[j][i])

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.12)

            # create slider to select the slice
            self.__ax_sliders[fig_idx] = plt.axes(self.__slider_location)
            self.__sliders[fig_idx] = Slider(self.__ax_sliders[fig_idx], 'Slice', 0, self.__domain_shape[1] - 1,
                                       valinit=self.__n_slices[fig_idx], valfmt='%0.0f')
            self.__sliders[fig_idx].on_changed(self.slider_callback)

            # create button to select the axis along which the slices are made
            rax = plt.axes(self.__button_location)
            label = ('  x-z', '  x-y', '  y-z')
            self.__buttons[fig_idx] = RadioButtons(rax, label, active=0)
            for circle in self.__buttons[fig_idx].circles:
                circle.set_radius(0.1)
            self.__buttons[fig_idx].on_clicked(self.radio_callback)



        if self.__plot_prediction or self.__plot_label or self.__plot_error or self.__plot_uncertainty or self.__plot_measurements or self.__plot_measurements_variance:
            for j in range(int(math.ceil(len(self.__provided_prediction_channels) / 4.0))):
                # get the number of columns for this figure
                n_columns = int(min(len(self.__provided_prediction_channels) - 4 * j, 4))
                n_rows = self.__plot_prediction + self.__plot_label + self.__plot_error + self.__plot_uncertainty + self.__plot_measurements + self.__plot_measurements_variance

                # create the new figure
                fig_in, ah_in = plt.subplots(n_rows, n_columns, squeeze=False, figsize=(14.5,12))
                fig_in.patch.set_facecolor('white')
                fig_idx += 1
                slice = self.__n_slices[fig_idx]

                for i in range(n_columns):
                    data_index = i + j * 4
                    idx_row = 0

                    if self.__plot_label:
                        im = {}
                        im['image'] = ah_in[idx_row][i].imshow(
                            self.__label[data_index,:,slice,:], origin='lower',
                            vmin=np.nanmin(self.__label[data_index,:,:,:]),
                            vmax=np.nanmax(self.__label[data_index,:,:,:]),
                            aspect = 'equal', cmap=self.__cmap)

                        chbar = fig_in.colorbar(im['image'], ax=ah_in[idx_row][i])
                        plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)

                        im['data'] = self.__label[data_index]
                        self.__images.append(im)

                        if self.__terrain_mask is not None:
                            im = {}
                            im['image'] = ah_in[idx_row][i].imshow(self.__terrain_mask[:,slice,:], cmap=self.__cmap_terrain, aspect = 'equal', origin='lower')
                            im['data'] = self.__terrain_mask
                            self.__images.append(im)

                        ah_in[idx_row][0].set_ylabel('CFD', fontsize=self.__label_fontsize)

                        idx_row += 1

                    if self.__plot_prediction:
                        if self.__plot_label:
                            lim_data = self.__label[data_index,:,:,:]
                        else:
                            lim_data = self.__prediction[data_index,:,:,:]

                        im = {}
                        im['image'] = ah_in[idx_row][i].imshow(
                            self.__prediction[data_index,:,slice,:], origin='lower',
                            vmin=np.nanmin(lim_data),
                            vmax=np.nanmax(lim_data),
                            aspect = 'equal', cmap=self.__cmap)

                        chbar = fig_in.colorbar(im['image'], ax=ah_in[idx_row][i])
                        plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)

                        im['data'] = self.__prediction[data_index]
                        self.__images.append(im)

                        if self.__terrain_mask is not None:
                            im = {}
                            im['image'] = ah_in[idx_row][i].imshow(self.__terrain_mask[:,slice,:], cmap=self.__cmap_terrain, aspect = 'equal', origin='lower')
                            im['data'] = self.__terrain_mask
                            self.__images.append(im)

                        ah_in[idx_row][0].set_ylabel('Prediction', fontsize=self.__label_fontsize)

                        idx_row += 1

                    if self.__plot_error:
                        im = {}
                        im['image'] = ah_in[idx_row][i].imshow(
                            self.__error[data_index,:,slice,:], origin='lower',
                            vmin=np.nanmin(self.__error[data_index,:,:,:]),
                            vmax=np.nanmax(self.__error[data_index,:,:,:]),
                            aspect='equal', cmap=self.__cmap)

                        chbar = fig_in.colorbar(im['image'], ax=ah_in[idx_row][i])
                        plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)

                        im['data'] = self.__error[data_index]
                        self.__images.append(im)

                        if self.__terrain_mask is not None:
                            im = {}
                            im['image'] = ah_in[idx_row][i].imshow(self.__terrain_mask[:,slice,:], cmap=self.__cmap_terrain, aspect = 'equal', origin='lower')
                            im['data'] = self.__terrain_mask
                            self.__images.append(im)

                        ah_in[idx_row][0].set_ylabel('Error', fontsize=self.__label_fontsize)

                        idx_row += 1

                    if self.__plot_measurements:
                        im = {}
                        im['image'] = ah_in[idx_row][i].imshow(
                            self.__measurements[data_index,:,slice,:], origin='lower',
                            vmin=np.nanmin(self.__measurements[data_index,:,:,:]),
                            vmax=np.nanmax(self.__measurements[data_index,:,:,:]),
                            aspect='equal', cmap=self.__cmap)

                        chbar = fig_in.colorbar(im['image'], ax=ah_in[idx_row][i])
                        plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)

                        im['data'] = self.__measurements[data_index]
                        self.__images.append(im)

                        if self.__terrain_mask is not None:
                            im = {}
                            im['image'] = ah_in[idx_row][i].imshow(self.__terrain_mask[:,slice,:], cmap=self.__cmap_terrain, aspect = 'equal', origin='lower')
                            im['data'] = self.__terrain_mask
                            self.__images.append(im)

                        ah_in[idx_row][0].set_ylabel('Measurements', fontsize=self.__label_fontsize)

                        idx_row += 1

                    if self.__plot_measurements_variance:
                        im = {}
                        im['image'] = ah_in[idx_row][i].imshow(
                            self.__measurements_variance[data_index,:,slice,:], origin='lower',
                            vmin=np.nanmin(self.__measurements_variance[data_index,:,:,:]),
                            vmax=np.nanmax(self.__measurements_variance[data_index,:,:,:]),
                            aspect='equal', cmap=self.__cmap)

                        chbar = fig_in.colorbar(im['image'], ax=ah_in[idx_row][i])
                        plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)

                        im['data'] = self.__measurements_variance[data_index]
                        self.__images.append(im)

                        if self.__terrain_mask is not None:
                            im = {}
                            im['image'] = ah_in[idx_row][i].imshow(self.__terrain_mask[:,slice,:], cmap=self.__cmap_terrain, aspect = 'equal', origin='lower')
                            im['data'] = self.__terrain_mask
                            self.__images.append(im)

                        ah_in[idx_row][0].set_ylabel('Measurements Variance', fontsize=self.__label_fontsize)

                        idx_row += 1

                    if self.__plot_uncertainty:
                        im = {}
                        im['image'] = ah_in[idx_row][i].imshow(
                            self.__uncertainty[data_index,:,slice,:], origin='lower',
                            vmin=np.nanmin(self.__uncertainty[data_index,:,:,:]),
                            vmax=np.nanmax(self.__uncertainty[data_index,:,:,:]),
                            aspect='equal', cmap=self.__cmap)

                        chbar = fig_in.colorbar(im['image'], ax=ah_in[idx_row][i])
                        plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)

                        im['data'] = self.__uncertainty[data_index]
                        self.__images.append(im)

                        if self.__terrain_mask is not None:
                            im = {}
                            im['image'] = ah_in[idx_row][i].imshow(self.__terrain_mask[:,slice,:], cmap=self.__cmap_terrain, aspect = 'equal', origin='lower')
                            im['data'] = self.__terrain_mask
                            self.__images.append(im)

                        ah_in[idx_row][0].set_ylabel('Uncertainty', fontsize=self.__label_fontsize)

                        idx_row += 1

                    ah_in[0][i].set_title(self.__title_dict[self.__provided_prediction_channels[data_index]], fontsize = self.__title_fontsize)

                    for iter in range(idx_row):
                        ah_in[iter][i].set_xticks([])
                        ah_in[iter][i].set_yticks([])

                plt.tight_layout()
                plt.subplots_adjust(bottom=0.12)

                # create slider to select the slice
                self.__ax_sliders[fig_idx] = plt.axes(self.__slider_location)
                self.__sliders[fig_idx] = Slider(self.__ax_sliders[fig_idx], 'Slice', 0, self.__domain_shape[1]-1, valinit=slice, valfmt='%0.0f')
                self.__sliders[fig_idx].on_changed(self.slider_callback)

                # create button to select the axis along which the slices are made
                rax = plt.axes(self.__button_location)
                label = ('  x-z', '  x-y', '  y-z')
                self.__buttons[fig_idx] = RadioButtons(rax, label, active=0)
                for circle in self.__buttons[fig_idx].circles:
                    circle.set_radius(0.1)
                self.__buttons[fig_idx].on_clicked(self.radio_callback)

        plt.show()

def plot_sample(provided_input_channels, input, provided_label_channels, label, channels_to_plot = 'all', input_mask = None, ds = None, title_dict = None):
    '''
    Creates the plots according to the input and label data.
    The axes along which the slices are made as well as the location of the slice
    can be set using sliders and buttons in the figure.
    '''
    if 'terrain' in provided_input_channels:
        terrain = input[provided_input_channels.index('terrain')].squeeze()
    else:
        terrain = None

    instance = PlotUtils(provided_input_channels = provided_input_channels,
                         provided_prediction_channels = provided_label_channels,
                         channels_to_plot = channels_to_plot,
                         input = input,
                         label = label,
                         terrain = terrain,
                         ds = ds,
                         input_mask = input_mask,
                         title_dict = title_dict)
    instance.plot()

def plot_prediction(provided_prediction_channels, prediction = None, label = None, uncertainty = None,
                    provided_input_channels = None, input = None, terrain = None,
                    measurements = None, measurements_mask = None,
                    ds = None, title_dict = None):
    '''
    Creates the plots according to the data provided.
    The axes along which the slices are made as well as the location of the slice
    can be set using sliders and buttons in the figure.
    '''

    input_mask = None
    if provided_input_channels is not None and input is not None:
        if 'mask' in provided_input_channels:
            idx = provided_input_channels.index('mask')
            input_mask = input[idx]

    instance = PlotUtils(provided_prediction_channels = provided_prediction_channels,
                         provided_input_channels = provided_input_channels, input_mask = input_mask,
                         prediction = prediction, input = input, label = label, uncertainty = uncertainty,
                         measurements = measurements, measurements_mask = measurements_mask,
                         terrain = terrain, ds = ds, title_dict = title_dict)
    instance.plot()

def plot_measurement(provided_measurement_channels, measurements, measurements_mask,
                     terrain = None, variance = None, prediction = None, title_dict = None):
    '''
    Create plots of the measurements and the respective variance
    '''
    instance = PlotUtils(provided_prediction_channels = provided_measurement_channels,
                         measurements = measurements,
                         measurements_mask = measurements_mask,
                         measurements_variance = variance,
                         terrain = terrain,
                         prediction = prediction,
                         title_dict = title_dict)
    instance.plot()

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
