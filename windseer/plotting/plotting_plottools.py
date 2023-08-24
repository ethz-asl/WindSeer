from windseer.utils import divergence
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
    '''

    def __init__(
            self,
            channels_to_plot='all',
            provided_input_channels=None,
            provided_prediction_channels=None,
            input=None,
            prediction=None,
            label=None,
            terrain=None,
            input_mask=None,
            measurements_variance=None,
            uncertainty=None,
            measurements=None,
            measurements_mask=None,
            ds=None,
            title_dict=None,
            title_fontsize=14,
            label_fontsize=12,
            tick_fontsize=8,
            cmap=cm.jet,
            terrain_color='grey',
            invalid_color='white',
            blocking=True
        ):
        '''
        Initializer
        
        Parameters
        ----------
        channels_to_plot : str or list, default: all
            Indicates which channels should be plotted, either 'all' or a list of the channels
        provided_input_channels : list or None, default: None
            List of the input channels to the network
        provided_prediction_channels : list or None, default: None
            List of the channels of the prediction tensor
        input : torch.Tensor or None, default: None
            4D input tensor with [channels, z, y, x]. Channels must be ordered as in provided_input_channels.
        prediction : torch.Tensor or None, default: None
            4D prediction tensor with [channels, z, y, x]. Channels must be ordered as in provided_prediction_channels.
        label : torch.Tensor or None, default: None
            4D label tensor with [channels, z, y, x]. Channels must be ordered as in provided_prediction_channels.
        terrain : torch.Tensor or None, default: None
            3D terrain tensor of terrain data [z, y, x].
        input_mask : torch.Tensor or None, default: None
            3D terrain tensor of input mask data indicating known cells [z, y, x].
        measurements_variance : torch.Tensor or None, default: None
            4D measurement variance tensor with [channels, z, y, x]. Channels must be ordered as in provided_prediction_channels.
        measurements : torch.Tensor or None, default: None
            4D measurement tensor with [channels, z, y, x]. Channels must be ordered as in provided_prediction_channels.
        uncertainty : torch.Tensor or None, default: None
            4D uncertainty tensor with [channels, z, y, x]. Channels must be ordered as in provided_prediction_channels.
        measurements_mask : torch,Tensor or None, default: None
            3D tensor with [z, y, x].
        ds : list or None, default: None
            Cell size of the data grid, used for plotting the divergence field, if set and the data is available the divergency is computed and plotted
        title_dict :  dict or None, default: None
            An optional title dict can be passed, if one would like to replace the default titles or plot new channels
        title_fontsize : int, default: 14
            Font size of the title
        label_fontsize : int, default: 12
            Font size of the label
        tick_fontsize : int, default: 8
            Font size of the tick
        cmap : cmap, default: jet
            Color map to use
        terrain_color : str, default: grey
            Color of the terrain
        invalid_color : str, default: white
            Color of invalid pixels (no measurements available for the sparse input)
        blocking : bool, default: True
            Indicates if the plot call is blocking by calling plt.show()
        '''

        # Input is the prediction, label is CFD
        self._axis = 'x-z'

        # set the default titles and titles
        default_title_dict = {
            'terrain': 'Distance field',
            'ux': 'Velocity X [m/s]',
            'uy': 'Velocity Y [m/s]',
            'uz': 'Velocity Z [m/s]',
            'turb': 'Turb. kin. energy [m2/s2]',
            'p': 'Rho-norm. pressure [m2/s2]',
            'epsilon': 'Dissipation [m2/s3]',
            'nut': 'Turb. viscosity [m2/s]',
            'ux_in': ' Input Velocity X [m/s]',
            'uy_in': 'Input Velocity Y [m/s]',
            'uz_in': 'Input Velocity Z [m/s]',
            'ux_cfd': ' CFD Velocity X [m/s]',
            'uy_cfd': 'CFD Velocity Y [m/s]',
            'uz_cfd': 'CFD Velocity Z [m/s]',
            'turb_cfd': 'CFD Turb. kin. energy [m2/s2]',
            'p_cfd': 'CFD Rho-norm. pressure [m2/s2]',
            'epsilon_cfd': 'CFD Dissipation [m2/s3]',
            'nut_cfd': 'CFD Turb. viscosity [m2/s]',
            'div': 'Divergence [1/s]',
            'mask': 'Input Mask'
            }

        # check what can be plotted
        self._domain_shape = None
        self._plot_input = False
        if input is not None and provided_input_channels is not None:
            if len(input.shape) != 4:
                raise ValueError('The input tensor must have 4 dimension')

            if input.shape[0] != len(provided_input_channels):
                raise ValueError(
                    'PlotUtils: The number of provided input labels must be equal to the number of channels in the input tensor'
                    )

            if self._domain_shape is None:
                self._domain_shape = input.shape[1:]
            else:
                if self._domain_shape != input.shape[1:]:
                    raise ValueError(
                        'The tensor sizes are inconsistent, the number of cells in x-y-z must be consistent in all tensors'
                        )

            self._plot_input = True

        self._plot_prediction = False
        if prediction is not None and provided_prediction_channels is not None:
            if len(prediction.shape) != 4:
                raise ValueError('The prediction tensor must have 4 dimension')

            if prediction.shape[0] != len(provided_prediction_channels):
                raise ValueError(
                    'PlotUtils: The number of provided prediction channels must be equal to the number of channels in the prediction tensor'
                    )

            if self._domain_shape is None:
                self._domain_shape = prediction.shape[1:]
            else:
                if self._domain_shape != prediction.shape[1:]:
                    raise ValueError(
                        'The tensor sizes are inconsistent, the number of cells in x-y-z must be consistent in all tensors'
                        )

            self._plot_prediction = True

        self._plot_label = False
        if label is not None and provided_prediction_channels is not None:
            if len(label.shape) != 4:
                raise ValueError('The label tensor must have 4 dimension')

            if label.shape[0] != len(provided_prediction_channels):
                raise ValueError(
                    'PlotUtils: The number of provided prediction channels must be equal to the number of channels in the label tensor'
                    )

            if self._domain_shape is None:
                self._domain_shape = label.shape[1:]
            else:
                if self._domain_shape != label.shape[1:]:
                    raise ValueError(
                        'The tensor sizes are inconsistent, the number of cells in x-y-z must be consistent in all tensors'
                        )

            self._plot_label = True

        self._plot_measurements = False
        if measurements is not None and provided_prediction_channels is not None:
            if len(measurements.shape) != 4:
                raise ValueError('The measurements tensor must have 4 dimension')

            if measurements.shape[0] != len(provided_prediction_channels):
                raise ValueError(
                    'PlotUtils: The number of provided prediction channels must be equal to the number of channels in the measurements tensor'
                    )

            if self._domain_shape is None:
                self._domain_shape = measurements.shape[1:]
            else:
                if self._domain_shape != measurements.shape[1:]:
                    raise ValueError(
                        'The tensor sizes are inconsistent, the number of cells in x-y-z must be consistent in all tensors'
                        )

            self._plot_measurements = True

        self._plot_measurements_variance = False
        if measurements_variance is not None and provided_prediction_channels is not None:
            if len(measurements_variance.shape) != 4:
                raise ValueError('The measurements tensor must have 4 dimension')

            if measurements_variance.shape[0] != len(provided_prediction_channels):
                raise ValueError(
                    'PlotUtils: The number of provided prediction channels must be equal to the number of channels in the measurements variance tensor'
                    )

            if self._domain_shape is None:
                self._domain_shape = measurements_variance.shape[1:]
            else:
                if self._domain_shape != measurements_variance.shape[1:]:
                    raise ValueError(
                        'The tensor sizes are inconsistent, the number of cells in x-y-z must be consistent in all tensors'
                        )

            self._plot_measurements_variance = True

        self._plot_error = False
        if label is not None and prediction is not None:
            if label.shape != prediction.shape:
                raise ValueError(
                    'PlotUtils: The shape of the label and prediction tensor have to be equal'
                    )

            self._plot_error = True
            plot_prediction_error = True
        elif measurements is not None and prediction is not None:
            if measurements.shape != prediction.shape:
                raise ValueError(
                    'PlotUtils: The shape of the measurements and prediction tensor have to be equal'
                    )

            self._plot_error = True
            plot_prediction_error = False

        self._plot_uncertainty = False
        if uncertainty is not None:
            assert len(
                uncertainty.shape
                ) == 4, 'The uncertainty tensor must have 4 dimension'

            assert uncertainty.shape == prediction.shape, 'PlotUtils: The shape of the uncertainty and prediction tensor have to be equal'

            if self._domain_shape is None:
                self._domain_shape = uncertainty.shape[1:]
            else:
                if self._domain_shape != uncertainty.shape[1:]:
                    raise ValueError(
                        'The tensor sizes are inconsistent, the number of cells in x-y-z must be consistent in all tensors'
                        )

            self._plot_uncertainty = True

        if terrain is not None:
            if len(terrain.shape) != 3:
                raise ValueError('The terrain tensor must have 3 dimension')

            if self._domain_shape != terrain.shape:
                raise ValueError(
                    'The tensor sizes are inconsistent, the number of cells in x-y-z must be consistent in all tensors'
                    )

        if input_mask is not None:
            if len(input_mask.shape) != 3:
                raise ValueError('The input_mask tensor must have 3 dimension')

            if self._domain_shape != input_mask.shape:
                raise ValueError(
                    'The tensor sizes are inconsistent, the number of cells in x-y-z must be consistent in all tensors'
                    )

        if measurements_mask is not None:
            if len(measurements_mask.shape) != 3:
                raise ValueError('The measurements_mask tensor must have 3 dimension')

            if self._domain_shape != measurements_mask.shape:
                raise ValueError(
                    'The tensor sizes are inconsistent, the number of cells in x-y-z must be consistent in all tensors'
                    )

        if (
            not self._plot_input and not self._plot_label and
            not self._plot_prediction and not self._plot_uncertainty and
            not self._plot_measurements and not self._plot_measurements_variance
            ):
            raise ValueError('PlotUtils: Data incomplete, cannot plot anything')

        # set the title dict and check that the titles for all channels are available
        self._title_dict = default_title_dict
        if title_dict is not None:
            self._title_dict.update(title_dict)

        if provided_input_channels is not None:
            for channel in provided_input_channels:
                if channel not in self._title_dict:
                    raise ValueError(
                        'PlotUtils Error: Input label not in labels: \'{}\', '
                        'not in label list {}'.format(channel, default_channels)
                        )

        if provided_prediction_channels is not None:
            for channel in provided_prediction_channels:
                if channel not in self._title_dict:
                    raise ValueError(
                        'PlotUtils Error: Prediction label not in labels: \'{}\', '
                        'not in label list {}'.format(channel, default_channels)
                        )

        # copy data to class variables
        self._provided_input_channels = provided_input_channels
        self._provided_prediction_channels = provided_prediction_channels

        # plot the divergence if ds is provided
        possible_velocity_channels = [['ux', 'uy', 'uz'],
                                      ['ux_cfd', 'uy_cfd', 'uz_cfd']]
        if ds is not None and terrain is not None:
            if self._plot_input:
                vel_indices = None
                for vel_labels in possible_velocity_channels:
                    if all(
                        elem in self._provided_input_channels for elem in vel_labels
                        ):
                        vel_indices = torch.LongTensor([
                            self._provided_input_channels.index(channel)
                            for channel in vel_labels
                            ]).to(input.device)

                if vel_indices is not None:
                    self._provided_input_channels += ['div']

                    div = divergence(
                        input.index_select(0, vel_indices).unsqueeze(0), ds,
                        terrain.unsqueeze(0)
                        ).squeeze().unsqueeze(0)
                    input = torch.cat((input, div), 0)

            if self._plot_prediction or self._plot_label:
                vel_indices = None
                for vel_labels in possible_velocity_channels:
                    if all(
                        elem in self._provided_prediction_channels
                        for elem in vel_labels
                        ):
                        vel_indices = torch.LongTensor([
                            self._provided_prediction_channels.index(channel)
                            for channel in vel_labels
                            ]).to(input.device)

                if vel_indices is not None:
                    self._provided_prediction_channels += ['div']

                    if self._plot_prediction:
                        div = divergence(
                            prediction.index_select(0, vel_indices).unsqueeze(0), ds,
                            terrain.unsqueeze(0)
                            ).squeeze().unsqueeze(0)
                        prediction = torch.cat((prediction, div), 0)

                    if self._plot_label:
                        div = divergence(
                            label.index_select(0, vel_indices).unsqueeze(0), ds,
                            terrain.unsqueeze(0)
                            ).squeeze().unsqueeze(0)
                        label = torch.cat((label, div), 0)

        # reduce the tensors to the data that should be plotted
        if channels_to_plot != 'all':
            if type(channels_to_plot) == list:
                # check if all channels are available
                provided_channels = []
                if self._plot_input:
                    provided_channels += self._provided_input_channels

                if self._plot_label or self._plot_prediction or self._plot_uncertainty:
                    provided_channels += self._provided_prediction_channels

                for channel in channels_to_plot:
                    if not (channel in provided_channels):
                        raise ValueError(
                            'PlotUtils Error: plotting the {} channel requested but it is not available'
                            .format(channel)
                            )

                if self._plot_input:
                    input_indices = [
                        self._provided_input_channels.index(channel)
                        for channel in self.channels_to_plot
                        if (channel in self._provided_input_channels)
                        ]

                    input = torch.index_select(
                        input, 0,
                        torch.LongTensor(input_indices).to(input.device)
                        )
                    self._provided_input_channels = [
                        self._provided_input_channels[i] for i in input_indices
                        ]

                if self._plot_label or self._plot_prediction or self._plot_uncertainty:
                    pred_indices = [
                        self._provided_prediction_channels.index(channel)
                        for channel in self.channels_to_plot
                        if (channel in self._provided_prediction_channels)
                        ]

                    if prediction is not None:
                        prediction = torch.index_select(
                            prediction, 0,
                            torch.LongTensor(pred_indices).to(input.device)
                            )

                    if label is not None:
                        label = torch.index_select(
                            label, 0,
                            torch.LongTensor(pred_indices).to(input.device)
                            )

                    if uncertainty is not None:
                        uncertainty = torch.index_select(
                            uncertainty, 0,
                            torch.LongTensor(pred_indices).to(input.device)
                            )

                    self._provided_prediction_channels = [
                        self._provided_prediction_channels[i] for i in pred_indices
                        ]

            else:
                raise ValueError(
                    'PlotUtils Error: channels_to_plot needs to be either a list or the string: all'
                    )

        # create list of buttons and sliders for each figure
        self._ax_sliders = []
        self._sliders = []
        self._buttons = []
        self._n_slices = []

        # prealocate lists to the correct size
        self._n_figures = 0
        if self._plot_input:
            self._n_figures += 1

        if self._plot_prediction or self._plot_label or self._plot_error or self._plot_measurements or self._plot_measurements_variance or self._plot_uncertainty:
            self._n_figures += int(
                math.ceil(len(self._provided_prediction_channels) / 4.0)
                )

        for j in range(self._n_figures):
            self._ax_sliders += [None]
            self._sliders += [None]
            self._buttons += [None]
            self._n_slices += [0]

        # get the fontsizes
        self._title_fontsize = title_fontsize
        self._label_fontsize = label_fontsize
        self._tick_fontsize = tick_fontsize

        # choose design layout
        self._slider_location = [0.15, 0.025, 0.77, 0.04]
        self._button_location = [0.05, 0.01, 0.05, 0.08]

        # initialize the images to be displayed
        self._images = []

        if self._plot_prediction:
            prediction = prediction.cpu().numpy()

        if self._plot_label:
            label = label.cpu().numpy()

        if self._plot_uncertainty:
            uncertainty = uncertainty.cpu().numpy()

        # mask the input by the input mask if one is provided
        if input_mask is not None and self._plot_input:
            input_tmp = np.ma.MaskedArray(np.zeros(input.shape))
            is_masked = np.logical_not(input_mask.cpu().numpy().astype(bool))
            for i, channel in enumerate(input.cpu()):
                if self._provided_input_channels[i] != 'terrain':
                    input_tmp[i] = np.ma.masked_where(is_masked, channel)
                else:
                    if terrain is not None:
                        is_terrain = np.logical_not(terrain.cpu().numpy().astype(bool))
                        input_tmp[i] = np.ma.masked_where(is_terrain, channel)
                    else:
                        input_tmp[i] = channel

            input = input_tmp
        elif self._plot_input:
            input = input.cpu().numpy()

        # mask the measurements by the mask if one is provided
        if measurements_mask is not None and self._plot_measurements:
            meas_tmp = np.ma.MaskedArray(np.zeros(measurements.shape))
            is_masked = np.logical_not(measurements_mask.cpu().numpy().astype(bool))
            for i, channel in enumerate(measurements.cpu()):
                meas_tmp[i] = np.ma.masked_where(is_masked, channel)

            measurements = meas_tmp
        elif self._plot_measurements:
            measurements = measurements.cpu().numpy()

        if measurements_mask is not None and self._plot_measurements_variance:
            meas_var_tmp = np.ma.MaskedArray(np.zeros(measurements_variance.shape))
            is_masked = np.logical_not(measurements_mask.cpu().numpy().astype(bool))
            for i, channel in enumerate(measurements_variance.cpu()):
                meas_var_tmp[i] = np.ma.masked_where(is_masked, channel)

            measurements_variance = meas_var_tmp
        elif self._plot_measurements_variance:
            measurements_variance = measurements_variance.cpu().numpy()

        # mask the data by the terrain if it is provided
        if terrain is not None:
            no_terrain = np.logical_not(
                np.logical_not(terrain.cpu().numpy().astype(bool))
                )
            self._terrain_mask = np.ma.masked_where(no_terrain, no_terrain)

        else:
            self._terrain_mask = None

        if self._plot_input:
            self._input = input

        if self._plot_prediction:
            self._prediction = prediction

        if self._plot_label:
            self._label = label

        if self._plot_measurements:
            self._measurements = measurements

        if self._plot_measurements_variance:
            self._measurements_variance = measurements_variance

        if self._plot_error:
            if plot_prediction_error:
                self._error = self._label - self._prediction
            else:
                self._error = self._measurements - self._prediction

        if self._plot_uncertainty:
            self._uncertainty = uncertainty

        # if only the labels and the inputs are plotted combine them and plot in one figure
        if (
            self._plot_input and self._plot_label and not self._plot_prediction and
            not self._plot_measurements and not self._plot_error and
            not self._plot_uncertainty and not self._plot_measurements_variance
            ):
            self._input = np.ma.concatenate((self._input, self._label), 0)
            self._provided_input_channels += self._provided_prediction_channels
            self._provided_prediction_channels = None
            self._label = None
            self._plot_label = False

        # color map
        self._cmap = cmap.copy()
        self._cmap.set_bad(invalid_color)

        # terrain color map
        self._cmap_terrain = colors.LinearSegmentedColormap.from_list(
            'custom', colors.to_rgba_array([terrain_color, terrain_color]), 2
            )  #cm.binary_r
        self._cmap_terrain.set_bad('grey', 0.0)

        # the number of already open figures, used in slider and button callbacks
        self._n_already_open_figures = 0

        self.blocking = blocking

    def update_images(self):
        '''
        Updates the images according to the slice and axis which should be displayed. 
        '''
        j = plt.gcf().number - 1 - self._n_already_open_figures
        slice_number = self._n_slices[j]
        if self._axis == '  y-z':
            for im in self._images:
                im['image'].set_data(im['data'][:, :, slice_number])
                im['image'].set_extent([0, im['data'].shape[1], 0, im['data'].shape[0]])

        elif self._axis == '  x-y':
            for im in self._images:
                im['image'].set_data(im['data'][slice_number, :, :])
                im['image'].set_extent([0, im['data'].shape[2], 0, im['data'].shape[1]])

        else:
            for im in self._images:
                im['image'].set_data(im['data'][:, slice_number, :])
                im['image'].set_extent([0, im['data'].shape[2], 0, im['data'].shape[0]])

        plt.draw()

    def slider_callback(self, val):
        '''
        Callback for the slider to change the slice to display.
        '''
        figure = plt.gcf().number - 1 - self._n_already_open_figures
        self._n_slices[figure] = int(round(val))
        self.update_images()

    def radio_callback(self, label):
        '''
        Callback for the radio button to change the axis along which the slices are made.
        '''
        figure = plt.gcf().number - 1 - self._n_already_open_figures
        if label != self._axis:
            if label == '  y-z':
                max_slice = self._domain_shape[2] - 1
            elif label == '  x-y':
                max_slice = self._domain_shape[0] - 1
            else:
                max_slice = self._domain_shape[1] - 1

            if self._n_slices[figure] > max_slice:
                self._n_slices[figure] = max_slice

            self._ax_sliders[figure].remove()
            self._ax_sliders[figure] = plt.axes(self._slider_location)
            self._sliders[figure] = Slider(
                self._ax_sliders[figure],
                'Slice',
                0,
                max_slice,
                valinit=self._n_slices[figure],
                valfmt='%0.0f'
                )
            self._sliders[figure].on_changed(self.slider_callback)
        self._axis = label
        self.update_images()

    def plot(self):
        '''
        Creates the plots according to the available data
        '''
        # get the number of already open figures, used in slider and button callbacks
        self._n_already_open_figures = len(list(map(plt.figure, plt.get_fignums())))
        fig_idx = -1

        if self._plot_input:
            n_columns = int(math.ceil(math.sqrt(len(self._provided_input_channels))))
            n_rows = int(math.ceil(len(self._provided_input_channels) / n_columns))

            fig_in, ah_in = plt.subplots(
                n_rows, n_columns, squeeze=False, figsize=(14.5, 12)
                )
            fig_in.patch.set_facecolor('white')
            fig_idx += 1
            slice = self._n_slices[fig_idx]

            data_index = 0
            for j in range(n_rows):
                n_col = int(
                    min(
                        len(self._provided_input_channels) - n_columns * (j), n_columns
                        )
                    )
                for i in range(n_col):
                    im = {}
                    im['image'] = ah_in[j][i].imshow(
                        self._input[data_index, :, slice, :],
                        origin='lower',
                        vmin=self._input[data_index, :, :, :].min(),
                        vmax=self._input[data_index, :, :, :].max(),
                        aspect='equal',
                        cmap=self._cmap
                        )

                    chbar = fig_in.colorbar(im['image'], ax=ah_in[j][i])
                    plt.setp(chbar.ax.get_yticklabels(), fontsize=self._tick_fontsize)

                    im['data'] = self._input[data_index]
                    self._images.append(im)

                    if self._terrain_mask is not None:
                        im = {}
                        im['image'] = ah_in[j][i].imshow(
                            self._terrain_mask[:, slice, :],
                            cmap=self._cmap_terrain,
                            aspect='equal',
                            origin='lower'
                            )
                        im['data'] = self._terrain_mask
                        self._images.append(im)

                    ah_in[j][i].set_title(
                        self._title_dict[self._provided_input_channels[data_index]],
                        fontsize=self._title_fontsize
                        )

                    ah_in[j][i].set_xticks([])
                    ah_in[j][i].set_yticks([])

                    data_index += 1

                # remove the extra empty figures
                if n_col < n_columns and n_rows > 1:
                    for i in range(n_col, n_columns):
                        fig_in.delaxes(ah_in[j][i])

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.12)

            # create slider to select the slice
            self._ax_sliders[fig_idx] = plt.axes(self._slider_location)
            self._sliders[fig_idx] = Slider(
                self._ax_sliders[fig_idx],
                'Slice',
                0,
                self._domain_shape[1] - 1,
                valinit=self._n_slices[fig_idx],
                valfmt='%0.0f'
                )
            self._sliders[fig_idx].on_changed(self.slider_callback)

            # create button to select the axis along which the slices are made
            rax = plt.axes(self._button_location)
            label = ('  x-z', '  x-y', '  y-z')
            self._buttons[fig_idx] = RadioButtons(rax, label, active=0)
            for circle in self._buttons[fig_idx].circles:
                circle.set_radius(0.1)
            self._buttons[fig_idx].on_clicked(self.radio_callback)

        if self._plot_prediction or self._plot_label or self._plot_error or self._plot_uncertainty or self._plot_measurements or self._plot_measurements_variance:
            for j in range(
                int(math.ceil(len(self._provided_prediction_channels) / 4.0))
                ):
                # get the number of columns for this figure
                n_columns = int(min(len(self._provided_prediction_channels) - 4 * j, 4))
                n_rows = self._plot_prediction + self._plot_label + self._plot_error + self._plot_uncertainty + self._plot_measurements + self._plot_measurements_variance

                # create the new figure
                fig_in, ah_in = plt.subplots(
                    n_rows, n_columns, squeeze=False, figsize=(14.5, 12)
                    )
                fig_in.patch.set_facecolor('white')
                fig_idx += 1
                slice = self._n_slices[fig_idx]

                for i in range(n_columns):
                    data_index = i + j * 4
                    idx_row = 0

                    if self._plot_label:
                        im = {}
                        im['image'] = ah_in[idx_row][i].imshow(
                            self._label[data_index, :, slice, :],
                            origin='lower',
                            vmin=np.nanmin(self._label[data_index, :, :, :]),
                            vmax=np.nanmax(self._label[data_index, :, :, :]),
                            aspect='equal',
                            cmap=self._cmap
                            )

                        chbar = fig_in.colorbar(im['image'], ax=ah_in[idx_row][i])
                        plt.setp(
                            chbar.ax.get_yticklabels(), fontsize=self._tick_fontsize
                            )

                        im['data'] = self._label[data_index]
                        self._images.append(im)

                        if self._terrain_mask is not None:
                            im = {}
                            im['image'] = ah_in[idx_row][i].imshow(
                                self._terrain_mask[:, slice, :],
                                cmap=self._cmap_terrain,
                                aspect='equal',
                                origin='lower'
                                )
                            im['data'] = self._terrain_mask
                            self._images.append(im)

                        ah_in[idx_row][0].set_ylabel(
                            'CFD', fontsize=self._label_fontsize
                            )

                        idx_row += 1

                    if self._plot_prediction:
                        if self._plot_label:
                            lim_data = self._label[data_index, :, :, :]
                        else:
                            lim_data = self._prediction[data_index, :, :, :]

                        im = {}
                        im['image'] = ah_in[idx_row][i].imshow(
                            self._prediction[data_index, :, slice, :],
                            origin='lower',
                            vmin=np.nanmin(lim_data),
                            vmax=np.nanmax(lim_data),
                            aspect='equal',
                            cmap=self._cmap
                            )

                        chbar = fig_in.colorbar(im['image'], ax=ah_in[idx_row][i])
                        plt.setp(
                            chbar.ax.get_yticklabels(), fontsize=self._tick_fontsize
                            )

                        im['data'] = self._prediction[data_index]
                        self._images.append(im)

                        if self._terrain_mask is not None:
                            im = {}
                            im['image'] = ah_in[idx_row][i].imshow(
                                self._terrain_mask[:, slice, :],
                                cmap=self._cmap_terrain,
                                aspect='equal',
                                origin='lower'
                                )
                            im['data'] = self._terrain_mask
                            self._images.append(im)

                        ah_in[idx_row][0].set_ylabel(
                            'Prediction', fontsize=self._label_fontsize
                            )

                        idx_row += 1

                    if self._plot_error:
                        im = {}
                        im['image'] = ah_in[idx_row][i].imshow(
                            self._error[data_index, :, slice, :],
                            origin='lower',
                            vmin=np.nanmin(self._error[data_index, :, :, :]),
                            vmax=np.nanmax(self._error[data_index, :, :, :]),
                            aspect='equal',
                            cmap=self._cmap
                            )

                        chbar = fig_in.colorbar(im['image'], ax=ah_in[idx_row][i])
                        plt.setp(
                            chbar.ax.get_yticklabels(), fontsize=self._tick_fontsize
                            )

                        im['data'] = self._error[data_index]
                        self._images.append(im)

                        if self._terrain_mask is not None:
                            im = {}
                            im['image'] = ah_in[idx_row][i].imshow(
                                self._terrain_mask[:, slice, :],
                                cmap=self._cmap_terrain,
                                aspect='equal',
                                origin='lower'
                                )
                            im['data'] = self._terrain_mask
                            self._images.append(im)

                        ah_in[idx_row][0].set_ylabel(
                            'Error', fontsize=self._label_fontsize
                            )

                        idx_row += 1

                    if self._plot_measurements:
                        im = {}
                        im['image'] = ah_in[idx_row][i].imshow(
                            self._measurements[data_index, :, slice, :],
                            origin='lower',
                            vmin=np.nanmin(self._measurements[data_index, :, :, :]),
                            vmax=np.nanmax(self._measurements[data_index, :, :, :]),
                            aspect='equal',
                            cmap=self._cmap
                            )

                        chbar = fig_in.colorbar(im['image'], ax=ah_in[idx_row][i])
                        plt.setp(
                            chbar.ax.get_yticklabels(), fontsize=self._tick_fontsize
                            )

                        im['data'] = self._measurements[data_index]
                        self._images.append(im)

                        if self._terrain_mask is not None:
                            im = {}
                            im['image'] = ah_in[idx_row][i].imshow(
                                self._terrain_mask[:, slice, :],
                                cmap=self._cmap_terrain,
                                aspect='equal',
                                origin='lower'
                                )
                            im['data'] = self._terrain_mask
                            self._images.append(im)

                        ah_in[idx_row][0].set_ylabel(
                            'Measurements', fontsize=self._label_fontsize
                            )

                        idx_row += 1

                    if self._plot_measurements_variance:
                        im = {}
                        im['image'] = ah_in[idx_row][i].imshow(
                            self._measurements_variance[data_index, :, slice, :],
                            origin='lower',
                            vmin=np.nanmin(
                                self._measurements_variance[data_index, :, :, :]
                                ),
                            vmax=np.nanmax(
                                self._measurements_variance[data_index, :, :, :]
                                ),
                            aspect='equal',
                            cmap=self._cmap
                            )

                        chbar = fig_in.colorbar(im['image'], ax=ah_in[idx_row][i])
                        plt.setp(
                            chbar.ax.get_yticklabels(), fontsize=self._tick_fontsize
                            )

                        im['data'] = self._measurements_variance[data_index]
                        self._images.append(im)

                        if self._terrain_mask is not None:
                            im = {}
                            im['image'] = ah_in[idx_row][i].imshow(
                                self._terrain_mask[:, slice, :],
                                cmap=self._cmap_terrain,
                                aspect='equal',
                                origin='lower'
                                )
                            im['data'] = self._terrain_mask
                            self._images.append(im)

                        ah_in[idx_row][0].set_ylabel(
                            'Measurements Variance', fontsize=self._label_fontsize
                            )

                        idx_row += 1

                    if self._plot_uncertainty:
                        im = {}
                        im['image'] = ah_in[idx_row][i].imshow(
                            self._uncertainty[data_index, :, slice, :],
                            origin='lower',
                            vmin=np.nanmin(self._uncertainty[data_index, :, :, :]),
                            vmax=np.nanmax(self._uncertainty[data_index, :, :, :]),
                            aspect='equal',
                            cmap=self._cmap
                            )

                        chbar = fig_in.colorbar(im['image'], ax=ah_in[idx_row][i])
                        plt.setp(
                            chbar.ax.get_yticklabels(), fontsize=self._tick_fontsize
                            )

                        im['data'] = self._uncertainty[data_index]
                        self._images.append(im)

                        if self._terrain_mask is not None:
                            im = {}
                            im['image'] = ah_in[idx_row][i].imshow(
                                self._terrain_mask[:, slice, :],
                                cmap=self._cmap_terrain,
                                aspect='equal',
                                origin='lower'
                                )
                            im['data'] = self._terrain_mask
                            self._images.append(im)

                        ah_in[idx_row][0].set_ylabel(
                            'Uncertainty', fontsize=self._label_fontsize
                            )

                        idx_row += 1

                    ah_in[0][i].set_title(
                        self._title_dict[self._provided_prediction_channels[data_index]
                                         ],
                        fontsize=self._title_fontsize
                        )

                    for iter in range(idx_row):
                        ah_in[iter][i].set_xticks([])
                        ah_in[iter][i].set_yticks([])

                plt.tight_layout()
                plt.subplots_adjust(bottom=0.12)

                # create slider to select the slice
                self._ax_sliders[fig_idx] = plt.axes(self._slider_location)
                self._sliders[fig_idx] = Slider(
                    self._ax_sliders[fig_idx],
                    'Slice',
                    0,
                    self._domain_shape[1] - 1,
                    valinit=slice,
                    valfmt='%0.0f'
                    )
                self._sliders[fig_idx].on_changed(self.slider_callback)

                # create button to select the axis along which the slices are made
                rax = plt.axes(self._button_location)
                label = ('  x-z', '  x-y', '  y-z')
                self._buttons[fig_idx] = RadioButtons(rax, label, active=0)
                for circle in self._buttons[fig_idx].circles:
                    circle.set_radius(0.1)
                self._buttons[fig_idx].on_clicked(self.radio_callback)

        if self.blocking:
            plt.show()


def plot_sample(
        provided_input_channels,
        input,
        provided_label_channels,
        label,
        channels_to_plot='all',
        input_mask=None,
        ds=None,
        title_dict=None,
        blocking=True
    ):
    '''
    Creates the plots according to the input and label data.
    The axes along which the slices are made as well as the location of the slice
    can be set using sliders and buttons in the figure.

    Parameters
    ----------
    provided_input_channels : list of str
        Input channel names
    input : torch.Tensor
        Input tensor
    provided_label_channels : list of str
        Label channel names
    label : torch.Tensor
        Label tensor
    channels_to_plot : str or list of str, default: all
        Indicates which channels should be plotted, either 'all' or a list of the channels
    input_mask : torch.Tensor or None, default: None
        3D terrain tensor of input mask data indicating known cells [z, y, x].
    ds : list or None, default: None
        Cell size of the data grid, used for plotting the divergence field, if set and the data is available the divergency is computed and plotted
    title_dict :  dict or None, default: None
        An optional title dict can be passed, if one would like to replace the default titles or plot new channels
    blocking : bool, default: True
            Indicates if the plot call is blocking by calling plt.show()
    '''
    if 'terrain' in provided_input_channels:
        terrain = input[provided_input_channels.index('terrain')].squeeze()
    else:
        terrain = None

    instance = PlotUtils(
        provided_input_channels=provided_input_channels,
        provided_prediction_channels=provided_label_channels,
        channels_to_plot=channels_to_plot,
        input=input,
        label=label,
        terrain=terrain,
        ds=ds,
        input_mask=input_mask,
        title_dict=title_dict,
        blocking=blocking
        )
    instance.plot()


def plot_prediction(
        provided_prediction_channels,
        prediction=None,
        label=None,
        uncertainty=None,
        provided_input_channels=None,
        input=None,
        terrain=None,
        measurements=None,
        measurements_mask=None,
        ds=None,
        title_dict=None,
        blocking=True
    ):
    '''
    Creates the plots according to the data provided.
    The axes along which the slices are made as well as the location of the slice
    can be set using sliders and buttons in the figure.

    Parameters
    ----------
    provided_prediction_channels : list or None, default: None
        List of the channels of the prediction tensor
    prediction : torch.Tensor or None, default: None
        4D prediction tensor with [channels, z, y, x]. Channels must be ordered as in provided_prediction_channels.
    label : torch.Tensor or None, default: None
        4D label tensor with [channels, z, y, x]. Channels must be ordered as in provided_prediction_channels.
    uncertainty : torch.Tensor or None, default: None
        4D uncertainty tensor with [channels, z, y, x]. Channels must be ordered as in provided_prediction_channels.
    provided_input_channels : list or None, default: None
        List of the input channels to the network
    input : torch.Tensor or None, default: None
        4D input tensor with [channels, z, y, x]. Channels must be ordered as in provided_input_channels.
    terrain : torch.Tensor or None, default: None
        3D terrain tensor of terrain data [z, y, x].
    measurements : torch.Tensor or None, default: None
        4D measurement tensor with [channels, z, y, x]. Channels must be ordered as in provided_prediction_channels.
    measurements_mask : torch,Tensor or None, default: None
        3D tensor with [z, y, x].
    ds : list or None, default: None
        Cell size of the data grid, used for plotting the divergence field, if set and the data is available the divergency is computed and plotted
    title_dict :  dict or None, default: None
        An optional title dict can be passed, if one would like to replace the default titles or plot new channels
    blocking : bool, default: True
        Indicates if the plot call is blocking by calling plt.show()
    '''

    input_mask = None
    if provided_input_channels is not None and input is not None:
        if 'mask' in provided_input_channels:
            idx = provided_input_channels.index('mask')
            input_mask = input[idx]

    instance = PlotUtils(
        provided_prediction_channels=provided_prediction_channels,
        provided_input_channels=provided_input_channels,
        input_mask=input_mask,
        prediction=prediction,
        input=input,
        label=label,
        uncertainty=uncertainty,
        measurements=measurements,
        measurements_mask=measurements_mask,
        terrain=terrain,
        ds=ds,
        title_dict=title_dict,
        blocking=blocking
        )
    instance.plot()


def plot_measurement(
        provided_measurement_channels,
        measurements,
        measurements_mask,
        terrain=None,
        variance=None,
        prediction=None,
        title_dict=None,
        blocking=True
    ):
    '''
    Create plots of the measurements and the respective variance

    Parameters
    ----------
    provided_measurement_channels : list or None, default: None
        List of the channels of the measurement tensor
    measurements : torch.Tensor or None, default: None
        4D measurement tensor with [channels, z, y, x]. Channels must be ordered as in provided_prediction_channels.
    measurements_mask : torch,Tensor or None, default: None
        3D tensor with [z, y, x].
    terrain : torch.Tensor or None, default: None
        3D terrain tensor of terrain data [z, y, x].
    variance : torch.Tensor or None, default: None
        4D measurement variance tensor with [channels, z, y, x]. Channels must be ordered as in provided_prediction_channels. 
    prediction : torch.Tensor or None, default: None
        4D prediction tensor with [channels, z, y, x]. Channels must be ordered as in provided_prediction_channels.
    title_dict :  dict or None, default: None
        An optional title dict can be passed, if one would like to replace the default titles or plot new channels
    blocking : bool, default: True
        Indicates if the plot call is blocking by calling plt.show()
    '''
    instance = PlotUtils(
        provided_prediction_channels=provided_measurement_channels,
        measurements=measurements,
        measurements_mask=measurements_mask,
        measurements_variance=variance,
        terrain=terrain,
        prediction=prediction,
        title_dict=title_dict,
        blocking=blocking
        )
    instance.plot()


def violin_plot(labels, data, xlabel, ylabel, ylim=None):
    '''
    Generate a violin plot.

    Parameters
    ----------
    labels : list of str
        Labels for each data group
    data : list of array
        Data that is plotted
    xlabel : str
        Label of the x-axis
    ylabel : str
        Label of the y-axis
    ylim : None or list of int
        If not none specifies the y-axis limits
    '''
    index = np.arange(len(labels))

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')

    # need to manually set the factor and make sure that it is not too small, otherwise a numerical underflow will happen
    factor = np.power(len(data[0]), -1.0 / (len(data) + 4))
    parts = ax.violinplot(
        data,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=300,
        bw_method=np.max([factor, 0.6])
        )

    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    quartile1 = []
    medians = []
    quartile3 = []
    for channel in data:
        quartile1_channel, medians_channel, quartile3_channel = np.percentile(
            channel, [25, 50, 75]
            )
        quartile1.append(quartile1_channel)
        medians.append(medians_channel)
        quartile3.append(quartile3_channel)

    whiskers = np.array([
        adjacent_values(sorted(sorted_array), q1, q3)
        for sorted_array, q1, q3 in zip(data, quartile1, quartile3)
        ])
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
    '''
    Compute the adjacent values for the violin plot

    Parameters
    ----------
    vals : array
        Data array
    q1 : float
        Value of first quartile
    q3 : float
        Value of third quartile
    '''
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value
