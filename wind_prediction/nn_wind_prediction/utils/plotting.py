from .derivation import divergence
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.widgets import Slider, RadioButtons
import torch
import math

class PlotUtils():
    '''
    Class providing the tools to plot the input and labels for the 2D and 3D case.

    Params:
        plot_mode: 'sample' or 'prediction'. Sample plot mode only requires input channels and terrain, whereas prediction
                    mode requires label as well.
        provided_channels: a list of the channels that were passed to input (and output for prediction mode)
        channels_to_plot: subset of provided channels, indicates the channels which will be in the plot. 'all' can be
                          used to plot all of the provided channels.
        input: 4D tensor with [channels, z, y, x]. Channels must be ordered as in default_channels.
        label: 4D tensor with [channels, z, y, x]. Channels must be ordered as in default_channels.
        terrain: 3D tensor of terrain data [z, y, x]
        design: 0 or 1. Indicates the desired design.
        uncertainty_predicted: if the uncertainty is predicted and not the actual channels. Used in prediction mode.
        plot_divergence: if the divergence field should be plotted. Note that ds must then be passed to kwargs and that
                         the velocities (ux, uy, uz) or (ux_cfd, uy_cfd, uz_cfd) need to be present.
        ds: grid size of the data, used for plotting the divergence field
        title_dict: an optional title dict can be passed, if one would like to replace the default titles
        title_fontsize: font size of the title
        label_fontsize: font size of the label
        tick_fontsize: font size of the tick
        cmap: color map to use
        terrain_color: color of the terrain

    the default channels for sample plotting
    ['ux_in', 'uy_in', 'uz_in','terrain', 'ux_cfd', 'uy_cfd', 'uz_cfd', 'turb', 'p', 'epsilon', 'nut']

    the default channels for prediction plotting
    ['terrain', 'ux', 'uy', 'uz', 'turb', 'p', 'epsilon', 'nut']
    '''
    def __init__(self, plot_mode, provided_channels, channels_to_plot, input, label, terrain, design, masked_input=None,
                 uncertainty = None, plot_divergence = False, ds = None, title_dict = None,
                 title_fontsize = 16, label_fontsize = 15, tick_fontsize = 10, cmap=cm.jet, terrain_color='grey'):

        if plot_mode != 'sample' and plot_mode !='prediction':
            raise ValueError('Unrecognized plot mode: {}'.format(plot_mode))

        # Input is the prediction, label is CFD
        self.__axis = 'x-z'
        self.__uncertainty_predicted = uncertainty is not None

        if channels_to_plot == 'all':
            channels_to_plot = list(provided_channels)

        if len(provided_channels) == 0 or len(channels_to_plot) == 0:
            raise ValueError('PlotUtils: List of channels cannot be empty')

        # set the default channels and titles
        default_channels = [ 'terrain','ux', 'uy', 'uz', 'turb', 'p', 'epsilon', 'nut','ux_in', 'uy_in', 'uz_in', 'ux_cfd',
                                 'uy_cfd', 'uz_cfd', 'turb_cfd', 'p_cfd', 'epsilon_cfd', 'nut_cfd']

        default_title_dict = {'terrain': 'Distance field','ux': 'Velocity X [m/s]', 'uy': 'Velocity Y [m/s]',
                              'uz': 'Velocity Z [m/s]', 'turb': 'Turb. kin. energy [m^2/s^2]', 'p': 'Rho-norm. pressure [m^2/s^2]',
                                'epsilon': 'Dissipation [m^2/s^3]', 'nut': 'Turb. viscosity [m^2/s]', 'ux_in': ' Input Velocity X [m/s]',
                              'uy_in': 'Input Velocity Y [m/s]', 'uz_in': 'Input Velocity Z [m/s]', 'ux_cfd': ' CFD Velocity X [m/s]',
                              'uy_cfd': 'CFD Velocity Y [m/s]', 'uz_cfd': 'CFD Velocity Z [m/s]', 'turb_cfd': 'CFD Turb. kin. energy [m^2/s^2]',
                              'p_cfd': 'CFD Rho-norm. pressure [m^2/s^2]', 'epsilon_cfd': 'CFD Dissipation [m^2/s^3]',
                              'nut_cfd': 'CFD Turb. viscosity [m^2/s]', 'div': 'Divergence [1/s]'}

        for channel in channels_to_plot:
            if channel not in provided_channels:
                raise ValueError('PlotUtils: channels_to_plot contains a channel that is not in provided_channels: {}'.format(channel))

        for channel in channels_to_plot:
            if channel not in default_channels:
                print('PlotUtils warning: None default channel_to_plot detected: \'{}\', '
                                 'correct channels are {}'.format(channel, default_channels))
                if title_dict is None:
                    raise ValueError('PlotUtils: None default channel provided, but no custom title dict')

        # set the title dict
        if title_dict is not None:
            default_title_dict.update(title_dict)
        self.__title_dict = default_title_dict

        # set the channels
        self.__channels_to_plot = channels_to_plot
        self.__provided_channels = provided_channels

        # divergence plotting handling. depends on plot mode
        if plot_mode == 'sample':
            div_required_channels = ['ux_cfd', 'uy_cfd', 'uz_cfd']
        else:
            div_required_channels = ['ux', 'uy', 'uz']

        if plot_divergence and ds and all(elem in self.__provided_channels for elem in div_required_channels):
            self.__channels_to_plot += ['div']
            self.__provided_channels += ['div']
            vel_indices = torch.LongTensor([self.__provided_channels.index(channel) for channel in ['ux_cfd', 'uy_cfd', 'uz_cfd']]).to(input.device)

            # get divergence of input field
            input_div = divergence(input.index_select(0,vel_indices).unsqueeze(0), ds, terrain.unsqueeze(0)).squeeze().unsqueeze(0)
            input = torch.cat((input, input_div), 0)

            if plot_mode == 'prediction':
                # get divergence of label field for prediciton
                label_div = divergence(label.index_select(0, vel_indices).unsqueeze(0), ds, terrain.unsqueeze(0)).squeeze().unsqueeze(0)
                label = torch.cat((label, label_div), 0)

        elif plot_divergence and not ds:
            print('PlotUtils warning: cannot plot divergence, grid size not passed')
        elif plot_divergence and not all(elem in self.__provided_channels for elem in div_required_channels):
            print('PlotUtils warning: cannot plot divergence, missing some or all of the required channels: {}'.format(div_required_channels))

        # get the number of channels to plot:
        self.__n_channels = len(self.__channels_to_plot)

        # get the indices of the channels that must be plotted
        self.__indices_to_plot = torch.LongTensor(
            [self.__provided_channels.index(channel) for channel in self.__channels_to_plot]).to(input.device)

        # reduce tensors to plot to the wanted channels
        if plot_mode == 'sample':
            # reduce the input to what is plotted. No need for label if plotting mode is sample
            input = torch.index_select(input, 0, self.__indices_to_plot)
            label = input

            # set the number of figures to plot
            self.__n_figures = 1

        else:
            # reduce the input and label to what is plotted
            input = torch.index_select(input,0, self.__indices_to_plot)
            label = torch.index_select(label,0, self.__indices_to_plot)

            # get the number of figures to plot
            self.__n_figures = math.ceil(math.ceil(self.__n_channels / 4))

        # create list of buttons and sliders for each figure
        self.__ax_sliders = []
        self.__sliders = []
        self.__buttons = []
        self.__n_slices = []

        # prealocate lists to the correct size
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
        if design == 1:
            self.__slider_location = [0.15, 0.025, 0.77, 0.04]
            self.__button_location = [0.05, 0.01, 0.05, 0.08]

        else:
            self.__slider_location = [0.15, 0.025, 0.77, 0.04]#[0.09, 0.02, 0.82, 0.04]
            self.__button_location = [0.05, 0.01, 0.05, 0.08]#[0.80, 0.16, 0.05, 0.10]

        # initialize the images to be displayed
        self.__in_images = []
        self.__out_images = []
        self.__error_images = []
        self.__uncertainty_images = []
        self.__mask_images = []

        # NUT
        self.__nut_images = []

        # Set the input to be a masked array so that we specify a terrain colour
        self.__input = np.ma.MaskedArray(np.zeros(input.shape))
        is_terrain = np.logical_not(terrain.cpu().numpy().astype(bool))
        for i, channel in enumerate(input.cpu()):
            self.__input[i] = np.ma.masked_where(is_terrain, channel)

        self.__label = np.ma.MaskedArray(np.zeros(label.shape))
        for i, channel in enumerate(label.cpu()):
            self.__label[i] = np.ma.masked_where(is_terrain, channel)

        if self.__uncertainty_predicted:
            self.__uncertainty = np.ma.MaskedArray(np.zeros(uncertainty.shape))
            for i, channel in enumerate(uncertainty.cpu()):
                self.__uncertainty[i] = np.ma.masked_where(is_terrain, channel)

        self.__cmap = cmap
        self.__cmap.set_bad(terrain_color)

        # get masked input if it is provided
        if masked_input is not None:
            self.__masked_input = np.ma.MaskedArray(np.zeros(masked_input.shape))
            is_mask = np.logical_not(masked_input[0].cpu().numpy().astype(bool))
            for i, channel in enumerate(masked_input.cpu()):
                self.__masked_input[i] = np.ma.masked_where(is_mask, channel)
        else:
            self.__masked_input = masked_input

        # handle uncertainty prediction case
        if uncertainty_predicted and plot_mode == 'prediction':
            self.__uncertainty = self.__input[int(self.__n_channels/2):,:]
            print('Warning: Uncertainty plotting has not been used in a while. It might be broken.')

        if design == 1:
            self.__error = self.__label - self.__input
        else:
            self.__error = None # in this case the error plot will not be executed anyway

        # the number of already open figures, used in slider and button callbacks
        self.__n_already_open_figures = 0

    def update_images(self):
        '''
        Updates the images according to the slice and axis which should be displayed. 
        '''
        j = plt.gcf().number - 1 -self.__n_already_open_figures
        slice_number = self.__n_slices[j]
        if self.__axis == '  y-z':
            for i, im in enumerate(self.__in_images):
                im.set_data(self.__input[i, :, :, slice_number])
                im.set_extent([0, self.__input.shape[2], 0, self.__input.shape[1]])

            for i, im in enumerate(self.__out_images):
                im.set_data(self.__label[i, :, :, slice_number])
                im.set_extent([0, self.__label.shape[2], 0, self.__label.shape[1]])

            for i, im in enumerate(self.__error_images):
                im.set_data(self.__error[i, :, :, slice_number])
                im.set_extent([0, self.__error.shape[2], 0, self.__error.shape[1]])

            for i, im in enumerate(self.__uncertainty_images):
                im.set_data(self.__uncertainty[i, :, :, slice_number])
                im.set_extent([0, self.__uncertainty.shape[2], 0, self.__uncertainty.shape[1]])

            for i, im in enumerate(self.__mask_images):
                im.set_data(self.__masked_input[i, :, :, slice_number])
                im.set_extent([0, self.__masked_input.shape[2], 0, self.__masked_input.shape[1]])

            # NUT
            for i, im in enumerate(self.__nut_images):
                im.set_data(self.__label[6, :, :, slice_number])
                im.set_extent([0, self.__label.shape[2], 0, self.__label.shape[1]])


        elif self.__axis == '  x-y':
            for i, im in enumerate(self.__in_images):
                im.set_data(self.__input[i, slice_number, :, :])
                im.set_extent([0, self.__input.shape[3], 0, self.__input.shape[2]])

            for i, im in enumerate(self.__out_images):
                im.set_data(self.__label[i, slice_number, :, :])
                im.set_extent([0, self.__label.shape[3], 0, self.__label.shape[2]])

            for i, im in enumerate(self.__error_images):
                im.set_data(self.__error[i, slice_number, :, :])
                im.set_extent([0, self.__error.shape[3], 0, self.__error.shape[2]])

            for i, im in enumerate(self.__uncertainty_images):
                im.set_data(self.__uncertainty[i, slice_number, :, :])
                im.set_extent([0, self.__uncertainty.shape[3], 0, self.__uncertainty.shape[2]])

            for i, im in enumerate(self.__mask_images):
                im.set_data(self.__masked_input[i, slice_number, :, :])
                im.set_extent([0, self.__masked_input.shape[3], 0, self.__masked_input.shape[2]])

            # NUT
            for i, im in enumerate(self.__nut_images):
                im.set_data(self.__label[6, slice_number, :, :])
                im.set_extent([0, self.__label.shape[3], 0, self.__label.shape[2]])

        else:
            for i, im in enumerate(self.__in_images):
                im.set_data(self.__input[i, :, slice_number, :])
                im.set_extent([0, self.__input.shape[3], 0, self.__input.shape[1]])

            for i, im in enumerate(self.__out_images):
                im.set_data(self.__label[i, :, slice_number, :])
                im.set_extent([0, self.__label.shape[3], 0, self.__label.shape[1]])

            for i, im in enumerate(self.__error_images):
                im.set_data(self.__error[i, :, slice_number, :])
                im.set_extent([0, self.__error.shape[3], 0, self.__error.shape[1]])

            for i, im in enumerate(self.__uncertainty_images):
                im.set_data(self.__uncertainty[i, :, slice_number, :])
                im.set_extent([0, self.__uncertainty.shape[3], 0, self.__uncertainty.shape[1]])

            for i, im in enumerate(self.__mask_images):
                im.set_data(self.__masked_input[i, :, slice_number, :])
                im.set_extent([0, self.__masked_input.shape[3], 0, self.__masked_input.shape[1]])

            # NUT
            for i, im in enumerate(self.__nut_images):
                im.set_data(self.__label[6, :, slice_number, :])
                im.set_extent([0, self.__label.shape[3], 0, self.__label.shape[2]])
        plt.draw()

    def slider_callback(self, val):
        '''
        Callback for the slider to change the slice to display.
        '''
        figure = plt.gcf().number - 1 -self.__n_already_open_figures
        self.__n_slices[figure] = int(round(val))
        self.update_images()

    def radio_callback(self, label):
        '''
        Callback for the radio button to change the axis along which the slices are made.
        '''
        figure = plt.gcf().number - 1 -self.__n_already_open_figures
        if label != self.__axis:
            if label == '  y-z':
                max_slice = self.__input.shape[3] - 1
            elif label == '  x-y':
                max_slice = self.__input.shape[1] - 1
            else:
                max_slice = self.__input.shape[2] - 1

            if self.__n_slices[figure] > max_slice:
                self.__n_slices[figure] = max_slice

            self.__ax_sliders[figure].remove()
            self.__ax_sliders[figure] = plt.axes(self.__slider_location)
            self.__sliders[figure] = Slider(self.__ax_sliders[figure], 'Slice', 0, max_slice, valinit=self.__n_slices[figure], valfmt='%0.0f')
            self.__sliders[figure].on_changed(self.slider_callback)
        self.__axis = label
        self.update_images()

    def plot_sample(self):
        '''
        Creates the plots of a single sample according to the input and label data.
        '''

        # get the number of already open figures, used in slider and button callbacks
        self.__n_already_open_figures = len(list(map(plt.figure, plt.get_fignums())))
        # 3D data
        if (len(list(self.__input.shape)) > 3):

            n_rows = math.ceil(self.__n_channels/4)
            n_columns = min(self.__n_channels, 4)

            fig_in, ah_in = plt.subplots(n_rows, n_columns, figsize=(16,13), squeeze=False)
            fig_in.patch.set_facecolor('white')
            data_index = 0
            for j in range(n_rows):
                n_columns = min(self.__n_channels - 4 * (j), 4)
                for i in range(n_columns):
                    self.__out_images.append(
                        ah_in[j][i].imshow(self.__input[data_index, :, self.__n_slices[0], :], origin='lower',
                                           vmin=self.__input[data_index, :, :, :].min(),
                                           vmax=self.__input[data_index, :, :, :].max(), aspect='auto', cmap=self.__cmap))

                    ah_in[j][i].set_title(self.__title_dict[self.__channels_to_plot[data_index]],
                                          fontsize=self.__title_fontsize)

                    ah_in[j][i].set_xticks([])
                    ah_in[j][i].set_yticks([])

                    chbar = fig_in.colorbar(self.__out_images[data_index], ax=ah_in[j][i])
                    plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
                    data_index += 1

                # remove the extra empty figures
                if n_columns<4 and n_rows>1:
                    for i in range(n_columns, 4):
                        fig_in.delaxes(ah_in[j][i])

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.12)

            # create slider to select the slice
            self.__ax_sliders[0] = plt.axes(self.__slider_location)
            self.__sliders[0] = Slider(self.__ax_sliders[0], 'Slice', 0, self.__input.shape[2] - 1,
                                       valinit=self.__n_slices[0], valfmt='%0.0f')
            self.__sliders[0].on_changed(self.slider_callback)

            # create button to select the axis along which the slices are made
            rax = plt.axes(self.__button_location)
            label = ('  x-z', '  x-y', '  y-z')
            self.__buttons[0] = RadioButtons(rax, label, active=0)
            for circle in self.__buttons[0].circles:
                circle.set_radius(0.1)
            self.__buttons[0].on_clicked(self.radio_callback)

            plt.show()

        # 2D data
        else:
            raise NotImplementedError('Sorry, 2D sample plotting needs to be reimplemented.')

    def plot_prediction(self, save=False, label_name='CFD', input_name='Prediction', add_sparse_mask_row=False):
        # get the number of already open figures, used in slider and button callbacks
        self.__n_already_open_figures = len(list(map(plt.figure, plt.get_fignums())))

        # 3D case
        if (len(list(self.__input.shape)) > 3):

            # loop over the channels to plot in each figure
            for j in range(self.__n_figures):

                # get the number of columns for this figure
                n_columns = min(self.__n_channels-4*(j), 4)

                column_size = 5

                # create new figure
                if self.__uncertainty_predicted and add_sparse_mask_row:
                    n_rows = 5
                elif self.__uncertainty_predicted and not add_sparse_mask_row:
                    n_rows = 4
                elif not self.__uncertainty_predicted and add_sparse_mask_row:
                    n_rows = 4
                else:
                    n_rows = 3
                fig_in, ah_in = plt.subplots(n_rows, n_columns, figsize=(n_columns * column_size, 12), squeeze=False)

                fig_in.patch.set_facecolor('white')
                slice = self.__n_slices[j]

                for i in range(n_columns):
                    data_index = i+j*4
                    self.__out_images.append(ah_in[0][i].imshow(
                        self.__label[data_index,:,slice,:], origin='lower',
                        vmin=np.nanmin(self.__label[data_index,:,:,:]),
                        vmax=np.nanmax(self.__label[data_index,:,:,:]),
                        aspect = 'auto', cmap=self.__cmap))
                    self.__in_images.append(ah_in[1][i].imshow(
                        self.__input[data_index,:,slice,:], origin='lower',
                        vmin=np.nanmin(self.__label[data_index,:,:,:]),
                        vmax=np.nanmax(self.__label[data_index,:,:,:]),
                        aspect = 'auto', cmap=self.__cmap))
                    self.__error_images.append(ah_in[2][i].imshow(
                        self.__error[data_index,:,slice,:], origin='lower',
                        vmin=np.nanmin(self.__error[data_index,:,:,:]),
                        vmax=np.nanmax(self.__error[data_index,:,:,:]),
                        aspect='auto', cmap=self.__cmap))

                    ah_in[0][i].set_title(self.__title_dict[self.__channels_to_plot[data_index]], fontsize = self.__title_fontsize)

                    ah_in[0][i].set_xticks([])
                    ah_in[0][i].set_yticks([])
                    ah_in[1][i].set_xticks([])
                    ah_in[1][i].set_yticks([])
                    ah_in[2][i].set_xticks([])
                    ah_in[2][i].set_yticks([])

                    chbar = fig_in.colorbar(self.__out_images[data_index], ax=ah_in[0][i])
                    plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
                    chbar = fig_in.colorbar(self.__in_images[data_index], ax=ah_in[1][i])
                    plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
                    chbar = fig_in.colorbar(self.__error_images[data_index], ax=ah_in[2][i])
                    plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)

                    if self.__uncertainty_predicted:
                        if add_sparse_mask_row:
                            row = n_rows - 2
                        else:
                            row = n_rows - 1
                        self.__uncertainty_images.append(
                            ah_in[row][i].imshow(self.__uncertainty[i,:,slice,:], origin='lower',
                                               vmin=self.__uncertainty[i,:,:,:].min(),
                                               vmax=self.__uncertainty[i,:,:,:].max(),
                                               aspect = 'auto', cmap=self.__cmap))
                        chbar = self.__figures[j].colorbar(self.__uncertainty_images[data_index], ax=ah_in[row][i])
                        plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
                        plt.setp(ah_in[row][i].get_xticklabels(), fontsize=self.__tick_fontsize)
                        plt.setp(ah_in[row][i].get_yticklabels(), fontsize=self.__tick_fontsize)

                        ah_in[row][i].set_xticks([])
                        ah_in[row][i].set_yticks([])
                    if add_sparse_mask_row:
                        row = n_rows - 1
                        self.__mask_images.append(
                            ah_in[row][i].imshow(self.__masked_input[i, :, slice, :], origin='lower',
                                               vmin=self.__masked_input[i, :, :, :].min(),
                                               vmax=self.__masked_input[i, :, :, :].max(),
                                               aspect='auto', cmap=self.__cmap))

                        ah_in[row][i].set_xticks([])
                        ah_in[row][i].set_yticks([])

                        # chbar = self.__figures[j].colorbar(self.__mask_images[data_index], ax=ah_in[row][i])
                        chbar = fig_in.colorbar(self.__mask_images[data_index], ax=ah_in[row][i])
                        plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)

                    if (i == 0):
                        ah_in[0][i].set_ylabel(label_name,
                                               fontsize=self.__label_fontsize)
                        ah_in[1][i].set_ylabel(input_name,
                                               fontsize=self.__label_fontsize)
                        ah_in[2][i].set_ylabel('Error', fontsize=self.__label_fontsize)
                        if self.__uncertainty_predicted:
                            if add_sparse_mask_row:
                                row = n_rows - 2
                            else:
                                row = n_rows - 1
                            ah_in[row][i].set_ylabel('Uncertainty', fontsize=self.__label_fontsize)
                        if add_sparse_mask_row:
                            row = n_rows - 1
                            ah_in[row][i].set_ylabel('Mask', fontsize=self.__label_fontsize)

                plt.tight_layout()
                plt.subplots_adjust(bottom=0.12)

                # create slider to select the slice
                self.__ax_sliders[j] = plt.axes(self.__slider_location)
                self.__sliders[j] = Slider(self.__ax_sliders[j], 'Slice', 0, self.__input.shape[2]-1, valinit=slice, valfmt='%0.0f')
                self.__sliders[j].on_changed(self.slider_callback)

                # create button to select the axis along which the slices are made
                rax = plt.axes(self.__button_location)
                label = ('  x-z', '  x-y', '  y-z')
                self.__buttons[j] = RadioButtons(rax, label, active=0)
                for circle in self.__buttons[j].circles:
                    circle.set_radius(0.1)
                self.__buttons[j].on_clicked(self.radio_callback)

            if not save:
                plt.show()

        # 2D case
        else:
            raise NotImplementedError('Sorry, 2D prediction plotting needs to be reimplemented.')

        return fig_in, ah_in

    def plotting_nut(self):

        # 3D data
        fh_in, ah_in = plt.subplots(2, 3, figsize=(10,7))


        # Viscosity CFD
        
        self.__nut_images.append(ah_in[0][0].imshow(self.__label[6,:,self.__n_slices[0],:], origin='lower', vmin=0, vmax=self.__label[6,:,:,:].max(), aspect = 'auto', cmap=self.__cmap))
        chbar = fh_in.colorbar(self.__nut_images[0], ax=ah_in[0][0])
        ah_in[0][0].set_title('CFD', fontsize = self.__title_fontsize)
        plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
        plt.setp(ah_in[0][0].get_xticklabels(), fontsize=self.__tick_fontsize)
        plt.setp(ah_in[0][0].get_yticklabels(), fontsize=self.__tick_fontsize)
        ah_in[0][0].set_xticks([])
        ah_in[0][0].set_yticks([])

        # Compute nu_t with the formula: 
        c_mu = 0.09
        k_square = np.multiply(self.__label[3,:,:,:],self.__label[3,:,:,:])
        nut = c_mu * np.divide(k_square, self.__label[5,:,:,:])

        self.__nut_images.append(ah_in[0][1].imshow(nut[:,self.__n_slices[0],:], origin='lower', vmin=0, vmax=self.__label[6,:,:,:].max(), aspect = 'auto', cmap=self.__cmap))
        chbar = fh_in.colorbar(self.__nut_images[1], ax=ah_in[0][1])
        ah_in[0][1].set_title('Calculation', fontsize = self.__title_fontsize)
        plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
        plt.setp(ah_in[0][1].get_xticklabels(), fontsize=self.__tick_fontsize)
        plt.setp(ah_in[0][1].get_yticklabels(), fontsize=self.__tick_fontsize)
        ah_in[0][1].set_xticks([])
        ah_in[0][1].set_yticks([])

        # Compute the error between the CFD and the formula
        error_nut = self.__label[6,:,:,:] - nut

        print('------------------------------------------------------------------------------')
        print('Viscosity: min. error: ', error_nut.min())
        print('Viscosity: max. error: ', error_nut.max())
        print('Viscosity: mean error: ', error_nut.mean())
        #print('\n\n', error_nut[:,32,32])
        #print('\n\n', error_nut[32,:,32])
        #print('\n\n', error_nut[32,32,:])
        print('------------------------------------------------------------------------------')

        self.__error_images.append(ah_in[0][2].imshow(error_nut[:,self.__n_slices[0],:], origin='lower', vmin=-1, vmax=5, aspect = 'auto', cmap=self.__cmap))
        chbar = fh_in.colorbar(self.__error_images[0], ax=ah_in[0][2])
        ah_in[0][2].set_title('Error', fontsize = self.__title_fontsize)
        plt.setp(chbar.ax.get_yticklabels(), fontsize=self.__tick_fontsize)
        plt.setp(ah_in[0][2].get_xticklabels(), fontsize=self.__tick_fontsize)
        plt.setp(ah_in[0][2].get_yticklabels(), fontsize=self.__tick_fontsize)
        ah_in[0][2].set_xticks([])
        ah_in[0][2].set_yticks([])

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)

        # create slider to select the slice
        self.__ax_sliders[0] = plt.axes(self.__slider_location)
        self.__sliders[0] = Slider(self.__ax_sliders[0], 'Slice', 0, self.__input.shape[2] - 1,
                                   valinit=self.__n_slices[0], valfmt='%0.0f')
        self.__sliders[0].on_changed(self.slider_callback)

        # create button to select the axis along which the slices are made
        rax = plt.axes(self.__button_location)
        label = ('  x-z', '  x-y', '  y-z')
        self.__buttons[0] = RadioButtons(rax, label, active=0)
        for circle in self.__buttons[0].circles:
            circle.set_radius(0.1)
        self.__buttons[0].on_clicked(self.radio_callback)

        plt.show()

def plot_sample(provided_channels, channels_to_plot, input, label, terrain, plot_divergence = False, ds = None, title_dict = None):
    '''
    Creates the plots according to the input and label data.
    Can handle 2D as well as 3D input. For the 3D input only slices are shown.
    The axes along which the slices are made as well as the location of the slice
    can be set using sliders and buttons in the figure.
    '''
    instance = PlotUtils('sample',provided_channels, channels_to_plot, input, label, terrain, 0, False, plot_divergence,
                         ds, title_dict =title_dict)
    instance.plot_sample()

def plot_prediction(provided_channels, channels_to_plot, output, label, terrain, uncertainty = None, plot_divergence = False,
                 ds = None, title_dict = None):
    instance = PlotUtils('prediction',provided_channels, channels_to_plot, output, label, terrain, 1, uncertainty,
                         plot_divergence, ds, title_dict =title_dict)
    instance.plot_prediction()

def plotting_nut(provided_channels, channels_to_plot, output, label, terrain):
    instance = PlotUtils('sample',provided_channels, channels_to_plot, output, label, terrain, 1)
    instance.plotting_nut()

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
