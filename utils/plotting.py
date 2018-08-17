import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

class PlotUtils():
    '''
    Class providing the tools to plot the input and labels for the 2D and 3D case.
    '''
    def __init__(self, input, label, design):
        self.__axis = 'x-z'
        self.__n_slice = 0

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

            for i, im in enumerate(self.__out_images):
                im.set_data(self.__label[i, :, :, self.__n_slice])
                im.set_extent([0, self.__label.shape[2], 0, self.__label.shape[1]])

            for i, im in enumerate(self.__error_images):
                im.set_data(self.__error[i, :, :, self.__n_slice])
                im.set_extent([0, self.__error.shape[2], 0, self.__error.shape[1]])

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

        plt.draw()

    def slider_callback(self, val):
        '''
        Callback for the slider to change the slice to display.
        '''
        self.__n_slice = int(val)
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
            fh_in, ah_in = plt.subplots(3, 3)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.1)
            fh_in.delaxes(ah_in[2][2])

            # plot the input data
            self.__in_images.append(ah_in[2][0].imshow(self.__input[0,:,self.__n_slice,:], origin='lower', vmin=self.__input[0,:,:,:].min(), vmax=self.__input[0,:,:,:].max(), aspect = 'auto')) #terrain
            self.__in_images.append(ah_in[0][0].imshow(self.__input[1,:,self.__n_slice,:], origin='lower', vmin=self.__label[0,:,:,:].min(), vmax=self.__label[0,:,:,:].max(), aspect = 'auto')) #ux
            self.__in_images.append(ah_in[0][1].imshow(self.__input[2,:,self.__n_slice,:], origin='lower', vmin=self.__label[1,:,:,:].min(), vmax=self.__label[1,:,:,:].max(), aspect = 'auto')) #uy
            self.__in_images.append(ah_in[0][2].imshow(self.__input[3,:,self.__n_slice,:], origin='lower', vmin=self.__label[2,:,:,:].min(), vmax=self.__label[2,:,:,:].max(), aspect = 'auto')) #uz
            ah_in[2][0].set_title('Terrain')
            ah_in[0][0].set_title('Ux in')
            ah_in[0][1].set_title('Uy in')
            ah_in[0][2].set_title('Uz in')
            fh_in.colorbar(self.__in_images[1], ax=ah_in[0][0])
            fh_in.colorbar(self.__in_images[2], ax=ah_in[0][1])
            fh_in.colorbar(self.__in_images[3], ax=ah_in[0][2])
            fh_in.colorbar(self.__in_images[0], ax=ah_in[2][0])

            # plot the label data
            self.__out_images.append(ah_in[1][0].imshow(self.__label[0,:,self.__n_slice,:], origin='lower', vmin=self.__label[0,:,:,:].min(), vmax=self.__label[0,:,:,:].max(), aspect = 'auto')) #ux
            self.__out_images.append(ah_in[1][1].imshow(self.__label[1,:,self.__n_slice,:], origin='lower', vmin=self.__label[1,:,:,:].min(), vmax=self.__label[1,:,:,:].max(), aspect = 'auto')) #uy
            self.__out_images.append(ah_in[1][2].imshow(self.__label[2,:,self.__n_slice,:], origin='lower', vmin=self.__label[2,:,:,:].min(), vmax=self.__label[2,:,:,:].max(), aspect = 'auto')) #uz
            try:
                self.__out_images.append(ah_in[2][1].imshow(self.__label[3,:,self.__n_slice,:], origin='lower', vmin=self.__label[3,:,:,:].min(), vmax=self.__label[3,:,:,:].max(), aspect = 'auto')) #turbulence viscosity
                fh_in.colorbar(self.__out_images[3], ax=ah_in[2][1])
            except:
                print('INFO: Turbulence viscosity not present as a label')
            ah_in[1][0].set_title('Ux label')
            ah_in[1][1].set_title('Uy label')
            ah_in[1][2].set_title('Uz label')
            ah_in[2][1].set_title('Turb. Viscosity label')
            fh_in.colorbar(self.__out_images[0], ax=ah_in[1][0])
            fh_in.colorbar(self.__out_images[1], ax=ah_in[1][1])
            fh_in.colorbar(self.__out_images[2], ax=ah_in[1][2])

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
            # 2D data
            fh_in, ah_in = plt.subplots(3, 2)

            h_ux_in = ah_in[0][0].imshow(self.__input[1,:,:], origin='lower', vmin=self.__label[0,:,:].min(), vmax=self.__label[0,:,:].max())
            h_uz_in = ah_in[0][1].imshow(self.__input[2,:,:], origin='lower', vmin=self.__label[1,:,:].min(), vmax=self.__label[1,:,:].max())
            ah_in[0][0].set_title('Ux in')
            ah_in[0][1].set_title('Uz in')
            fh_in.colorbar(h_ux_in, ax=ah_in[0][0])
            fh_in.colorbar(h_uz_in, ax=ah_in[0][1])

            h_ux_in = ah_in[1][0].imshow(self.__label[0,:,:], origin='lower', vmin=self.__label[0,:,:].min(), vmax=self.__label[0,:,:].max())
            h_uz_in = ah_in[1][1].imshow(self.__label[1,:,:], origin='lower', vmin=self.__label[1,:,:].min(), vmax=self.__label[1,:,:].max())
            ah_in[1][0].set_title('Ux label')
            ah_in[1][1].set_title('Uz label')
            fh_in.colorbar(h_ux_in, ax=ah_in[1][0])
            fh_in.colorbar(h_uz_in, ax=ah_in[1][1])

            h_ux_in = ah_in[2][0].imshow(self.__input[0,:,:], origin='lower', vmin=self.__input[0,:,:].min(), vmax=self.__input[0,:,:].max())
            try:
                h_uz_in = ah_in[2][1].imshow(self.__label[2,:,:], origin='lower', vmin=self.__label[2,:,:].min(), vmax=self.__label[2,:,:].max())
            except:
                print('INFO: Turbulence viscosity not present as a label')
            ah_in[2][0].set_title('Terrain')
            ah_in[2][1].set_title('Turbulence viscosity label')
            fh_in.colorbar(h_ux_in, ax=ah_in[2][0])
            fh_in.colorbar(h_uz_in, ax=ah_in[2][1])

        plt.show()

    def plot_prediction(self):
        if (len(list(self.__label.size())) > 3):
            use_turbulence = self.__label.shape[0] > 3
            fh_in, ah_in = plt.subplots(3, self.__label.shape[0])
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.12)

            title = ['Ux', 'Uy','Uz', 'Turbulence']
            for i in range(self.__label.shape[0]):
                self.__out_images.append(ah_in[0][i].imshow(self.__label[i,:,self.__n_slice,:], origin='lower', vmin=self.__label[i,:,:,:].min(), vmax=self.__label[i,:,:,:].max(), aspect = 'auto'))
                self.__in_images.append(ah_in[1][i].imshow(self.__input[i,:,self.__n_slice,:], origin='lower', vmin=self.__label[i,:,:,:].min(), vmax=self.__label[i,:,:,:].max(), aspect = 'auto'))
                self.__error_images.append(ah_in[2][i].imshow(self.__error[i,:,self.__n_slice,:], origin='lower', vmin=self.__error[i,:,:,:].min(), vmax=self.__error[i,:,:,:].max(), aspect = 'auto'))
                ah_in[0][i].set_title(title[i] + ' Label')
                ah_in[1][i].set_title(title[i] + ' Prediction')
                ah_in[2][i].set_title(title[i] + ' Error')
                fh_in.colorbar(self.__out_images[i], ax=ah_in[0][i])
                fh_in.colorbar(self.__in_images[i], ax=ah_in[1][i])
                fh_in.colorbar(self.__error_images[i], ax=ah_in[2][i])

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
            fh_in, ah_in = plt.subplots(3, self.__label.shape[0])

            title = ['Ux', 'Uz', 'Turbulence']
            for i in range(self.__label.shape[0]):
                self.__in_images.append(ah_in[0][i].imshow(self.__label[i,:,:], origin='lower', vmin=self.__label[i,:,:].min(), vmax=self.__label[i,:,:].max(), aspect = 'auto'))
                self.__out_images.append(ah_in[1][i].imshow(self.__input[i,:,:], origin='lower', vmin=self.__label[i,:,:].min(), vmax=self.__label[i,:,:].max(), aspect = 'auto'))
                self.__error_images.append(ah_in[2][i].imshow(self.__error[i,:,:], origin='lower', vmin=self.__error[i,:,:].min(), vmax=self.__error[i,:,:].max(), aspect = 'auto'))
                ah_in[0][i].set_title(title[i] + ' Label')
                ah_in[1][i].set_title(title[i] + ' Prediction')
                ah_in[2][i].set_title(title[i] + ' Error')
                fh_in.colorbar(self.__in_images[i], ax=ah_in[0][i])
                fh_in.colorbar(self.__out_images[i], ax=ah_in[1][i])
                fh_in.colorbar(self.__error_images[i], ax=ah_in[2][i])

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

def plot_prediction(output, label):
    instance = PlotUtils(output, label, 1)
    instance.plot_prediction()