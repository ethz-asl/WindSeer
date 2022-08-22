try:
    from mayavi import mlab
    from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel
    from mayavi.core.api import PipelineBase
    from traits.api import HasTraits, Range, Instance, on_trait_change, Enum
    from traitsui.api import View, Item, VGroup, HGroup
    mayavi_available = True
except ImportError:
    print('mayavi could not get imported, disabling plotting with mayavi. Refer to the README for install instructions')
    mayavi_available = False

import numpy as np
import scipy

def mlab_plot_terrain(terrain, mode='blocks', uniform_color=False, figure=None):
    '''
    Plot the terrain into the current figure

    Terrain Modes:
      blocks: Blocks with the extend of the actual grid cells
      mesh: Linearly interpolated mesh of the terrain with the control points at the center location of the cell
    '''
    if mayavi_available:
        # convert to numpy and permute axis such that the order is: [x,y,z]
        terrain_np = terrain.cpu().squeeze().permute(2,1,0).numpy()

        Z = (terrain_np == 0).sum(axis=2).astype(np.float)

        keep_idx = (terrain_np == 0)[:, :, 0]
        Z[~keep_idx] = np.NaN

        terrain_modes = ['mesh', 'blocks']
        if mode == terrain_modes[0]:
            indices = np.indices(terrain.cpu().squeeze().numpy().shape)
            X = indices[0, :, :, 0]
            Y = indices[1, :, :, 0]

            X = np.concatenate([X, np.flip(X, axis=0), X[0, None, :]], axis=0)
            Y = np.concatenate([Y, np.flip(Y, axis=0), Y[0, None, :]], axis=0)
            Z = np.concatenate([Z, np.flip(Z * 0.0, axis=0), Z[0, None, :]], axis=0)

            X = np.concatenate([X[:, 0, None], X, X[:,-1, None]], axis=1)
            Y = np.concatenate([Y[:, 0, None], Y, Y[:,-1, None]], axis=1)
            Z = np.concatenate([Z[:, 0, None] * 0.0, Z, Z[:,-1, None] * 0.0], axis=1)

            if uniform_color:
                terr = mlab.mesh(X, Y, Z, representation='surface', mode='cube', color=(160.0/255.0 ,82.0/255.0 ,45.0/255.0), figure=figure)

            else:
                terr = mlab.mesh(X, Y, Z, representation='surface', mode='cube', figure=figure, colormap='terrain')

        elif mode == terrain_modes[1]:
            if uniform_color:
                terr = mlab.barchart(Z, color=(160.0/255.0 ,82.0/255.0 ,45.0/255.0), figure=figure, auto_scale=False, lateral_scale=1.0)
            else:
                terr = mlab.barchart(Z, figure=figure, colormap='terrain')

        else:
            raise ValueError('Unknown terrain plotting mode: ' + str(mode) + ' supported: ' + str(terrain_modes))

        return terr

def mlab_plot_slice(title, input_data, terrain, terrain_mode='blocks', terrain_uniform_color=False, prediction_channels=None, blocking=True, white_background=True):
    '''
    Plot the data using the image_plane_widgets and traits to modify the slice direction and number, as well as the displayed channel
    '''
    if mayavi_available:
        if input_data.shape[1] != input_data.shape[2] or input_data.shape[1] != input_data.shape[3]:
            raise ValueError('Only data with the same number of elements in each spatial dimension supported')

        class VisualizationWorker(HasTraits):
            slice = Range(0, input_data.shape[1]-1, 0)
            channel = Range(0, input_data.shape[0]-1, 0)
            direction = Enum('x_axes', 'y_axes', 'z_axes')
            scene = Instance(MlabSceneModel, ())
            data = input_data
            plane_orientation_list = ['x_axes', 'y_axes', 'z_axes']
            labels = prediction_channels

            ipw = Instance(PipelineBase)

            def __init__(self):
                HasTraits.__init__(self)

            @on_trait_change('scene.activated')
            def plot(self):
                if white_background:
                    self.scene.scene.background = (1, 1, 1)
                    self.scene.scene.foreground = (0, 0, 0)

                terrain_shape = terrain.squeeze().shape

                self.t = mlab_plot_terrain(terrain, terrain_mode, terrain_uniform_color, figure=self.scene.mayavi_scene)
                self.scalar = mlab.pipeline.scalar_field(self.data[0])
                self.scalar.origin = [0, 0, 0.5] # shift by 0.5 such that the slice is in the middle of the cell and not at the bottom

                self.ipw = mlab.pipeline.image_plane_widget(
                            self.scalar,
                            plane_orientation='x_axes',
                            slice_index=0,
                            figure=self.scene.mayavi_scene)

                self.o = mlab.outline(extent=[0, terrain_shape[2]-1, 0, terrain_shape[1]-1, 0, terrain_shape[0]-1])

                self.colorbar()

            def colorbar(self):
                if self.labels is not None:
                    self.scene.mlab.scalarbar(self.ipw, title=title + ' ' + self.labels[self.channel], label_fmt='%.1f')
                else:
                    self.scene.mlab.scalarbar(self.ipw, title=title + ' Channel ' + str(self.channel), label_fmt='%.1f')

            @on_trait_change('channel')
            def channel_changed(self):
                self.ipw.mlab_source.scalars = self.data[self.channel]
                self.colorbar()

            @on_trait_change('slice')
            def slice_changed(self):
                self.ipw.widgets[0].slice_index = self.slice

            @on_trait_change('direction')
            def direction_changed(self):
                self.ipw.widgets[0].plane_orientation = self.direction
                self.ipw.widgets[0].slice_index = self.slice

            view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene)),
                        VGroup('_', 'channel', 'slice', 'direction'),
                        resizable=True,
                        )

        vis = VisualizationWorker()

        ui = vis.edit_traits()

        if blocking:
            mlab.show()

        return ui

def mlab_plot_trajectories(trajectories, terrain, terrain_mode='blocks', terrain_uniform_color=False, blocking=True,
                           white_background=True):
    '''
    Plotting the trajectories of all flights
    '''
    if mayavi_available:
        if white_background:
            fig = mlab.figure(fgcolor=(0., 0., 0.), bgcolor=(1, 1, 1))
        else:
            fig = mlab.figure()

        mlab_plot_terrain(terrain, terrain_mode, True)

        for traj in trajectories:
            mlab.plot3d(traj['x'], traj['y'], traj['z'])

#         mlab.view(azimuth=-10,
#                   elevation=80,
#                       distance=160,
#                       focalpoint=[32,32,25])

        if blocking:
            mlab.show()

def mlab_plot_measurements(measurements, mask, terrain, terrain_mode='blocks', terrain_uniform_color=False, blocking=True,
                           white_background=True):
    '''
    Visualize the measurements using mayavi
    The inputs are assumed to be torch tensors.
    '''
    if mayavi_available:
        measurements_np = measurements.cpu().squeeze().numpy()
        mask_np = terrain.cpu().squeeze().numpy()
        measurement_idx = mask.squeeze().nonzero(as_tuple=False).cpu().numpy()

        if white_background:
            fig = mlab.figure(fgcolor=(0., 0., 0.), bgcolor=(1, 1, 1))
        else:
            fig = mlab.figure()

        mlab_plot_terrain(terrain, terrain_mode, terrain_uniform_color)

        if measurements_np.shape[0] == 3:
            wind_vel = measurements_np[:,measurement_idx[:, 0], measurement_idx[:, 1], measurement_idx[:, 2]]
        elif measurements_np.shape[0] == 2:
            wind_vel = np.zeros((3, measurement_idx.shape[0]))
            wind_vel[:2] = measurements_np[:,measurement_idx[:, 0], measurement_idx[:, 1], measurement_idx[:, 2]]

        else:
            raise ValueError('Unsupported number of channels in the measurements tensor')

        mlab.quiver3d(measurement_idx[:, 2], measurement_idx[:, 1], measurement_idx[:, 0], wind_vel[0], wind_vel[1], wind_vel[2] * 11.5/15.5, color=(0.6,0,0), scale_factor=1.5)

        mlab.outline(extent=[0, mask_np.shape[2]-1, 0, mask_np.shape[1]-1, 0, mask_np.shape[0]-1])

#         mlab.view(azimuth=210,
#                   elevation=70,
#                       distance=180,
#                       focalpoint=[32,32,25])

        if blocking:
            mlab.show()

def mlab_plot_prediction(prediction, terrain, terrain_mode='blocks', terrain_uniform_color=False,
                         prediction_channels=None, blocking=True, white_background=True, quiver_mask_points=100):
    '''
    Visualize the prediction using mayavi
    The inputs are assumed to be torch tensors.
    '''
    if mayavi_available:
        prediction_np = prediction.cpu().squeeze().permute(0,3,2,1).numpy()
        terrain_shape = terrain.squeeze().shape

        # quiver slice plot
        if white_background:
            mlab.figure(fgcolor=(0., 0., 0.), bgcolor=(1, 1, 1))
        else:
            mlab.figure()

        mlab_plot_terrain(terrain, terrain_mode, terrain_uniform_color)
        mlab.outline(extent=[0, terrain_shape[2]-1, 0, terrain_shape[1]-1, 0, terrain_shape[0]-1])

        src = mlab.pipeline.vector_field(prediction_np[0], prediction_np[1], prediction_np[2])
        ret = mlab.pipeline.vector_cut_plane(src, mask_points=2, scale_factor=2.5, resolution=50, view_controls=False, mode='arrow')
        ret.implicit_plane.normal = np.array([np.sqrt(0.5), np.sqrt(0.5), 0])

#         mlab.view(azimuth=38,
#                   elevation=88,
#                       distance=170,
#                       focalpoint=[32,32,25])

        # quiver plot
        if white_background:
            mlab.figure(fgcolor=(0., 0., 0.), bgcolor=(1, 1, 1))
        else:
            mlab.figure()

        mlab_plot_terrain(terrain, terrain_mode, terrain_uniform_color)
        mlab.outline(extent=[0, terrain_shape[2]-1, 0, terrain_shape[1]-1, 0, terrain_shape[0]-1])

        mlab.quiver3d(prediction_np[0], prediction_np[1], prediction_np[2], mask_points = quiver_mask_points, mode='arrow')



        # slice plot
        if prediction_np.shape[1] == prediction_np.shape[2] and prediction_np.shape[1] == prediction_np.shape[3]:
            ui = mlab_plot_slice('Prediction', prediction_np, terrain, terrain_mode, terrain_uniform_color, prediction_channels, False, white_background)
        else:
            ui = []

        mlab.outline(extent=[0, terrain_shape[2]-1, 0, terrain_shape[1]-1, 0, terrain_shape[0]-1])

        if blocking:
            mlab.show()

#         mlab_animate_rotate(True)
        return [ui]

def mlab_plot_error(error, terrain, error_mode='norm', terrain_mode='blocks', terrain_uniform_color=False, prediction_channels=None, blocking=True, white_background=True):
    '''
    Visualize the prediction error using mayavi
    The inputs are assumed to be torch tensors.

    Error Modes:
      norm: Compute the norm across all channels
      channels: Display one figure per channel
    '''
    if mayavi_available:
        # error clouds
        error_np = error.cpu().squeeze().permute(0,3,2,1).numpy()
        terrain_shape = terrain.squeeze().shape

        error_modes = ['norm', 'channels']
        if error_mode == error_modes[0]:
            if white_background:
                mlab.figure(fgcolor=(0., 0., 0.), bgcolor=(1, 1, 1))
            else:
                mlab.figure()
            # take the norm across all channels
            mlab_plot_terrain(terrain, terrain_mode, terrain_uniform_color)

            error_norm = np.linalg.norm(error_np, axis=0)
            scalar = mlab.pipeline.scalar_field(error_norm)
            scalar.origin = [0,0,0]
            vol = mlab.pipeline.volume(scalar, vmin=1.0)

            mlab.outline(extent=[0, terrain_shape[2]-1, 0, terrain_shape[1]-1, 0, terrain_shape[0]-1])

            mlab.colorbar(vol, title='Prediction Error Norm [m/s]', label_fmt='%.1f')

#             mlab.view(azimuth=210,
#                   elevation=70,
#                   distance=180,
#                   focalpoint=[32,32,25])
#             mlab_animate_rotate(False)


        elif error_mode == error_modes[1]:
            # one figure per channel
            for i in range(error_np.shape[0]):
                if white_background:
                    mlab.figure(fgcolor=(0., 0., 0.), bgcolor=(1, 1, 1))
                else:
                    mlab.figure()

                mlab_plot_terrain(terrain, terrain_mode, terrain_uniform_color)

                scalar = mlab.pipeline.scalar_field(np.linalg.norm(error_np[i], axis=0))
                scalar.origin = [0,0,0]
                vol = mlab.pipeline.volume(scalar, vmin=np.abs(error_np[i]).max()*0.1)

                mlab.outline(extent=[0, terrain_shape[2]-1, 0, terrain_shape[1]-1, 0, terrain_shape[0]-1])

                if prediction_channels is not None:
                    mlab.scalarbar(vol, title='Prediction Error ' + prediction_channels[i], label_fmt='%.1f')
                else:
                    mlab.scalarbar(vol, title='Prediction Error Channel ' + str(i), label_fmt='%.1f')

        else:
            raise ValueError('Unknown error plotting mode: ' + str(error_mode) + ' supported: ' + str(error_modes))

        if error_np.shape[1] == error_np.shape[2] and error_np.shape[1] == error_np.shape[3]:
            ui = mlab_plot_slice('Prediction Error', error_np, terrain, terrain_mode, terrain_uniform_color, prediction_channels, False, white_background)
        else:
            ui = []

        if blocking:
            mlab.show()

        return [ui]

def mlab_plot_uncertainty(uncertainty, terrain, uncertainty_mode=0, terrain_mode=0,
                          terrain_uniform_color=False, prediction_channels=None, blocking=True, white_background=True):
    '''
    Visualize the prediction uncertainty using mayavi
    The inputs are assumed to be torch tensors.

    Uncertainty Modes:
      norm: Compute the norm across all channels
      channels: Display one figure per channel
    '''
    if mayavi_available:
        # error clouds
        uncertainty_np = uncertainty.cpu().squeeze().permute(0,3,2,1).numpy()
        terrain_shape = terrain.squeeze().shape

        uncertainty_modes = ['norm', 'channels']
        if uncertainty_mode == uncertainty_modes[0]:
            if white_background:
                mlab.figure(fgcolor=(0., 0., 0.), bgcolor=(1, 1, 1))
            else:
                mlab.figure()

            # take the norm across all channels
            mlab_plot_terrain(terrain, terrain_mode, terrain_uniform_color)

            scalar = mlab.pipeline.scalar_field(np.linalg.norm(np.exp(uncertainty_np), axis=0))
            scalar.origin = [0,0,0]
            vol = mlab.pipeline.volume(scalar, vmax=uncertainty_np.max()*0.5)

            mlab.outline(extent=[0, terrain_shape[2]-1, 0, terrain_shape[1]-1, 0, terrain_shape[0]-1])

            mlab.scalarbar(vol, title='Uncertainty', label_fmt='%.2f')

        elif uncertainty_mode == uncertainty_modes[1]:
            # one figure per channel
            for i in range(uncertainty_np.shape[0]):
                if white_background:
                    mlab.figure(fgcolor=(0., 0., 0.), bgcolor=(1, 1, 1))
                else:
                    mlab.figure()

                mlab_plot_terrain(terrain, terrain_mode, terrain_uniform_color)

                scalar = mlab.pipeline.scalar_field(np.linalg.norm(uncertainty_np[i], axis=0))
                scalar.origin = [0,0,0]
                vol = mlab.pipeline.volume(scalar, vmin=np.abs(uncertainty_np[i]).max()*0.1)

                mlab.outline(extent=[0, terrain_shape[2]-1, 0, terrain_shape[1]-1, 0, terrain_shape[0]-1])

                if prediction_channels is not None:
                    mlab.scalarbar(vol, title='Uncertainty ' + prediction_channels[i], label_fmt='%.1f')
                else:
                    mlab.scalarbar(vol, title='Uncertainty Channel ' + str(i), label_fmt='%.1f')

        else:
            raise ValueError('Unknown uncertainty plotting mode: ' + str(uncertainty_mode) + ' supported: ' + str(uncertainty_modes))

        if uncertainty_np.shape[1] == uncertainty_np.shape[2] and uncertainty_np.shape[1] == uncertainty_np.shape[3]:
            ui = mlab_plot_slice('Uncertainty', uncertainty_np, terrain, terrain_mode, terrain_uniform_color, prediction_channels, False, white_background)
        else:
            ui = []

        if blocking:
            mlab.show()

        return [ui]

def mlab_animate_rotate(save=True, magnification=5):
    '''
    Animate the current figure by rotating it around the z-axis and the current focal point
    '''
    @mlab.animate(delay=50)
    def anim():
        """Animate the b1 box."""
        iter = 0
        while 1:
            current_view = mlab.view()
            mlab.view(azimuth=current_view[0] + 2,
                      elevation=current_view[1] * 0 + 85,
                      distance=current_view[2] * 0 + 150,
                      focalpoint=current_view[3])

            if save:
                mlab.savefig('/tmp/frame_' + str(iter).zfill(3) +'.png', magnification=magnification)
                iter += 1
            yield

    anim()
    mlab.show()
