try:
    from mayavi import mlab
    from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel
    from traits.api import HasTraits, Range, Instance, on_trait_change, Enum
    from traitsui.api import View, Item, VGroup, HGroup
    mayavi_available = True
except ImportError:
    print('mayavi could not get imported, disabling plotting with mayavi. Refer to the README for install instructions')
    mayavi_available = False

import numpy as np
import scipy

def mlab_plot_terrain(terrain, mode=0, uniform_color=False, figure=None):
    '''
    Plot the terrain into the current figure
    '''
    if mayavi_available:
        # convert to numpy and permute axis such that the order is: [x,y,z]
        terrain_np = terrain.cpu().squeeze().permute(2,1,0).numpy()

        Z = (terrain_np == 0).sum(axis=2).astype(np.float)

        keep_idx = (terrain_np == 0)[:, :, 0]
        Z[~keep_idx] = np.NaN

        if mode == 0:
            indices = np.indices(terrain.cpu().squeeze().numpy().shape)
            X = indices[0, :, :, 0]
            Y = indices[1, :, :, 0]

            X = np.concatenate([X, np.flip(X, axis=0), X[0, None, :]], axis=0)
            Y = np.concatenate([Y, np.flip(Y, axis=0), Y[0, None, :]], axis=0)
            Z = np.concatenate([Z, np.flip(Z * 0.0 + 1.0, axis=0), Z[0, None, :]], axis=0)

            X = np.concatenate([X[:, 0, None], X, X[:,-1, None]], axis=1)
            Y = np.concatenate([Y[:, 0, None], Y, Y[:,-1, None]], axis=1)
            Z = np.concatenate([Z[:, 0, None] * 0.0  + 1.0, Z, Z[:,-1, None] * 0.0  + 1.0], axis=1)

            if uniform_color:
                mlab.mesh(X, Y, Z, representation='surface', mode='cube', color=(160.0/255.0 ,82.0/255.0 ,45.0/255.0), figure=figure)

            else:
                mlab.mesh(X, Y, Z, representation='surface', mode='cube', figure=figure)

        elif mode == 1:
            if uniform_color:
                mlab.barchart(Z, color=(160.0/255.0 ,82.0/255.0 ,45.0/255.0), figure=figure, auto_scale=False, lateral_scale=1.0)
            else:
                mlab.barchart(Z, figure=figure)

        else:
            raise ValueError('Unknown terrain plotting mode')

def mlab_plot_slice(title, input_data, terrain, terrain_mode=0, terrain_uniform_color=False, prediction_channels=None, blocking=True):
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

            view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene)),
                        VGroup('_', 'channel', 'slice', 'direction'),
                        resizable=True,
                        )

            def __init__(self):
                HasTraits.__init__(self)
                self.t = mlab_plot_terrain(terrain, terrain_mode, terrain_uniform_color, figure=self.scene.mayavi_scene)
                self.p = mlab.pipeline.image_plane_widget(
                            mlab.pipeline.scalar_field(self.data[0]),
                            plane_orientation='x_axes',
                            slice_index=0,
                            figure=self.scene.mayavi_scene)

            def colorbar(self):
                if self.labels is not None:
                    self.scene.mlab.scalarbar(self.p, title=title + ' ' + self.labels[self.channel], label_fmt='%.1f')
                else:
                    self.scene.mlab.scalarbar(self.p, title=title + ' Channel ' + str(self.channel), label_fmt='%.1f')

            @on_trait_change('channel')
            def channel_changed(self):
                self.p.mlab_source.scalars = self.data[self.channel]
                self.colorbar()

            @on_trait_change('slice')
            def slice_changed(self):
                self.p.widgets[0].slice_index = self.slice

            @on_trait_change('direction')
            def direction_changed(self):
                self.p.widgets[0].plane_orientation = self.direction
                self.p.widgets[0].slice_index = self.slice

        vis = VisualizationWorker()
        vis.edit_traits()
        vis.colorbar()

        if blocking:
            mlab.show()

def mlab_plot_measurements(measurements, mask, terrain, terrain_mode=0, terrain_uniform_color=False, blocking=True):
    '''
    Visualize the measurements using mayavi
    The inputs are assumed to be torch tensors.
    '''
    if mayavi_available:
        measurements_np = measurements.cpu().squeeze().numpy()
        mask_np = terrain.cpu().squeeze().numpy()
        measurement_idx = mask.squeeze().nonzero(as_tuple=False).cpu().numpy()

        mlab.figure()
        mlab_plot_terrain(terrain, terrain_mode, terrain_uniform_color)

        if measurements_np.shape[0] == 3:
            wind_vel = measurements_np[:,measurement_idx[:, 0], measurement_idx[:, 1], measurement_idx[:, 2]]
        elif measurements_np.shape[0] == 2:
            wind_vel = np.zeros((3, measurement_idx.shape[0]))
            wind_vel[:2] = measurements_np[:,measurement_idx[:, 0], measurement_idx[:, 1], measurement_idx[:, 2]]

        else:
            raise ValueError('Unsupported number of channels in the measurements tensor')

        mlab.quiver3d(measurement_idx[:, 2], measurement_idx[:, 1], measurement_idx[:, 0], wind_vel[0], wind_vel[1], wind_vel[2])

        if blocking:
            mlab.show()

def mlab_plot_prediction(prediction, terrain, terrain_mode=0, terrain_uniform_color=False, prediction_channels=None, blocking=True):
    '''
    Visualize the prediction using mayavi
    The inputs are assumed to be torch tensors.
    '''
    if mayavi_available:
        prediction_np = prediction.cpu().squeeze().permute(0,3,2,1).numpy()

        mlab_plot_slice('Prediction', prediction_np, terrain, terrain_mode, terrain_uniform_color, prediction_channels, False)

        if blocking:
            mlab.show()

def mlab_plot_error(error, terrain, error_mode=0, terrain_mode=0, terrain_uniform_color=False, prediction_channels=None, blocking=True):
    '''
    Visualize the prediction error using mayavi
    The inputs are assumed to be torch tensors.
    '''
    if mayavi_available:
        # error clouds
        error_np = error.cpu().squeeze().permute(0,3,2,1).numpy()

        if error_mode == 0:
            # take the norm across all channels
            mlab.figure()
            mlab_plot_terrain(terrain, terrain_mode, terrain_uniform_color)

            vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(np.linalg.norm(error_np, axis=0)),
                                       vmin=error_np.max()*0.1)

            mlab.scalarbar(vol, title='Prediction Error Norm [m/s]', label_fmt='%.1f')

        elif error_mode == 1:
            # one figure per channel
            for i in range(error_np.shape[0]):
                mlab.figure()
                mlab_plot_terrain(terrain, terrain_mode, terrain_uniform_color)

                vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(np.abs(error_np[i])), vmin=np.abs(error_np[i]).max()*0.1)

                if prediction_channels is not None:
                    mlab.scalarbar(vol, title='Absolute Prediction Error ' + prediction_channels[i], label_fmt='%.1f')
                else:
                    mlab.scalarbar(vol, title='Absolute Prediction Error Channel ' + str(i), label_fmt='%.1f')

        else:
            raise ValueError('Unknown error plotting mode')

        mlab_plot_slice('Prediction Error', error_np, terrain, terrain_mode, terrain_uniform_color, prediction_channels, False)

        if blocking:
            mlab.show()


def mlab_plot_uncertainty():
    print('TODO')
