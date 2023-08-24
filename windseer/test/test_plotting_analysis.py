#!/usr/bin/env python
'''
Testcases for the neural network prediction
'''

import windseer.utils as utils
import windseer.plotting as plotting
import windseer

import numpy as np
import os
import unittest

windseer_path = os.path.dirname(windseer.__file__)
testdata_folder = os.path.join(windseer_path, 'test', 'testdata')
test_file_tiff = os.path.join(testdata_folder, 'test_geotiff.tif')
test_file_hdf5 = os.path.join(testdata_folder, 'test_hdf5.hdf5')
test_file_data = os.path.join(testdata_folder, 'test_cosmo.nc')
test_file_terrain = os.path.join(testdata_folder, 'test_cosmo_terrain.nc')


class TestPlottingAnalysis(unittest.TestCase):

    def setUp(self):
        self.wind_data = utils.extract_wind_data(test_file_hdf5, True)

        self.cosmo_data = utils.extract_cosmo_data(
            test_file_data, self.wind_data['lat'][0], self.wind_data['lon'][0], 0,
            test_file_terrain
            )

        self.plane_pos = np.array([
            self.wind_data['x'], self.wind_data['y'], self.wind_data['alt']
            ])
        self.w_vanes = np.array([
            self.wind_data['we'], self.wind_data['wn'], self.wind_data['wd']
            ])

        x_target, y_target, z_target, z_out, full_block_dist = utils.get_terrain(
            test_file_tiff, [self.wind_data['x'].min(), self.wind_data['x'].max()],
            [self.wind_data['y'].min(), self.wind_data['y'].max()],
            z=None,
            resolution=(64, 64, 64),
            build_block=True,
            plot=False,
            distance_field=True,
            horizontal_overflow=0,
            return_is_wind=True
            )

        self.x_terr = x_target
        self.y_terr = y_target
        self.z_terr = z_target
        self.h_terr = z_out
        self.terrain_tensor = full_block_dist

        vane_lims = plotting.vector_lims(self.w_vanes, axis=0)
        cosmo_lims = plotting.vector_lims(
            np.array([
                self.cosmo_data['wind_x'], self.cosmo_data['wind_y'],
                self.cosmo_data['wind_z']
                ]),
            axis=0
            )
        self.Vlims = (
            0.0, max(vane_lims[1], cosmo_lims[1])
            )  # min(vane_lims[0], cosmo_lims[0])

        terrain_corners = self.h_terr[::self.h_terr.shape[0] -
                                      1, ::self.h_terr.shape[1] - 1]
        self.cosmo_corners = utils.cosmo_corner_wind(
            self.cosmo_data,
            self.z_terr,
            terrain_height=terrain_corners,
            rotate=0,
            scale=1
            )

    def test_plot_wind_3d(self):
        fh, ah = plotting.plot_wind_3d(
            self.plane_pos,
            self.w_vanes,
            self.x_terr,
            self.y_terr,
            self.h_terr,
            self.cosmo_data,
            origin=self.plane_pos[:, 0].flat,
            Vlims=self.Vlims,
            plot_cosmo=True,
            wskip=50
            )

    def test_plot_vertical_profile(self):
        plot_time = (self.wind_data['time'] - self.wind_data['time'][0]) * 1e-6
        cx = int((self.plane_pos[0, 0] - self.x_terr[0]) /
                 (self.x_terr[-1] - self.x_terr[0]) > 0.5)
        cy = int((self.plane_pos[1, 0] - self.y_terr[0]) /
                 (self.y_terr[-1] - self.y_terr[0]) > 0.5)
        fh, ah = plotting.plot_vertical_profile(
            self.z_terr, self.cosmo_corners[:, :, cy, cx], self.w_vanes,
            self.wind_data['alt'], plot_time
            )

    def test_plot_lateral_variation(self):
        plot_time = (self.wind_data['time'] - self.wind_data['time'][0]) * 1e-6
        plotting.plot_lateral_variation(self.w_vanes, self.plane_pos, plot_time)

    def test_plot_wind_estimates(self):
        plot_time = (self.wind_data['time'] - self.wind_data['time'][0]) * 1e-6
        plotting.plot_wind_estimates(
            plot_time, [self.w_vanes], ['Wind estimated'], True
            )


if __name__ == '__main__':
    unittest.main()
