import numpy as np
import sys


def add_Askervein_measurement_lines(
        ds_file, key, x_inter, y_inter, z_inter, grid_interpolator, measurement_offset
    ):
    '''
    Add the Askervein measurement line information to the dataset hdf5 file.

    Parameters
    ----------
    ds_file : h5py.File
        Dataset file
    key : str
        Scale key
    x_inter : interp1d
        Interpolater object for the local x-coordinates to the cell index
    y_inter : interp1d
        Interpolater object for the local y-coordinates to the cell index
    z_inter : interp1d
        Interpolater object for the local z-coordinates to the cell index
    grid_interpolator : RegularGridInterpolator
        Interpolater object for the terrain height
    measurement_offset : np.array
        Offset of the measurements in the local frame
    '''
    ds_file['lines'].create_group(key)

    # line A
    t = np.linspace(0, 1500, 301)
    start = np.array([414.0, 1080.0]) - measurement_offset  #ASW85
    end = np.array([1262.0, 1975.0]) - measurement_offset  #ANE40
    dir = (end - start)
    dir /= np.linalg.norm(dir)
    positions = np.expand_dims(start, 1) + np.expand_dims(t, 0) * np.expand_dims(dir, 1)
    z = grid_interpolator((positions[1], positions[0]))
    # minimum distance to ht
    idx_center = np.argmin(
        np.linalg.norm(
            positions - np.expand_dims((np.array([984, 1695]) - measurement_offset), 1),
            axis=0
            )
        )
    t -= idx_center * (t[1] - t[0])
    ds_file['lines'][key].create_group('lineA_10m')
    ds_file['lines'][key]['lineA_10m'].create_dataset('x', data=x_inter(positions[0]))
    ds_file['lines'][key]['lineA_10m'].create_dataset('y', data=y_inter(positions[1]))
    ds_file['lines'][key]['lineA_10m'].create_dataset('z', data=z_inter(z + 10.0))
    ds_file['lines'][key]['lineA_10m'].create_dataset('terrain', data=z)
    ds_file['lines'][key]['lineA_10m'].create_dataset('dist', data=t)

    # line AA
    t = np.linspace(0, 1500, 301)
    start = np.array([670.0, 778.0]) - measurement_offset  #AASW90
    end = np.array([1674.0, 1844.0]) - measurement_offset  #AANE60
    dir = (end - start)
    dir /= np.linalg.norm(dir)
    positions = np.expand_dims(start, 1) + np.expand_dims(t, 0) * np.expand_dims(dir, 1)
    z = grid_interpolator((positions[1], positions[0]))
    # minimum distance to ht
    idx_center = np.argmin(
        np.linalg.norm(
            positions - np.expand_dims((np.array([984, 1695]) - measurement_offset), 1),
            axis=0
            )
        )
    t -= idx_center * (t[1] - t[0])
    ds_file['lines'][key].create_group('lineAA_10m')
    ds_file['lines'][key]['lineAA_10m'].create_dataset('x', data=x_inter(positions[0]))
    ds_file['lines'][key]['lineAA_10m'].create_dataset('y', data=y_inter(positions[1]))
    ds_file['lines'][key]['lineAA_10m'].create_dataset('z', data=z_inter(z + 10.0))
    ds_file['lines'][key]['lineAA_10m'].create_dataset('terrain', data=z)
    ds_file['lines'][key]['lineAA_10m'].create_dataset('dist', data=t)

    # line B
    t = np.linspace(0, 2200, 441)
    start = np.array([2237.0, 543.0]) - measurement_offset  #BSE170
    end = np.array([844.0, 1833.0]) - measurement_offset  #BNW20
    dir = (end - start)
    dir /= np.linalg.norm(dir)
    positions = np.expand_dims(start, 1) + np.expand_dims(t, 0) * np.expand_dims(dir, 1)
    z = grid_interpolator((positions[1], positions[0]))
    # minimum distance to ht
    idx_center = np.argmin(
        np.linalg.norm(
            positions - np.expand_dims((np.array([984, 1695]) - measurement_offset), 1),
            axis=0
            )
        )
    t -= idx_center * (t[1] - t[0])
    ds_file['lines'][key].create_group('lineB_10m')
    ds_file['lines'][key]['lineB_10m'].create_dataset('x', data=x_inter(positions[0]))
    ds_file['lines'][key]['lineB_10m'].create_dataset('y', data=y_inter(positions[1]))
    ds_file['lines'][key]['lineB_10m'].create_dataset('z', data=z_inter(z + 10.0))
    ds_file['lines'][key]['lineB_10m'].create_dataset('terrain', data=z)
    ds_file['lines'][key]['lineB_10m'].create_dataset('dist', data=t)


def add_Bolund_measurement_lines(
        ds_file, key, x_inter, y_inter, z_inter, grid_interpolator
    ):
    '''
    Add the Bolund measurement line information to the dataset hdf5 file.

    Parameters
    ----------
    ds_file : h5py.File
        Dataset file
    key : str
        Scale key
    x_inter : interp1d
        Interpolater object for the local x-coordinates to the cell index
    y_inter : interp1d
        Interpolater object for the local y-coordinates to the cell index
    z_inter : interp1d
        Interpolater object for the local z-coordinates to the cell index
    grid_interpolator : RegularGridInterpolator
        Interpolater object for the terrain height
    '''
    ds_file['lines'].create_group(key)

    ds_file['lines'][key].create_group('lineA_2m')
    ds_file['lines'][key].create_group('lineA_5m')
    t = np.linspace(-200, 200, 401)
    x = np.cos(31.0 / 180.0 * np.pi) * t
    y = np.sin(31.0 / 180.0 * np.pi) * t
    z = grid_interpolator((y, x))

    ds_file['lines'][key]['lineA_2m'].create_dataset('x', data=x_inter(x))
    ds_file['lines'][key]['lineA_2m'].create_dataset('y', data=y_inter(y))
    ds_file['lines'][key]['lineA_2m'].create_dataset('z', data=z_inter(z + 2.0))
    ds_file['lines'][key]['lineA_2m'].create_dataset('terrain', data=z)
    ds_file['lines'][key]['lineA_2m'].create_dataset('dist', data=t)
    ds_file['lines'][key]['lineA_5m'].create_dataset('x', data=x_inter(x))
    ds_file['lines'][key]['lineA_5m'].create_dataset('y', data=y_inter(y))
    ds_file['lines'][key]['lineA_5m'].create_dataset('z', data=z_inter(z + 5.0))
    ds_file['lines'][key]['lineA_5m'].create_dataset('terrain', data=z)
    ds_file['lines'][key]['lineA_5m'].create_dataset('dist', data=t)

    ds_file['lines'][key].create_group('lineB_2m')
    ds_file['lines'][key].create_group('lineB_5m')
    x = np.cos(0.0) * t
    y = np.sin(0.0) * t
    z = grid_interpolator((y, x))

    ds_file['lines'][key]['lineB_2m'].create_dataset('x', data=x_inter(x))
    ds_file['lines'][key]['lineB_2m'].create_dataset('y', data=y_inter(y))
    ds_file['lines'][key]['lineB_2m'].create_dataset('z', data=z_inter(z + 2.0))
    ds_file['lines'][key]['lineB_2m'].create_dataset('terrain', data=z)
    ds_file['lines'][key]['lineB_2m'].create_dataset('dist', data=t)
    ds_file['lines'][key]['lineB_5m'].create_dataset('x', data=x_inter(x))
    ds_file['lines'][key]['lineB_5m'].create_dataset('y', data=y_inter(y))
    ds_file['lines'][key]['lineB_5m'].create_dataset('z', data=z_inter(z + 5.0))
    ds_file['lines'][key]['lineB_5m'].create_dataset('terrain', data=z)
    ds_file['lines'][key]['lineB_5m'].create_dataset('dist', data=t)


def add_Perdigao_measurement_lines(
        ds_file, key, x_inter, y_inter, z_inter, terrain_interpolator, measurements_dict
    ):
    '''
    Perdigao measurement line information to the dataset hdf5 file.

    Parameters
    ----------
    ds_file : h5py.File
        Dataset file
    key : str
        Scale key
    x_inter : interp1d
        Interpolater object for the local x-coordinates to the cell index
    y_inter : interp1d
        Interpolater object for the local y-coordinates to the cell index
    z_inter : interp1d
        Interpolater object for the local z-coordinates to the cell index
    terrain_interpolator : RegularGridInterpolator
        Interpolater object for the terrain height
    measurements_dict : dict
        Dictionary containing the measurements
    '''
    ds_file['lines'].create_group(key)

    # line TSE
    t = np.linspace(0, 2500, 501) - 100
    start = measurements_dict['tse01']['pos'][0, :2]
    end = measurements_dict['tse13']['pos'][0, :2]
    dir = (end - start)
    dir /= np.linalg.norm(dir)
    positions = np.expand_dims(start, 1) + np.expand_dims(t, 0) * np.expand_dims(dir, 1)
    z = terrain_interpolator((positions[0], positions[1]))

    # minimum distance to center mast
    idx_center = np.argmin(
        np.linalg.norm(
            positions - np.expand_dims(measurements_dict['tse09']['pos'][0, :2], 1),
            axis=0
            )
        )
    t -= idx_center * (t[1] - t[0]) - 100

    ds_file['lines'][key].create_group('lineTSE_30m')
    ds_file['lines'][key]['lineTSE_30m'].create_dataset('x', data=x_inter(positions[0]))
    ds_file['lines'][key]['lineTSE_30m'].create_dataset('y', data=y_inter(positions[1]))
    ds_file['lines'][key]['lineTSE_30m'].create_dataset('z', data=z_inter(z + 30.0))
    ds_file['lines'][key]['lineTSE_30m'].create_dataset('terrain', data=z)
    ds_file['lines'][key]['lineTSE_30m'].create_dataset('dist', data=t)

    # line TNW
    t = np.linspace(0, 2500, 501) - 200
    start = measurements_dict['tnw01']['pos'][0, :2]
    end = measurements_dict['tnw11']['pos'][0, :2]
    dir = (end - start)
    dir /= np.linalg.norm(dir)
    positions = np.expand_dims(start, 1) + np.expand_dims(t, 0) * np.expand_dims(dir, 1)

    z = terrain_interpolator((positions[0], positions[1]))
    # minimum distance to center mast
    idx_center = np.argmin(
        np.linalg.norm(
            positions - np.expand_dims(measurements_dict['tnw07']['pos'][0, :2], 1),
            axis=0
            )
        )
    t -= idx_center * (t[1] - t[0]) - 200

    ds_file['lines'][key].create_group('lineTNW_20m')
    ds_file['lines'][key]['lineTNW_20m'].create_dataset('x', data=x_inter(positions[0]))
    ds_file['lines'][key]['lineTNW_20m'].create_dataset('y', data=y_inter(positions[1]))
    ds_file['lines'][key]['lineTNW_20m'].create_dataset('z', data=z_inter(z + 20.0))
    ds_file['lines'][key]['lineTNW_20m'].create_dataset('terrain', data=z)
    ds_file['lines'][key]['lineTNW_20m'].create_dataset('dist', data=t)


def get_tower_distances_on_line(line):
    '''
    Get the towers and the respective distances along this line for the measurement campaigns.

    Parameters
    ----------
    line : str
        Key of the measurement line.

    Returns
    -------
    towers : dict
        Dictionary tower names as key and the respective distances in the data
    '''
    if line == 'lineB_5m':
        line_height = 5
        towers = {'M3': 3.2, 'M6': -46.1, 'M7': -66.9, 'M8': 92.0}

    elif line == 'lineA_10m':
        towers = {
            'ASW85': -841,
            'ASW60': -608,
            'ASW50': -492,
            'ASW35': -327,
            'ASW20': -186,
            'ASW10': -96,
            'HT': 0,
            'ANE10': 100,
            'ANE20': 198,
            'ANE40': 393,
            }

    elif line == 'lineTSE_30m':
        towers = {
            'TSE01': -1495,
            'TSE02': -1140,
            'TSE04': -960,
            'TSE06': -630,
            'TSE07*': -480,
            'TSE08*': -265,
            'TSE09': 0,
            'TSE10': 145,
            'TSE11': 220,
            'TSE12*': 355,
            'TSE13': 465,
            }

    elif line == 'lineTNW_20m':
        towers = {
            'TNW01': -1220,
            'TNW02': -990,
            'TNW03*': -830,
            'TNW05': -495,
            'TNW06': -200,
            'TNW07': 0,
            'TNW08': 205,
            'TNW09*': 360,
            'TNW10': 455,
            'TNW11': 570,
            'TNW12': 625,
            'TNW13': 695,
            'TNW14': 785,
            'TNW15': 840,
            }

    else:
        raise ValueError('Unknown measurement line: ' + line)

    return towers
