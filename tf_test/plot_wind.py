import numpy as np
import matplotlib.pyplot as plt
import read_wind_data as rw

def plot_data(wind_data, sp='p', sUx='U:0', sUz='U:2'):
    fh, ah = plt.subplots(2, 1)     # , {'aspect':'equal'})
    fh.set_size_inches([5, 8])

    p = wind_data.get(sp).values.reshape([rw.WINDNZ, rw.WINDNX])
    # x = wind_data.get('Points:0').values.reshape([WINDNZ, WINDNX])
    # z = wind_data.get('Points:2').values.reshape([WINDNZ, WINDNX])
    Ux = wind_data.get(sUx).values.reshape([rw.WINDNZ, rw.WINDNX])
    Uz = wind_data.get(sUz).values.reshape([rw.WINDNZ, rw.WINDNX])

    ah[0].imshow(p, origin='lower')
    ah[1].quiver(Ux, Uz, np.sqrt(Ux**2 + Uz**2))
    ah[1].set_aspect('equal')
    return fh, ah


def plot_input_output(wind_in, wind_out):

    fh, ah = plt.subplots(2, 1)  # , {'aspect':'equal'})
    fh.set_size_inches([6.2, 5.4])
    Ux = wind_out.get('U:0').values.reshape([rw.WINDNZ, rw.WINDNX])
    Uz = wind_out.get('U:2').values.reshape([rw.WINDNZ, rw.WINDNX])
    h_ux_out = ah[0].imshow(Ux, origin='lower')
    h_uz_out = ah[1].imshow(Uz, origin='lower')
    ah[0].set_title('Ux out')
    ah[1].set_title('Uz out')
    fh.colorbar(h_ux_out, ax=ah[0])
    fh.colorbar(h_uz_out, ax=ah[1])

    fh_in, ah_in = plt.subplots(3, 1)
    fh_in.set_size_inches([6.2, 7.6])
    isWind = wind_in.get('isWind').values.reshape([rw.WINDNZ, rw.WINDNX])
    Ux_in = wind_in.get('Ux').values.reshape([rw.WINDNZ, rw.WINDNX])
    Uz_in = wind_in.get('Uz').values.reshape([rw.WINDNZ, rw.WINDNX])
    ah_in[0].imshow(~isWind, origin='lower')
    h_ux_in = ah_in[1].imshow(Ux_in, origin='lower', vmin=Ux.min(), vmax=Ux.max())
    h_uz_in = ah_in[2].imshow(Uz_in, origin='lower', vmin=Uz.min(), vmax=Uz.max())
    ah_in[0].set_title('isTerrain')
    ah_in[1].set_title('Ux in')
    ah_in[2].set_title('Uz in')
    fh_in.colorbar(h_ux_in, ax=ah_in[1])
    fh_in.colorbar(h_uz_in, ax=ah_in[2])

    return [fh_in, fh], [ah_in, ah]


if __name__ == "__main__":
    wind = rw.read_wind_csv('/home/nick/src/intel_wind/openfoam_slicer/csv/hill1_downsized_Y+196W140.csv')
    wind_in = rw.build_input_from_output(wind)

    fig, ax = plot_data(wind)

    # f_in, a_in = plot_data(wind_in, sp='isWind', sUx='Ux', sUz='Uz')
    f_in, a_in = plot_input_output(wind_in, wind)
    plt.show(block=False)
