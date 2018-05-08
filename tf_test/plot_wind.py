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


if __name__ == "__main__":
    wind = rw.read_wind_csv('data/Y+001W150.csv')
    wind_in = rw.build_input_from_output(wind)

    fig, ax = plot_data(wind)

    f_in, a_in = plot_data(wind_in, sp='isWind', sUx='Ux', sUz='Uz')
    plt.show(block=False)
