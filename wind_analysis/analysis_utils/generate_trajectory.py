import numpy as np
from scipy.interpolate import RectBivariateSpline


def generate_trajectory(x, y, z, points_along_traj, terrain, z_above_terrain=False):
    h_above_terrain = 100
    x0 = np.zeros(len(x))
    y0 = np.zeros(len(y))
    z0 = np.zeros(len(z))
    for i in range(0, len(x)):
        x0[i] = x[i] + terrain.x_terr[0]
        y0[i] = y[i] + terrain.y_terr[0]
        if z_above_terrain:
            z0[i] = z[i]
        else:
            interp_spline = RectBivariateSpline(terrain.x_terr, terrain.y_terr, terrain.h_terr)
            z0[i] = z[i] + interp_spline(x0[i], y0[i])

    # Generate points along the trajectory
    x_pts = np.zeros((len(x) - 1, points_along_traj))
    y_pts = np.zeros((len(y) - 1, points_along_traj))
    z_pts = np.zeros((len(z) - 1, points_along_traj))
    for i in range(0, len(x) - 1):
        dist = np.sqrt((x0[i + 1] - x0[i]) ** 2 + (y0[i + 1] - y0[i]) ** 2 + (z0[i + 1] - z0[i]) ** 2)
        n = 1
        for j in range(0, points_along_traj):
            t = n / (points_along_traj + 1)
            x_pts[i][j] = x0[i] + t * (x0[i + 1] - x0[i])
            y_pts[i][j] = y0[i] + t * (y0[i + 1] - y0[i])
            z_pts[i][j] = z0[i] + t * (z0[i + 1] - z0[i])
            n += 1
    return x_pts, y_pts, z_pts
