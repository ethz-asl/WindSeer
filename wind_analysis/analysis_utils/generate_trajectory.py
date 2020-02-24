import numpy as np
from scipy.interpolate import RectBivariateSpline


def generate_trajectory(x, y, z, uav_speed, dt, terrain, z_above_terrain=False):
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
    x_pts, y_pts, z_pts = [], [], []
    for i in range(0, len(x) - 1):
        dist = np.sqrt((x0[i + 1] - x0[i]) ** 2 + (y0[i + 1] - y0[i]) ** 2 + (z0[i + 1] - z0[i]) ** 2)
        t = dist/uav_speed
        points_along_traj = int(t/dt)
        n = 1
        for j in range(0, points_along_traj):
            t = n / (points_along_traj + 1)
            x_pts.append(x0[i] + t * (x0[i + 1] - x0[i]))
            y_pts.append(y0[i] + t * (y0[i + 1] - y0[i]))
            z_pts.append(z0[i] + t * (z0[i + 1] - z0[i]))
            n += 1
    return np.array(x_pts), np.array(y_pts), np.array(z_pts)
