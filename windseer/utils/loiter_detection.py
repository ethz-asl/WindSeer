import numpy as np
from scipy import optimize


def fit_circle(x, y):
    '''
    Fit a circle through a set of points with least squares optimization.

    Parameters
    ----------
    x : np.array
        X-coordinates of the points
    y : np.array
        Y-coordinates of the points

    Returns
    -------
    xc : float
        Center of the fit circle, x-position
    yc : float
        Center of the fit circle, y-position
    R_fit : float
        Radius of the fit circle
    max_error : float
        Maximum error of the circle to a input point
    '''

    def calculate_radius(xc, yc):
        return np.sqrt((x - xc)**2 + (y - yc)**2)

    def radius_error(c):
        Ri = calculate_radius(*c)
        return Ri - Ri.mean()

    def Dradius_error(c):
        xc, yc = c
        derr = np.empty((len(c), x.size))

        Ri = calculate_radius(xc, yc)
        derr[0] = (xc - x) / Ri
        derr[1] = (yc - y) / Ri
        derr = derr - derr.mean(axis=1)[:, np.newaxis]

        return derr

    center_init = np.mean(x), np.mean(y)
    center_fit, ier = optimize.leastsq(
        radius_error, center_init, Dfun=Dradius_error, col_deriv=True
        )

    xc, yc = center_fit
    Ri_fit = calculate_radius(*center_fit)
    R_fit = Ri_fit.mean()
    residual = np.sum((Ri_fit - R_fit)**2)
    max_error = np.abs(Ri_fit - R_fit).max()

    return xc, yc, R_fit, max_error


def detect_loiters(wind_data, config):
    '''
    Detect the loiter patterns in the flight data.

    The following configurations are supported or the loiter detection:
    min_window_time: Lower bound for a time in a loiter pattern [s]
    target_radius: Expected radius of the loiter in meter
    radius_tolerance: Allowed deviation in meter from the target radius
    error_tolerance: Maximum allowed error for the circle fit [m]
    max_climb_rate: Maximum allowed climb rate within the loiter [m/s]
    step: Number of new data points considered when iterating through the data
    max_altitude_change: Maximum allowed altitude deviation within one loiter [m]
    loiter_threshold: Number of revolutions required until a loiter is accepted [-]
    plot_results: Visualize the fit loiters with matplotlib

    Parameters
    ----------
    wind_data : dict
        Dictionary with the wind and trajectory data
    conig : dict
        Dictionary with the settings for the loiter detection

    Returns
    -------
    fit_loiters : list of dict
        List of the fit loiter patterns
    '''

    loiter_threshold = 1.0

    samples_rate = 1e6 / np.diff(wind_data['time']).mean()

    window_size = int(config['min_window_time'] * samples_rate)

    fit_loiters = []
    idx = 0
    while (idx < len(wind_data['x'])):
        xc, yc, R, max_error = fit_circle(
            wind_data['x'][idx:idx + window_size], wind_data['y'][idx:idx + window_size]
            )

        avg_climb_rate = np.polyfit(
            wind_data['time_gps'][idx:idx + window_size],
            wind_data['alt'][idx:idx + window_size], 1
            )[0]
        if (
            np.abs(R - config['target_radius']) < config['radius_tolerance'] and
            np.abs(avg_climb_rate) < config['max_climb_rate']
            ):
            iter = 0
            ws_loiter = window_size + iter * int(config['step'] * samples_rate)
            while (
                np.abs(R - config['target_radius']) < config['radius_tolerance']
                and  # check if the radius is close to the expected loiter radius
                max_error < config['error_tolerance']
                and  # check if the maximum error is below the requested threshold
                (
                    wind_data['alt'][idx:idx + ws_loiter].max() -
                    wind_data['alt'][idx:idx + ws_loiter].min()
                    ) < config['max_altitude_change']
                and  # check if altitude changes too much
                np.abs(avg_climb_rate) < config['max_climb_rate']
                and  # check if the climb rate is too large recently
                idx + ws_loiter < len(wind_data['x'])
                ):

                iter += 1
                ws_loiter = window_size + iter * int(config['step'] * samples_rate)
                xc, yc, R, max_error = fit_circle(
                    wind_data['x'][idx:idx + ws_loiter],
                    wind_data['y'][idx:idx + ws_loiter]
                    )
                avg_climb_rate = np.polyfit(
                    wind_data['time_gps'][idx + ws_loiter - window_size:idx +
                                          ws_loiter],
                    wind_data['alt'][idx + ws_loiter - window_size:idx + ws_loiter], 1
                    )[0]

            # check if at least one full loiter was completed
            dx = np.diff(wind_data['x'][idx:idx + ws_loiter])
            dy = np.diff(wind_data['y'][idx:idx + ws_loiter])

            if np.sum(np.sqrt(dx**2 + dy**2)) / (2.0 * np.pi * config['target_radius']
                                                 ) > config['loiter_threshold']:
                loiter = {
                    'R': R,
                    'xc': xc,
                    'yc': yc,
                    'idx_start': idx,
                    'idx_stop': idx + ws_loiter
                    }

                fit_loiters.append(loiter)

            idx += ws_loiter
        else:
            idx += int(config['step'] * samples_rate)

    if config['plot_results']:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='3d'))
        ax.plot(wind_data['x'], wind_data['y'], wind_data['alt'], color='black')

        for loiter in fit_loiters:
            ax.plot(
                wind_data['x'][loiter['idx_start']:loiter['idx_stop']],
                wind_data['y'][loiter['idx_start']:loiter['idx_stop']],
                wind_data['alt'][loiter['idx_start']:loiter['idx_stop']]
                )

        fig, ax = plt.subplots(3, 1)
        ax[0].plot(wind_data['time_gps'], wind_data['x'], color='black')
        ax[1].plot(wind_data['time_gps'], wind_data['y'], color='black')
        ax[2].plot(wind_data['time_gps'], wind_data['alt'], color='black')
        for loiter in fit_loiters:
            ax[0].plot(
                wind_data['time_gps'][loiter['idx_start']:loiter['idx_stop']],
                wind_data['x'][loiter['idx_start']:loiter['idx_stop']]
                )
            ax[1].plot(
                wind_data['time_gps'][loiter['idx_start']:loiter['idx_stop']],
                wind_data['y'][loiter['idx_start']:loiter['idx_stop']]
                )
            ax[2].plot(
                wind_data['time_gps'][loiter['idx_start']:loiter['idx_stop']],
                wind_data['alt'][loiter['idx_start']:loiter['idx_stop']]
                )

        plt.show()

    return fit_loiters
