import numpy as np
import sys
import os
import argparse
import re
from tqdm import tqdm


def get_clocktime(log_file):
    with open(log_file, 'r') as fh:
        for line in reversed(list(fh)):
            xt = re.search('ClockTime = (\d+)', line)
            if xt is not None:
                return int(xt.group(1))
        return None


def estimate_times(cases_dir):
    # Extract the final ClockTime value (in s) for each converged run
    # Return two dictionaries (converged and un.) with keys of the case run name
    # (e.g. 'batch05_N_11_S15x15'), and then an array of solve times
    # Note that the first two elements are the two-phase solve for W1, then
    # element 2 is W2 and so on until convergence failed.
    converged_times = {}
    unconverged_times = {}
    subdirs = os.listdir(cases_dir)

    for dir in tqdm(subdirs):

        # Always start with W1 - if it did't converge we didn't try the later
        # wind speeds
        wdir = os.path.join(cases_dir, dir, 'W1')
        if not os.path.isdir(wdir):
            continue

        # First check for simpleFoam2.err (this was a file I created)
        if not os.path.exists(os.path.join(wdir, "simpleFoam2.err")): continue

        # If it's an empty file, there was no error and the solution converged
        error_size = os.path.getsize(os.path.join(wdir, "simpleFoam2.err"))

        # Extract the clocktimes. For the W1 case we always did a two-phase
        # solver, so there are actually two times (add them together for the
        # total solve time)
        t0 = get_clocktime(os.path.join(wdir, "simpleFoam.log"))
        t1 = get_clocktime(os.path.join(wdir, "simpleFoam2.log"))
        if error_size != 0:
            # If there was an error, get the two times anyway
            unconverged_times[dir] = [t0, t1]
            continue
        converged_times[dir] = [t0, t1]

        # Now get all the remaining wind speeds
        wind = 2
        wdir = os.path.join(cases_dir, dir, 'W{0}'.format(wind))
        while os.path.isdir(wdir):
            if not os.path.exists(os.path.join(wdir, "simpleFoam.err")):
                continue

            error_size = os.path.getsize(os.path.join(wdir, "simpleFoam.err"))
            tn = get_clocktime(os.path.join(wdir, "simpleFoam.log"))
            if error_size != 0:
                try:
                    unconverged_times[dir].append(tn)
                except KeyError:
                    unconverged_times[dir] = [tn]
            else:
                converged_times[dir].append(tn)
            wind += 1
            wdir = os.path.join(cases_dir, dir, 'W{0}'.format(wind))

    return converged_times, unconverged_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run through a directory and estimate computational times')
    parser.add_argument('cases_dir', type=str, nargs=1, help='Root input directory (with cases as subdirs)')
    args = parser.parse_args()
    converged_times, unconverged_times = estimate_times(args.cases_dir[0])

    print('Stats for {0}'.format(args.cases_dir[0]))

    print('Converged:')
    t_full = [t for t in converged_times.values()]
    t_all = np.concatenate(t_full).astype(float)
    w0_times = np.array([(t[0] + t[1]) for t in t_full])
    wn_times = np.concatenate([t[2:] for t in t_full]).astype(float)

    print('Total time for {0} runs: {1:0.2f} hours'.format(len(t_all), np.nansum(t_all)/3600.0))
    print('Full average time: {0:0.2f} mins'.format(np.nanmean(t_all)/60.0))
    print('W1 average times: {0:0.2f} mins'.format(np.nanmean(w0_times) / 60.0))
    print('W2+ average times: {0:0.2f} mins'.format(np.nanmean(wn_times) / 60.0))

    print('Unconverged:')
    t_full = [t for t in unconverged_times.values()]
    t_all = np.concatenate(t_full).astype(float)

    print('Total time for {0} runs: {1:0.2f} hours'.format(len(t_all), np.nansum(t_all)/3600.0))
    print('Full average time: {0:0.2f} mins'.format(np.nanmean(t_all)/60.0))




