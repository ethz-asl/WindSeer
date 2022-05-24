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

    full_times = {}
    subdirs = os.listdir(cases_dir)

    for dir in tqdm(subdirs):
        wdir = os.path.join(cases_dir, dir, 'W1')
        if not os.path.isdir(wdir):
            continue

        # First check for simpleFoam2.err
        if not os.path.exists(os.path.join(wdir, "simpleFoam2.err")): continue

        error_size = os.path.getsize(os.path.join(wdir, "simpleFoam2.err"))
        if error_size != 0:
            continue

        t0 = get_clocktime(os.path.join(wdir, "simpleFoam.log"))
        t1 = get_clocktime(os.path.join(wdir, "simpleFoam2.log"))
        full_times[dir] = [t0, t1]

        wind = 2
        wdir = os.path.join(cases_dir, dir, 'W{0}'.format(wind))
        while os.path.isdir(wdir):
            if not os.path.exists(os.path.join(wdir, "simpleFoam.err")): break

            error_size = os.path.getsize(os.path.join(wdir, "simpleFoam.err"))
            if error_size != 0: break
            tn = get_clocktime(os.path.join(wdir, "simpleFoam.log"))
            full_times[dir].append(tn)
            wind += 1
            wdir = os.path.join(cases_dir, dir, 'W{0}'.format(wind))

    return full_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run through a directory and estimate computational times')
    parser.add_argument('cases_dir', type=str, nargs=1, help='Root input directory (with cases as subdirs)')
    args = parser.parse_args()
    full_times = estimate_times(args.cases_dir[0])
    t_full = [t for t in full_times.values()]
    t_all = np.concatenate(t_full).astype(float)
    w0_times = np.array([(t[0] + t[1]) for t in t_full])
    wn_times = np.concatenate([t[2:] for t in t_full]).astype(float)

    print('Total time for {0} runs: {1:0.2f} hours'.format(len(t_all), t_all.sum()/3600.0))
    print('Full average time: {0:0.2f} mins'.format(t_all.mean()/60.0))
    print('W1 average times: {0:0.2f} mins'.format(w0_times.mean() / 60.0))
    print('W2: average times: {0:0.2f} mins'.format(wn_times.mean() / 60.0))




