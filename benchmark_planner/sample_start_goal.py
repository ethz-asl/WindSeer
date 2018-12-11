'''
Sample start and goal configurations and write them to a file (start_goal_configurations.txt) inside the build folder.
Each line contains one pair in the following format:
x_start y_start z_start yaw_start x_goal y_goal z_goal yaw_goal

While x and y are actually expressed in meter z is the fraction of the distance between the terrain and domain height.
'''

#----------------------------------------------------------
# default values
x_min_default = 0
x_max_default = 1100
y_min_default = 0
y_max_default = 1100
z_max_default = 1.0
n_samples_default = 20
#----------------------------------------------------------

import argparse
import math
import random

parser = argparse.ArgumentParser(description='Script to sample start and goal configurations for the planning benchmark')
parser.add_argument('-xmin', type=float, dest='x_min', default=x_min_default, help='Minimum x-coordinate [m]')
parser.add_argument('-xmax', type=float, dest='x_max', default=x_max_default, help='Maximum x-coordinate [m]')
parser.add_argument('-ymin', type=float, dest='y_min', default=y_min_default, help='Minimum y-coordinate [m]')
parser.add_argument('-ymax', type=float, dest='y_max', default=y_max_default, help='Maximum y-coordinate [m]')
parser.add_argument('-zmax', type=float, dest='z_max', default=z_max_default, help='Maximum z-coordinate, fraction of the distance between terrain height and max-domain height []')
parser.add_argument('-n', type=int, dest='n_samples', default=n_samples_default, help='Number of start and goal pairs to sample []')
args = parser.parse_args()


f = open('build/start_goal_configurations.txt', 'w')

generator = random.SystemRandom()

for i in range(args.n_samples):
    f.write('{} {} {} {} {} {} {} {}\n'.format(
        generator.uniform(args.x_min, args.x_max),  # x_start
        generator.uniform(args.y_min, args.y_max),  # y_start
        generator.triangular(0.0, args.z_max, 0.0), # z_start
        generator.uniform(-math.pi, math.pi),       # yaw_start
        generator.uniform(args.x_min, args.x_max),  # x_goal
        generator.uniform(args.y_min, args.y_max),  # y_goal
        generator.triangular(0.0, args.z_max, 0.0), # z_goal
        generator.uniform(-math.pi, math.pi)))      # yaw_goal

f.close()

print('Done, generated {} start/goal pairs'.format(args.n_samples))
