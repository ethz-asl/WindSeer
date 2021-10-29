# This script load the required modules on the cluster
# and sets the necessary environment variables

#modules
module load gcc/4.8.5
module load eigen
module load hdf5
module load python/3.7.4
module load cuda/11.1.1
module load cudnn/8.1.0.77


# set the stacksize, if not set the benchmark planner will run out of memory and generate a Segfault
export OMP_STACKSIZE=8m
