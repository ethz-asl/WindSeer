# Benchmark Planner

This folder contains the source code for the benchmark planner. This is a RRT* straight line path airmass relative path planner based on OMPL.

## Compiling
### Local Machine
1. First install all the required components by following the instructions in the toplevel README file.

2. Change to the build folder

3. Run cmake:
```
    cmake ../
```

4. Build the code:
```
    make -j 4
```

### Cluster
1. First install all the required components by following the instructions in the toplevel README file.

2. Execute the `cluster_setup.sh` script to load the right modules:
```
    source cluster_setup.sh
```

3. Start an interactive shell to build it on the cluster and not on the login node:
```
    bsub -Is -n 4 -W 4:00 -R "rusage[mem=4096]" bash
```

4. Change to the build folder and run cmake:
```
    cmake -DCMAKE_PREFIX_PATH=~/software/ompl/build/lib/ ../
```

5. Build the code:
```
    make -j 4
```

## Execution
1. Set the required environment variables, where N is the number of available threads:
```
    export OMP_NUM_THREADS={$N}
    export OMP_STACKSIZE=8m
```

2. On the cluster execute the `cluster_setup.sh` script to load the right modules:
```
    source cluster_setup.sh
```

3. If not already done, sample the start and goal poses, for example here 10:
```
    python3 sample_start_goal.py -n 10
```

4. If not already done, generate the prediction database. Instructions are found in the README of the `wind_prediction` folder.

5. Start the planning. Specify the command line arguments if the input files and output files are not in the build folder:
```
    # local machinge:
    ./benchmark
    # cluster
    bsub -n {$N} -W 4:00 -R "rusage[mem=1000]" ./benchmark -i /cluster/work/riner/users/intel/planning/prediction.hdf5 -o /cluster/work/riner/users/intel/planning/planning_results.hdf5
```

## Visualize the results
1. Execute the `plot_planning_results.py` script. If the database is not in the build folder specify the location, for example:
```
    python3 plot_planning_results.py -d /path/planning_results.hdf5
```
