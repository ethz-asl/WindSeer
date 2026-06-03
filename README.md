<img align="right" height="60" src="https://user-images.githubusercontent.com/5248102/126074528-004a32b9-7911-486a-9e79-8b78e6e66fdc.png">

# WindSeer

[![Unit Test](https://github.com/ethz-asl/WindSeer/actions/workflows/unit_test.yaml/badge.svg)](https://github.com/ethz-asl/WindSeer/actions/workflows/unit_test.yaml)

This repository contains the tools to predict the wind using a neural network.

- Achermann, Florian, Thomas Stastny, Bogdan Danciu, Andrey Kolobov, Jen Jen Chung, Roland Siegwart, and Nicholas Lawrance. "WindSeer: real-time volumetric wind prediction over complex terrain aboard a small uncrewed aerial vehicle." Nature Communications 15, no. 1 (2024): 3507. [[paper](https://www.nature.com/articles/s41467-024-47778-4)][[video](https://youtu.be/SRKmbexGg3U?si=W32h9FkS6hVohsdY)]
```
@article{achermann2024windseer,
  title={WindSeer: real-time volumetric wind prediction over complex terrain aboard a small uncrewed aerial vehicle},
  author={Achermann, Florian and Stastny, Thomas and Danciu, Bogdan and Kolobov, Andrey and Chung, Jen Jen and Siegwart, Roland and Lawrance, Nicholas},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={3507},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

If you are viewing a downloaded copy of this repository (ZIP archive), consider obtaining the latest version online from https://github.com/ethz-asl/WindSeer.

## Structure
### Benchmark Planner
The planning benchmark tools used to compare the planning performance using different wind predictions.

### Data Generation
The pipeline to generate the training data to learn the wind prediction.

### Libs
Submodules with alternative optimizers.

### WindSeer
The tools to train and evaluate the networks for the wind prediction.

## Installation
This guide explains how to set up the environment in Ubuntu to make the scripts in the repository run.

1. Python3 (at least version 3.6) is required, usually it is already present. If not install it with:
   `sudo apt-get install python3.6`

2. Install (Cuda)[https://developer.nvidia.com/cuda-zone] if a graphics card is present and (PyTorch)[https://pytorch.org/get-started/locally/] according to your CUDA, Ubuntu, and Python version. At least PyTorch 1.9.0 is required.
   Earlier Pytorch Version have some Cudnn issues with Cuda 11.1.

3. Install the following required python packages:
   `pip3 install scipy sklearn pyproj h5py matplotlib GDAL netCDF4 tqdm pandas tensorboard pyulog interpolation mayavi mpl-scatter-density PyQt5 yapf`

4. Clone the repo and update the submodules with:
   `git clone --recurse-submodules https://github.com/ethz-asl/intel_wind.git`

5. Install the `windseer` package in developer mode. To do so change into the `intel_wind` directory and execute the following command:
   `pip3 install -e .`

6. Install the LV03 projection to pyproj:
   `python3 windseer/proj_definitions/install_ch_defs.py`

7. Install RAdam and Ranger if you want to use the Ranger optimizer. To do so change into respective folders and install the packages using:
   `pip3 install . --user --no-deps`

### llmvlite build errors
If you have trouble installing `llvmlite` on Ubuntu 18.04 (a dependency of `interpolate`), it may be due to a version mismatch. Try this:
   ```
   sudo apt-get install llvm-10-dev
   export LLVM_CONFIG='/usr/bin/llvm-config-10'
   pip3 install interpolation
   ```

## Running Wind Predictions from Terrain Data and a Flight Log

This section describes how to predict the wind field over a terrain using sparse wind measurements extracted from a flight log.

### Inputs

| Input | Format | Description |
|---|---|---|
| Terrain data | GeoTIFF (`.tif`) | Digital elevation model of the flight area |
| Flight log | PX4 ULog (`.ulg`) or pre-processed HDF5 (`.hdf5`) | Flight log containing GPS position, airspeed, and wind measurements |
| Weather model (optional) | NetCDF (`.nc`) | COSMO weather model data for background wind |
| Trained model | Directory | A trained WindSeer neural network |

### Step 1: Create a Configuration File

Copy `configs/example_sparse.yaml` as a starting point and edit the `measurements.log` section:

```yaml
measurements:
    type: 'log'

    log:
        filename: 'path/to/flight_log.ulg'       # .ulg or .hdf5 flight log
        geotiff_file: 'path/to/terrain.tif'       # terrain elevation GeoTIFF
        cosmo_file: 'path/to/cosmo.nc'            # optional COSMO weather data
        filter_window_size: 11    # moving average filter size (must be odd, 0 to disable)
        distance_field: True
        use_cosmo_grid: False
        enforce_grid_size: True
        num_cells: 64             # grid resolution (cells per side)
        alt_offset: 200           # lower domain bound offset below takeoff altitude [m]
        d_horizontal: 1100        # horizontal domain extent [m]
        d_vertical: 700           # vertical domain extent [m]
        t_start: 0                # start time relative to first log timestamp [s]
        t_end: 1000000            # end time relative to first log timestamp [s]
```

The domain is centred on the flight path. `d_horizontal` and `d_vertical` control the physical size of the prediction grid; `num_cells` controls its resolution.

### Step 2: Predict the Wind Field

Run `predict_sparse.py` with the config file and a trained model directory to generate a wind field prediction and visualise it:

```bash
python windseer/predict_sparse.py \
    configs/my_config.yaml \
    -model_dir trained_models/my_model
```

Optional flags:

| Flag | Description |
|---|---|
| `-model_version VERSION` | Use a specific checkpoint (default: latest) |
| `--mayavi` | Generate 3D visualisations using Mayavi |

### Step 3 (Optional): Evaluate Against Flight Measurements

Use `evaluate_flight_data.py` to quantitatively compare the predicted wind field against measured wind along the flight path:

```bash
python windseer/evaluate_flight_data.py \
    configs/my_config.yaml \
    -model_dir trained_models/my_model
```

Optional flags:

| Flag | Description |
|---|---|
| `-model_version VERSION` | Use a specific checkpoint (default: latest) |
| `--paths` | Visualise flight paths with Mayavi |
| `--loiter` | Evaluate using detected loiter circles instead of the full path |

### Quick Test with Example Data

The repository includes sample files in `windseer/test/testdata/` to verify your setup:

```bash
python windseer/predict_sparse.py \
    configs/example_sparse.yaml \
    -model_dir trained_models/my_model
```

The example config already points to `test_hdf5.hdf5` (flight log), `test_geotiff.tif` (terrain), and `test_cosmo.nc` (weather model).
