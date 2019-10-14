# Wind Prediction
This package contains the utilities to train a neural network for wind prediction and plot its predictions.

## Usage
### Generating the datasets
The CFD pipeline generates a set of csv file. Each flow is an individual csv file. As csv files are slow to parse they are converted to torch tensors before they are used for training. The prerequisite is an uncompressed tar file containing all the tar files for the desired dataset.
Then to convert the csv files to pytorch tensors the `convert_dataset.py` script is used. For example to convert the input.tar dataset in the data folder use:
```
    python3 convert_dataset.py -i data/input.tar -o data/output.tar
```
The output data can be compressed in case of very large datasets but this slows down the dataloading. So unless really required it is not recommended.
Downscaling the already converted dataset can be done using `compress_dataset.py`. For example to downscale a 64x64x64 dataset to 32x32x32 use:
```
    python3 compress_dataset.py -i data/input.tar -o data/output.tar -s_hor 2 -s_ver 2
```

To sample a fixed dataset from an existing dataset use the following script after setting the parameters as desired:
```
    python3 sample_dataset.py
```

### Train the model
Training the model is done using the `train.py` script. It needs a configuration file as an input which defines the dataset, model, and train parameter. Usage:

```
	python3 train.py -y config/example.yaml
```

Submitting a train job on the cluster for 4 hours using 6 cores, 1 GPU and a total of 66 GB of RAM:
```
    bsub -n 6 -W 4:00 -R "rusage[mem=11000, ngpus_excl_p=1]" python3 train.py -y config/example.yaml
```

### Predicting the wind
To predict the wind the `predict.py` script is used. Inside the script it can be configured which dataset/sample/model is used and also if the full dataset should be evaluated.

### Generating a prediction database
To generate a prediction database which can be used in the planning benchmark evaluate the planning performance using the different predictions use the `generate_prediction_database.py` script. In the scipt the models, the input dataset and the output database location need to be specified. Then simply run the script on the local machine or on the cluster. When running it on the cluster make sure you store the output database in /cluster/work/riner/users/intel/ as it can be quite large and exceed the home storage space.

### Testing
The `test_*` scripts can be used to test different classes/parts of the package such as the models or the dataset implementation.

#### Plot the RAM usage of the model over time
1. Uncomment the `@profile` defines in the `test_model.py` script
2. Run the script with `mprof`:
    ` mprof run --interval 0.001 test_model.py `
3. Display the results with:
    `mprof plot`
