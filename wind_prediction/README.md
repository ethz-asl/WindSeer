# Wind Prediction
This package contains the utilities to train a neural network for wind prediction and plot its predictions.


## Usage
### As a ROS Node
TODO

### Generating the datasets
To convert the csv files to pytorch tensors the `convert_dataset.py` script is used. Downscaling the already converted dataset can be done using `compress_dataset.py`.

### Train the model
Training the model is done using the `train.py` script. It needs a configuration file as an input which defines the dataset, model, and train parameter. Usage:

```
	python3 train.py -y config/example.yaml
```

### Predicting the wind
To predict the wind the `predict.py` script is used. Inside the script it can be configured which dataset/sample/model is used and also if the full dataset should be evaluated.

### Testing
The `test_*` scripts can be used to test different classes/parts of the package such as the models or the dataset implementation.

#### Plot the RAM usage of the model over time
1. Uncomment the `@profile` defines in the `test_model.py` script
2. Run the script with `mprof`:
    ` mprof run --interval 0.001 test_model.py `
3. Display the results with:
    `mprof plot`
