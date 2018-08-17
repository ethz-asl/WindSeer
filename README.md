# intel_wind
 
## Usage

#### Plot the RAM usage of the model over time
1. Uncomment the `@profile` defines in the `test_model.py` script
2. Run the script with `mprof`:
    ` mprof run --interval 0.001 test_model.py `
3. Display the results with:
    `mprof plot`
