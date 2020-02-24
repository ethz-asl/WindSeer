import torch
import argparse
from wind_optimiser import WindOptimiser, OptTest, SimpleStepOptimiser
from analysis_utils.wind_optimiser_output import WindOptimiserOutput


parser = argparse.ArgumentParser(description='Optimise wind speed and direction from COSMO data using observations')
parser.add_argument('input_yaml', help='Input yaml config')
parser.add_argument('-n', '--n_steps', type=int, default=200, help='Number of optimisation steps')
args = parser.parse_args()

# Range of different optimisers from torch.optim, and a basic gradient step (SimpleStepOptimiser)
optimisers = [OptTest(SimpleStepOptimiser, {'lr': 5.0, 'lr_decay': 0.01}),
              OptTest(torch.optim.Adadelta, {'lr': 1.0}),
              OptTest(torch.optim.Adagrad, {'lr': 1.0, 'lr_decay': 0.1}),
              OptTest(torch.optim.Adam, {'lr': 1.0, 'betas': (.9, .999)}),
              OptTest(torch.optim.Adamax, {'lr': 1.0, 'betas': (.9, .999)}),
              OptTest(torch.optim.ASGD, {'lr': 2.0, 'lambd': 1e-3}),
              OptTest(torch.optim.SGD, {'lr': 2.0, 'momentum': 0.5, 'nesterov': True}),
              ]

# Create WindOptimiser object using yaml config
wind_opt = WindOptimiser(args.input_yaml)

# TODO: hardcoded flags to be put in config file
original_input = False
optimise_corners = False


# Optimise wind variables using each optimisation method
if optimise_corners:
    all_ov, losses, grads = [], [], []
    for i, o in enumerate(optimisers):
        ov, loss, grad = wind_opt.optimise_wind_variables(o.opt, n=args.n_steps, opt_kwargs=o.kwargs, verbose=False)
        all_ov.append(ov)
        losses.append(loss)
        grads.append(grad)

if wind_opt.flag.test_simulated_data:
    if original_input:
        wind_predictions, losses = wind_opt.get_original_input_prediction()
    elif wind_opt.flag.use_sparse_data:
        wind_predictions, losses, inputs = wind_opt.sparse_data_prediction()
    elif wind_opt.flag.use_trajectory:
        wind_predictions, losses, inputs = wind_opt.cfd_trajectory_optimisation()
if wind_opt.flag.test_flight_data:
    if wind_opt.flag.predict_flight:
        wind_predictions, losses, inputs = wind_opt.flight_prediction()
    if wind_opt.flag.optimise_flight:
        wind_predictions, losses, inputs = wind_opt.flight_optimisation()

# Analyse optimised wind
# wind_opt_output = WindOptimiserOutput(wind_opt, wind_predictions, losses, inputs)
# Plot graphs
# wind_opt_output.plot()
# # Print losses
# wind_opt_output.print_losses()
