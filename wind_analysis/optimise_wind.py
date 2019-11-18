import torch
import argparse
from wind_optimiser import WindOptimiser, OptTest, SimpleStepOptimiser
from analysis_utils.wind_optimiser_output import WindOptimiserOutput

parser = argparse.ArgumentParser(description='Optimise wind speed and direction from COSMO data using observations')
parser.add_argument('input_yaml', help='Input yaml config')
parser.add_argument('-n', '--n_steps', type=int, default=200, help='Number of optimisation steps')
parser.add_argument('-r', '--rotation', type=float, default=[0.0, 0.0, 0.0, 0.0], help='Initial rotation (rad)')
parser.add_argument('-s', '--scale', type=float, default=[1.0, 1.0, 1.0, 1.0], help='Initial scale')
args = parser.parse_args()

# Create WindOptimiser object using yaml config
wind_opt = WindOptimiser(args.input_yaml)

optimise_wind = wind_opt._cosmo_args.params['optimise_wind']
if optimise_wind:
    # Testing a range of different optimisers from torch.optim, and a basic gradient step (SimpleStepOptimiser)
    optimisers = [
                  OptTest(torch.optim.Adadelta, {'lr': 1.0}),
                  OptTest(torch.optim.Adagrad, {'lr': 1.0, 'lr_decay': 0.1}),
                  OptTest(torch.optim.Adam, {'lr': 1.0, 'betas': (.9, .999)}),
                  OptTest(torch.optim.Adamax, {'lr': 1.0, 'betas': (.9, .999)}),
                  OptTest(torch.optim.ASGD, {'lr': 2.0, 'lambd': 1e-3}),
                  OptTest(torch.optim.SGD, {'lr': 2.0, 'momentum': 0.5, 'nesterov': True}),
                  ]

    # Try each optimisation method
    all_rs, losses, grads = [], [], []
    for i, o in enumerate(optimisers):
        wind_opt.reset_rotation_scale(args.rotation, args.scale)
        rs, loss, grad = wind_opt.optimise_rotation_scale(o.opt, n=args.n_steps, opt_kwargs=o.kwargs, verbose=False)
        all_rs.append(rs)
        losses.append(loss)
        grads.append(grad)

    # Analyse optimised wind
    wind_opt_output = WindOptimiserOutput(wind_opt, optimisers, all_rs, losses, grads)
    # Plot graphs
    wind_opt_output.plot()
    # Print losses
    wind_opt_output.print_losses()


predict_ulog = wind_opt._ulog_args.params['predict_ulog']
if predict_ulog:
    # Get NN output using the sparse ulog data as input
    output_wind = wind_opt.get_ulog_prediction()

    # Interpolate prediction to scattered ulog data locations used for testing
    predicted_ulog_data = wind_opt.get_predicted_interpolated_ulog_data(output_wind)

    # Print metrics
    wind_opt.calculate_metrics(predicted_ulog_data)