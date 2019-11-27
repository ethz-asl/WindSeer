import torch
import argparse
from wind_optimiser import WindOptimiser, OptTest, SimpleStepOptimiser, WindTest
from analysis_utils.wind_optimiser_output import WindOptimiserOutput
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

test_wind = True
if test_wind:
    # Create WindTest object using yaml configuration
    wind_test = WindTest(args.input_yaml)
    # Try each optimisation method
    if wind_test.optimisation_args.params['optimisation_method'] == 0:
        all_ov, losses, grads = [], [], []
        for i, o in enumerate(optimisers):
            ov, loss, grad = wind_test.run_wind_profile_optimisation(o.opt, n=args.n_steps, opt_kwargs=o.kwargs, verbose=False)
            all_ov.append(ov)
            losses.append(loss)
            grads.append(grad)

        # Plots

    elif wind_test.optimisation_args.params['optimisation_method'] == 1:
        losses, percentages = wind_test.run_scattered_optimisation()
        # Plots
        plt.plot(percentages, losses, c='blue')
        plt.title('Scattered points optimisation')
        plt.xlabel('Percentage')
        plt.ylabel('Loss')
        # Save figure
        data_set_name = str(wind_test.data_set_name)
        # pp = PdfPages('analysis_output/' + data_set_name + '_scatt_opt' + '.pdf')
        # pp.savefig(fig)
        plt.savefig('analysis_output/' + data_set_name + '_scatt_opt' + '.pdf')
        plt.show()
    elif wind_test.optimisation_args.params['optimisation_method'] == 2:
        b = wind_test.run_trajectory_optimisation()
        # Plots


optimise_wind = False
if optimise_wind:
    # Create WindOptimiser object using yaml config
    wind_opt = WindOptimiser(args.input_yaml)

    # Try each optimisation method
    all_ov, losses, grads = [], [], []
    for i, o in enumerate(optimisers):
        ov, loss, grad = wind_opt.optimise_wind_variables(o.opt, n=args.n_steps, opt_kwargs=o.kwargs, verbose=False)
        all_ov.append(ov)
        losses.append(loss)
        grads.append(grad)

    # Analyse optimised wind
    wind_opt_output = WindOptimiserOutput(wind_opt, optimisers, all_ov, losses, grads)
    # Plot graphs
    wind_opt_output.plot()
    # Print losses
    wind_opt_output.print_losses()


predict_ulog = False
if predict_ulog:
    # Create WindOptimiser object using yaml config
    wind_opt = WindOptimiser(args.input_yaml)

    # Get NN output using the sparse ulog data as input
    output_wind = wind_opt.get_ulog_prediction()

    # Interpolate prediction to scattered ulog data locations used for testing
    predicted_ulog_data = wind_opt.get_predicted_interpolated_ulog_data(output_wind)

    # Print metrics
    wind_opt.calculate_metrics(predicted_ulog_data)