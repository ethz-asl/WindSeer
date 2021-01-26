import argparse
import numpy as np
import torch


import nn_wind_prediction.models as models
import nn_wind_prediction.utils as nn_utils
from analysis_utils import utils
from analysis_utils.WindOptimizer import WindOptimizer

parser = argparse.ArgumentParser(description='Optimise wind speed and direction from COSMO data using observations')
parser.add_argument('config_yaml', help='Input yaml config')
parser.add_argument('-model_dir', dest='model_dir', required=True, help='The directory of the model')
parser.add_argument('-model_version', dest='model_version', default='latest', help='The model version')
parser.add_argument('-p', '--plot', action='store_true', help='Plot the optimization results')
parser.add_argument('-test', '--optimizer_test', action='store_true', help='Loop through all possible optimizers and report the results')
parser.add_argument('-s', '--save', action='store_true', help='Store the results of the optimization')
parser.add_argument('-o', '--out_file', default='/tmp/opt_results.npy', help='Filename of the optimization results')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = nn_utils.BasicParameters(args.config_yaml)

# load the NN
nn_params = nn_utils.EDNNParameters(args.model_dir + '/params.yaml')
NetworkType = getattr(models, nn_params.model['model_type'])
net = NetworkType(**nn_params.model_kwargs())
state_dict = torch.load(args.model_dir + '/' + args.model_version + '.model',
                        map_location=lambda storage, loc: storage)
net.load_state_dict(state_dict)
net.to(device)
net.eval()

if nn_params.data['input_mode'] != 1:
    raise ValueError('Models with an input mode other than 1 are not supported')

config.params['model'] = {}
config.params['model']['input_channels'] = nn_params.data['input_channels']
config.params['model']['label_channels'] = nn_params.data['label_channels']
config.params['model']['autoscale'] = nn_params.data['autoscale']
for key in nn_params.data.keys():
    if 'scaling' in key:
        config.params['model'][key] = nn_params.data[key]

# optimizer
loss_fn = utils.get_loss_fn(config.params['loss'])

wind_opt = WindOptimizer(net, loss_fn, device)

measurement, terrain, ground_truth, mask, scale = utils.load_measurements(config.params['measurements'], config.params['model'])

measurement = measurement.to(device)
terrain = terrain.to(device)
if ground_truth is not None:
    ground_truth = ground_truth[0].to(device)
mask = mask.to(device)

config.params['input_fn']['kwargs']['num_channels'] = len(config.params['model']['input_channels']) - 1
generate_input_fn, initial_parameter = utils.get_input_fn(config.params['input_fn'], measurement, mask)

if args.optimizer_test:
    optimizers = [{'name': 'SimpleStepOptimizer', 'kwargs': {'lr': 5.0, 'lr_decay': 0.01}},
                  {'name': 'Adadelta', 'kwargs': {'lr': 1.0}},
                  {'name': 'Adagrad', 'kwargs': {'lr': 1.0}},
                  {'name': 'Adam', 'kwargs': {'lr': 1.0, 'betas': (.9, .999)}},
                  {'name': 'Adamax', 'kwargs': {'lr': 1.0, 'betas': (.9, .999)}},
                  {'name': 'ASGD', 'kwargs': {'lr': 2.0, 'lambd': 1e-3}},
                  {'name': 'SGD', 'kwargs': {'lr': 2.0, 'momentum': 0.5, 'nesterov': True}},
                 ]

    losses_list = []
    parameter_list = []
    gradients_list = []

    for opt in optimizers:
        # copy the params in this way to initialize the optimization always with the same value
        print('Evaluating optimizer: ' + opt['name'])
        params = torch.Tensor(initial_parameter.cpu().detach().clone()).to(device).requires_grad_()
        config.params['optimizer'] = opt
        prediction, optimization_params, losses, parameter, gradients, input = \
            wind_opt.run(generate_input_fn, params, terrain, measurement, mask, scale, config.params)

        gradients = torch.stack(gradients)
        parameter = torch.stack(parameter)
        gradients_list.append(gradients.view(gradients.shape[0], gradients.shape[1], -1))
        parameter_list.append(parameter.view(parameter.shape[0], parameter.shape[1], -1))
        losses_list.append(losses)

else:
    prediction, optimization_params, losses, parameter, gradients, input = \
        wind_opt.run(generate_input_fn, initial_parameter, terrain, measurement, mask, scale, config.params)

    optimizers = [config.params['optimizer']]

    gradients = torch.stack(gradients)
    parameter = torch.stack(parameter)
    gradients_list = [gradients.view(gradients.shape[0], gradients.shape[1], -1)]
    parameter_list = [parameter.view(parameter.shape[0], parameter.shape[1], -1)]
    losses_list = [losses]

if args.save:
    out = {'optimizers': optimizers,
           'gradients': gradients_list,
           'parameters': parameter_list,
           'losses': losses_list}
    np.save(args.out_file, out)

if args.plot:
    # visualize the results
    import matplotlib.pyplot as plt

    x = np.arange(len(losses_list[0]))

    # plot the losses
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for i, loss in enumerate(losses_list):
        plt.plot(x, loss, label = optimizers[i]['name'])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(loc='best')

    # plot the parameter and the gradients
    max_figs_per_figure = 4
    num_params = parameter_list[0].shape[2]
    num_fig = int(np.ceil(num_params / max_figs_per_figure))

    for co in range(gradients_list[0].shape[1]):
        for i in range(num_fig):
            num_plots = min(max_figs_per_figure, num_params - i * max_figs_per_figure)

            fig, ah = plt.subplots(2, num_plots, squeeze=False)

            if gradients_list[0].shape[1] > 1:
                fig.suptitle('Parameter Corner ' + str(co))
            else:
                fig.suptitle('Parameter for all Corners')

            for j in range(len(gradients_list)):
                for k in range(num_plots):
                    ah[0][k].plot(x, parameter_list[j][:, co, i * max_figs_per_figure + k].numpy(), label = optimizers[j]['name'])
                    ah[1][k].plot(x, gradients_list[j][:, co, i * max_figs_per_figure + k].numpy(), label = optimizers[j]['name'])
                    ah[1][k].set_xlabel('Iteration')
                    ah[0][k].set_title('Parameter ' + str(i * max_figs_per_figure + k))

            ah[0][0].set_ylabel('Parameter Value')
            ah[1][0].set_ylabel('Gradients')
            plt.legend(loc='best')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # plot the corner profiles
    corner_counter = 0
    fig, ah = plt.subplots(input[0, 1:].shape[0], 4)
    fig.suptitle('Velocity Profiles')
    for i in [0, -1]:
        for j in [0, -1]:
            corner_input = input[0, 1: , :, i, j].cpu().detach().numpy()
            height = np.arange(corner_input.shape[1])

            if ground_truth is not None:
                corner_gt = ground_truth[:, :, i, j].cpu().detach().numpy()

            xlabels = ['ux', 'uy', 'uz']

            for k in range(corner_input.shape[0]):
                ah[k][corner_counter].plot(corner_input[k], height, label = 'prediction')
                if ground_truth is not None:
                    ah[k][corner_counter].plot(corner_gt[k], height, label = 'ground truth')

                ah[k][corner_counter].set_xlabel(xlabels[k] + ' corner ' + str(corner_counter))
                ah[k][0].set_ylabel('Height [cells]')

            corner_counter += 1

    if ground_truth is not None:
        plt.legend(loc='best')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if mask is not None:
        mask = mask.squeeze()

    nn_utils.plot_prediction(nn_params.data['label_channels'],
                             prediction = prediction['pred'][0].detach(),
                             label = ground_truth,
                             provided_input_channels = nn_params.data['input_channels'],
                             input = input[0].detach(),
                             terrain = terrain.squeeze(),
                             measurements = measurement[0].detach(),
                             measurements_mask = mask)
