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

        gradients_list.append(torch.stack(gradients))
        parameter_list.append(torch.stack(parameter))
        losses_list.append(losses)

else:
    prediction, optimization_params, losses, parameter, gradients, input = \
        wind_opt.run(generate_input_fn, initial_parameter, terrain, measurement, mask, scale, config.params)

    optimizers = [config.params['optimizer']]
    gradients_list = [torch.stack(gradients)]
    parameter_list = [torch.stack(parameter)]
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

    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for i, loss in enumerate(losses_list):
        plt.plot(x, loss, label = optimizers[i]['name'])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(loc='best')

    fig, ah = plt.subplots(2, gradients_list[0].shape[1])
    fig.patch.set_facecolor('white')
    for j in range(len(gradients_list)):
        for i in range(gradients_list[j].shape[1]):
            ah[0][i].plot(x, parameter_list[j][:, i].numpy(), label = optimizers[j]['name'])
            ah[1][i].plot(x, gradients_list[j][:, i].numpy(), label = optimizers[j]['name'])

            if j == 0:
                ah[0][i].set_title('Parameter' + str(i))
                ah[1][i].set_xlabel('Iteration')

    ah[0][0].set_ylabel('Parameter Value')
    ah[1][0].set_xlabel('Gradients')
    plt.legend(loc='best')

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
