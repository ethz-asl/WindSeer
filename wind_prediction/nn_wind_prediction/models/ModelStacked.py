import importlib
import sys
import torch
import torch.nn as nn

import nn_wind_prediction.utils as utils

from .ModelBase import ModelBase

class ModelStacked(ModelBase):
    __default_pass_full_output = False
    __default_submodel_terrain_mask = False
    __default_use_terrain_mask = True

    def __init__(self, **kwargs):
        super(ModelStacked, self).__init__()

        parser = utils.KwargsParser(kwargs, 'ModelStacked')
        verbose = parser.get_safe('verbose', False, bool, False)
        N = parser.get_safe('n_stacked', 1, int, verbose)
        n_epochs = parser.get_safe('n_epochs', N, int, verbose)
        submodel_terrain_mask = parser.get_safe('submodel_terrain_mask', False, bool, verbose)
        use_terrain_mask = parser.get_safe('use_terrain_mask', True, bool, verbose)
        self.__pass_full_output = parser.get_safe('pass_full_output', False, bool, verbose)

        try:
            classModule = importlib.import_module('nn_wind_prediction.models.' + kwargs['submodel_type'])
        except KeyError:
            print('ModelStacked ERROR: The key "submodel_type" was not defined in the model args and is required')
            sys.exit()
        except ImportError:
           print('ModelStacked ERROR: Requested model does not exist:', kwargs['submodel_type'])
           sys.exit()

        try:
            Model = getattr(classModule, kwargs['submodel_type'])
        except AttributeError:
            print('ModelStacked ERROR: The module has no attribute:', kwargs['submodel_type'])
            sys.exit()

        # generate the stacked models based on the input params
        self.__models = nn.ModuleList()
        kwargs['use_terrain_mask'] = submodel_terrain_mask
        self.__models += [Model(**kwargs)]

        if self.__pass_full_output:
            kwargs['force_num_inputs'] = self.__models[0].get_num_outputs() + self.__models[0].get_num_inputs()
        else:
            kwargs['force_num_inputs'] = self.__models[0].get_num_outputs() + 1

        kwargs['force_num_outputs'] = self.__models[0].get_num_outputs()

        for i in range(1, N):
            if i == (N -1):
                kwargs['use_terrain_mask'] = use_terrain_mask
            self.__models += [Model(**kwargs)]

        # the prediction level is by default the full model
        self.__prediction_level = N

        # freeze all the submodels by default except the first one
        self.__train_level = 0
        self.__train_epoch_step = max(1, int(n_epochs / N))
        self.__warning_printed = False

    def set_prediction_level(self, N):
        self.__prediction_level = N

        if self.__prediction_level > len(self.__models):
            print('ModelStacke WARNING: Invalid prediction level (', N, '), setting it to the max value:', len(self.__models))
            self.__prediction_level = len(self.__models)

    def new_epoch_callback(self, epoch):
        if self.__train_level < len(self.__models):
            if (epoch >= self.__train_level * self.__train_epoch_step):
                # freeze all the model weights except for the one to train
                for model in self.__models:
                    model.freeze_model()

                self.__models[self.__train_level].unfreeze_model()
                print('ModelStacked INFO: Training submodel at idx', self.__train_level)
                self.__train_level += 1

            elif (epoch == 0):
                # freeze all the model weights except for the one to train
                for model in self.__models:
                    model.freeze_model()

                self.__models[self.__train_level].unfreeze_model()
                print('ModelStacked INFO: Training submodel at idx', self.__train_level)
                self.__train_level += 1

        else:
            if (not self.__warning_printed):
                self.__warning_printed = True
                print('ModelStacked WARNING: Maximum train level reached, continue to train last submodel')

        self.__prediction_level = self.__train_level

    def get_num_inputs(self):
        return self.__models[0].get_num_inputs()

    def get_num_outputs(self):
        return self.__models[-1].get_num_outputs()

    def forward(self, x):
        input = x.clone()
        first_iter = True
        for i in range(self.__prediction_level):
            if first_iter:
                x = self.__models[i](x)
                first_iter = False
            else:
                if self.__pass_full_output:
                    x = self.__models[i](torch.cat((input, x['pred']),1))
                else:
                    # only pass the terrain information
                    x = self.__models[i](torch.cat((input[:,0,:].unsqueeze(1), x['pred']),1))

        return x
