import importlib
import sys
import torch
import torch.nn as nn

class ModelStacked(nn.Module):
    def __init__(self, **kwargs):
        super(ModelStacked, self).__init__()

        try:
            N = kwargs['n_stacked']
        except KeyError:
            print('ModelStacked ERROR: The key "n_stacked" was not defined in the model args and is required')
            sys.exit()

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

        self.__models = nn.ModuleList()
        self.__models += [Model(**kwargs)]
        kwargs['force_num_inputs'] = self.__models[0].num_outputs()
        kwargs['force_num_outputs'] = self.__models[0].num_outputs()

        for i in range(1, N):
            self.__models += [Model(**kwargs)]

    def init_params(self):
        for model in self.__models:
            x = model.init_params()

    def num_inputs(self):
        return self.__models[0].num_inputs()

    def num_outputs(self):
        return self.__models[-1].num_outputs()

    def freeze_model_idx(self, N):
        if (N < 0 or N >= len(self.__models)):
            print('ModelStacked WARNING: Invalid index to freeze model: ', N, '. Not doing anything')
            return

        self.__models[N].freeze_model()

    def unfreeze_model_idx(self, N):
        if (N < 0 or N >= len(self.__models)):
            print('ModelStacked WARNING: Invalid index to unfreeze model: ', N, '. Not doing anything')
            return

        self.__models[N].unfreeze_model()

    def forward(self, x):
        for model in self.__models:
            x = model(x)
        return x
