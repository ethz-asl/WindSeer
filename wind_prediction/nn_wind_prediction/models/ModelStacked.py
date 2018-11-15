import importlib
import sys
import torch
import torch.nn as nn

class ModelStacked(nn.Module):
    __default_pass_full_output = False
    __default_submodel_terrain_mask = False
    __default_use_terrain_mask = True

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

        try:
            self.__pass_full_output = kwargs['pass_full_output']
        except KeyError:
            self.__pass_full_output = self.__default_pass_full_output
            if verbose:
                print('ModelStacked WARNING: pass_full_output not present in kwargs, using default value:', self.__default_pass_full_output)

        try:
            self.__submodel_terrain_mask = kwargs['submodel_terrain_mask']
        except KeyError:
            self.__submodel_terrain_mask = self.__default_submodel_terrain_mask
            if verbose:
                print('ModelStacked WARNING: submodel_terrain_mask not present in kwargs, using default value:', self.__default_submodel_terrain_mask)

        try:
            self.__submodel_terrain_mask = kwargs['submodel_terrain_mask']
        except KeyError:
            self.__submodel_terrain_mask = self.__default_submodel_terrain_mask
            if verbose:
                print('ModelStacked WARNING: submodel_terrain_mask not present in kwargs, using default value:', self.__default_submodel_terrain_mask)

        try:
            self.__use_terrain_mask = kwargs['use_terrain_mask']
        except KeyError:
            self.__use_terrain_mask = self.__default_use_terrain_mask
            if verbose:
                print('ModelStacked WARNING: use_terrain_mask not present in kwargs, using default value:', self.__default_use_terrain_mask)

        # generate the stacked models based on the input params
        self.__models = nn.ModuleList()
        kwargs['use_terrain_mask'] = self.__submodel_terrain_mask
        self.__models += [Model(**kwargs)]

        if self.__pass_full_output:
            kwargs['force_num_inputs'] = self.__models[0].num_outputs() + self.__models[0].num_inputs()
        else:
            kwargs['force_num_inputs'] = self.__models[0].num_outputs() + 1

        kwargs['force_num_outputs'] = self.__models[0].num_outputs()

        for i in range(1, N):
            if i == (N -1):
                kwargs['use_terrain_mask'] = self.__use_terrain_mask
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
        input = x.clone()
        first_iter = True
        for model in self.__models:
            if first_iter:
                x = model(x)
                first_iter = False
            else:
                if self.__pass_full_output:
                    x = model(torch.cat((x, input),1))
                else:
                    # only pass the terrain information
                    x = model(torch.cat((x, input[:,0,:].unsqueeze(1)),1))

        return x
