import importlib
import sys
import torch
import torch.nn as nn

import windseer.utils as utils

from .ModelBase import ModelBase


class ModelStacked(ModelBase):
    '''
    Experimental model consisting of multiple stacked models. 
    '''

    def __init__(self, **kwargs):
        super(ModelStacked, self).__init__()

        parser = utils.KwargsParser(kwargs, 'ModelStacked')
        verbose = parser.get_safe('verbose', False, bool, False)
        N = parser.get_safe('n_stacked', 1, int, verbose)
        n_epochs = parser.get_safe('n_epochs', N, int, verbose)
        submodel_terrain_mask = parser.get_safe(
            'submodel_terrain_mask', False, bool, verbose
            )
        use_terrain_mask = parser.get_safe('use_terrain_mask', True, bool, verbose)
        self._pass_full_output = parser.get_safe(
            'pass_full_output', False, bool, verbose
            )

        try:
            classModule = importlib.import_module(
                'windseer.nn.models.' + kwargs['submodel_type']
                )
        except KeyError:
            print(
                'ModelStacked ERROR: The key "submodel_type" was not defined in the model args and is required'
                )
            sys.exit()
        except ImportError:
            print(
                'ModelStacked ERROR: Requested model does not exist:',
                kwargs['submodel_type']
                )
            sys.exit()

        try:
            Model = getattr(classModule, kwargs['submodel_type'])
        except AttributeError:
            print(
                'ModelStacked ERROR: The module has no attribute:',
                kwargs['submodel_type']
                )
            sys.exit()

        # generate the stacked models based on the input params
        self._models = nn.ModuleList()
        kwargs['use_terrain_mask'] = submodel_terrain_mask
        self._models += [Model(**kwargs)]

        if self._pass_full_output:
            kwargs['force_num_inputs'] = self._models[0].get_num_outputs(
            ) + self._models[0].get_num_inputs()
        else:
            kwargs['force_num_inputs'] = self._models[0].get_num_outputs() + 1

        kwargs['force_num_outputs'] = self._models[0].get_num_outputs()

        for i in range(1, N):
            if i == (N - 1):
                kwargs['use_terrain_mask'] = use_terrain_mask
            self._models += [Model(**kwargs)]

        # the prediction level is by default the full model
        self._prediction_level = N

        # freeze all the submodels by default except the first one
        self._train_level = 0
        self._train_epoch_step = max(1, int(n_epochs / N))
        self._warning_printed = False

    def set_prediction_level(self, N):
        '''
        Sets the number of submodels used to get the predictions.
        Setting a level below the maximum number of submodules
        allows to observe the predictions of the intermediate
        models.

        Parameters
        ----------
        N : int
            Requested prediction level
        '''
        self._prediction_level = N

        if self._prediction_level > len(self._models):
            print(
                'ModelStacke WARNING: Invalid prediction level (', N,
                '), setting it to the max value:', len(self._models)
                )
            self._prediction_level = len(self._models)

    def new_epoch_callback(self, epoch):
        '''
        Callback executed before each training episode.
        Adjust the prediction level according to train_epoch_step
        and the current epoch.

        Parameters
        ----------
        epoch : int
            Current epoch
        '''
        if self._train_level < len(self._models):
            if (epoch >= self._train_level * self._train_epoch_step):
                # freeze all the model weights except for the one to train
                for model in self._models:
                    model.freeze_model()

                self._models[self._train_level].unfreeze_model()
                print('ModelStacked INFO: Training submodel at idx', self._train_level)
                self._train_level += 1

            elif (epoch == 0):
                # freeze all the model weights except for the one to train
                for model in self._models:
                    model.freeze_model()

                self._models[self._train_level].unfreeze_model()
                print('ModelStacked INFO: Training submodel at idx', self._train_level)
                self._train_level += 1

        else:
            if (not self._warning_printed):
                self._warning_printed = True
                print(
                    'ModelStacked WARNING: Maximum train level reached, continue to train last submodel'
                    )

        self._prediction_level = self._train_level

    def get_num_inputs(self):
        '''
        Get the number of input channels.

        Returns
        -------
        num_inputs : float
            Number of input channels
        '''
        return self._models[0].get_num_inputs()

    def get_num_outputs(self):
        '''
        Get the number of output channels.

        Returns
        -------
        num_inputs : float
            Number of output channels
        '''
        return self._models[-1].get_num_outputs()

    def forward(self, x):
        '''
        Model prediction function.
        The output is a dictionary consisting of the
        prediction and based on the configuration the
        uncertainty as well.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        output : dict
            Prediction dictionary
        '''
        input = x.clone()
        first_iter = True
        for i in range(self._prediction_level):
            if first_iter:
                x = self._models[i](x)
                first_iter = False
            else:
                if self._pass_full_output:
                    x = self._models[i](torch.cat((input, x['pred']), 1))
                else:
                    # only pass the terrain information
                    x = self._models[i](
                        torch.cat((input[:, 0, :].unsqueeze(1), x['pred']), 1)
                        )

        return x
