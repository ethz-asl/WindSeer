import importlib
import torch
import torch.nn as nn
import sys

import windseer.utils as utils

from .ModelBase import ModelBase


class ModelTwin(ModelBase):
    '''
    Experimental model consisting of two separate models predicting the wind
    and the uncertainty separately.
    '''

    def __init__(self, **kwargs):
        super(ModelTwin, self).__init__()

        parser = utils.KwargsParser(kwargs, 'ModelTwin')
        verbose = parser.get_safe('verbose', False, bool, False)
        self._uncertainty_train_mode = parser.get_safe(
            'uncertainty_train_mode', 'alternating', str, verbose
            )

        # determine the model class
        try:
            classModule = importlib.import_module(
                'windseer.nn.models.' + kwargs['submodel_type']
                )
        except KeyError:
            print(
                'ModelTwin ERROR: The key "submodel_type" was not defined in the model args and is required'
                )
            sys.exit()
        except ImportError:
            print(
                'ModelTwin ERROR: Requested model does not exist:',
                kwargs['submodel_type']
                )
            sys.exit()

        Model = getattr(classModule, kwargs['submodel_type'])

        if (
            self._uncertainty_train_mode != 'mean' and
            self._uncertainty_train_mode != 'uncertainty' and
            self._uncertainty_train_mode != 'both' and
            self._uncertainty_train_mode != 'alternating'
            ):
            raise ValueError(
                'Unknown uncertainty train mode: ', self._uncertainty_train_mode
                )

        self._model_mean = Model(**kwargs)
        self._model_uncertainty = Model(**kwargs)

    def new_epoch_callback(self, epoch):
        '''
        Callback executed before each training episode.
        Switch between training the different models.

        Parameters
        ----------
        epoch : int
            Current epoch
        '''
        if self._uncertainty_train_mode == 'mean':
            self.freeze_uncertainty()
            self.unfreeze_mean()
        elif self._uncertainty_train_mode == 'uncertainty':
            self.unfreeze_uncertainty()
            self.freeze_mean()
        elif self._uncertainty_train_mode == 'both':
            self.unfreeze_uncertainty()
            self.unfreeze_mean()
        else:
            if epoch % 2 == 0:
                self.freeze_uncertainty()
                self.unfreeze_mean()
            else:
                self.unfreeze_uncertainty()
                self.freeze_mean()

    def freeze_mean(self):
        '''
        Freeze the model predicting the wind.
        '''
        self._model_mean.freeze_model()

    def freeze_uncertainty(self):
        '''
        Freeze the model predicting the uncertainty.
        '''
        self._model_uncertainty.freeze_model()

    def unfreeze_mean(self):
        '''
        Unfreeze the model predicting the wind.
        '''
        self._model_mean.unfreeze_model()

    def unfreeze_uncertainty(self):
        '''
        Unfreeze the model predicting the uncertainty.
        '''
        self._model_uncertainty.unfreeze_model()

    def get_num_inputs(self):
        '''
        Get the number of input channels.

        Returns
        -------
        num_inputs : float
            Number of input channels
        '''
        return self._model_mean.get_num_inputs()

    def get_num_outputs(self):
        '''
        Get the number of output channels.

        Returns
        -------
        num_inputs : float
            Number of output channels
        '''
        return self._model_mean.get_num_outputs()

    def init_params(self):
        '''
        Custom parameter initialization.
        '''
        self._model_mean.init_params()
        self._model_uncertainty.init_params()

    def forward(self, x):
        '''
        Model prediction function.
        The output is a dictionary consisting of the
        prediction and uncertainty.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        output : dict
            Prediction dictionary
        '''
        mean = self._model_mean.forward(x)['pred']
        logvar = self._model_uncertainty.forward(x)['pred']
        return {'pred': mean, 'logvar': logvar}
