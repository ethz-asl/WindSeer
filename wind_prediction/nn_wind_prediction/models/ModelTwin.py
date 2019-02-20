import importlib
import torch
import torch.nn as nn
import sys

class ModelTwin(nn.Module):
    __default_uncertainty_train_mode = 'alternating'

    def __init__(self, **kwargs):
        super(ModelTwin, self).__init__()

        # determine the model class
        try:
            classModule = importlib.import_module('nn_wind_prediction.models.' + kwargs['submodel_type'])
        except KeyError:
            print('ModelTwin ERROR: The key "submodel_type" was not defined in the model args and is required')
            sys.exit()
        except ImportError:
           print('ModelTwin ERROR: Requested model does not exist:', kwargs['submodel_type'])
           sys.exit()

        Model = getattr(classModule, kwargs['submodel_type'])

        try:
            self.__uncertainty_train_mode = kwargs['uncertainty_train_mode']
        except KeyError:
            self.__uncertainty_train_mode = self.__default_uncertainty_train_mode
            if verbose:
                print('ModelTwin WARNING: uncertainty_train_mode not present in kwargs, using default value:', self.__default_uncertainty_train_mode)

        if (self.__uncertainty_train_mode != 'mean' or self.__uncertainty_train_mode != 'uncertainty' or
            self.__uncertainty_train_mode != 'both' or self.__uncertainty_train_mode != 'alternating'):
            print('Unknown train mode ', self.__uncertainty_train_mode, ', setting it to the default value:', self.__default_uncertainty_train_mode)
            self.__uncertainty_train_mode = self.__default_uncertainty_train_mode

        self.__model_mean = Model(**kwargs)
        self.__model_uncertainty = Model(**kwargs)

    def new_epoch_callback(self, epoch):
        if self.__uncertainty_train_mode == 'mean':
            self.freeze_uncertainty()
            self.unfreeze_mean()
        elif self.__uncertainty_train_mode == 'uncertainty':
            self.unfreeze_uncertainty()
            self.freeze_mean()
        elif self.__uncertainty_train_mode == 'both':
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
        self.__model_mean.freeze_model()

    def freeze_uncertainty(self):
        self.__model_uncertainty.freeze_model()

    def unfreeze_mean(self):
        self.__model_mean.unfreeze_model()

    def unfreeze_uncertainty(self):
        self.__model_uncertainty.unfreeze_model()

    def num_inputs(self):
        return self.__model_mean.num_inputs()

    def num_outputs(self):
        return self.__model_mean.num_outputs() + self.__model_uncertainty.num_outputs()

    def init_params(self):
        self.__model_mean.init_params()
        self.__model_uncertainty.init_params()

    def forward(self, x):
        x1 = self.__model_mean.forward(x)
        x2 = self.__model_uncertainty.forward(x)
        x = torch.cat([x1,x2],1)
        return x

    def predict_mean(self, x):
        return self.__model_mean.forward(x)

    def predict_uncertainty(self, x):
        return self.__model_uncertainty.forward(x)