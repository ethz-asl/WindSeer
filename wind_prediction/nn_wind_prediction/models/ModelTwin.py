import importlib
import torch
import torch.nn as nn
import sys

class ModelTwin(nn.Module):
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

        self.__model_mean = Model(**kwargs)
        self.__model_uncertainty = Model(**kwargs)

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
