import torch
import torch.nn as nn
from nn_wind_prediction.models import ModelEDNN3D

class ModelEDNN3D_Twin(nn.Module):
    def __init__(self, n_input_layers, n_output_layers, n_x, n_y, n_z, n_downsample_layers,
                 interpolation_mode, align_corners, skipping, use_terrain_mask, pooling_method,
                 use_mapping_layer, use_fc_layers, fc_scaling):
        super(ModelEDNN3D_Twin, self).__init__()
        
        # check if the number of output layers is divisible by 2
        if (n_output_layers!= 2 * n_input_layers):
            raise ValueError('The number of output channels has to be twice the input channels')
        
        self._n_mean = int(n_output_layers/2)
        
        self.__model_mean = ModelEDNN3D(n_input_layers, int(n_output_layers/2), n_x, n_y, n_z, n_downsample_layers,
                 interpolation_mode, align_corners, skipping, use_terrain_mask, pooling_method,
                 use_mapping_layer, use_fc_layers, fc_scaling)

        self.__model_uncertainty = ModelEDNN3D(n_input_layers, int(n_output_layers/2), n_x, n_y, n_z, n_downsample_layers,
                 interpolation_mode, align_corners, skipping, use_terrain_mask, pooling_method,
                 use_mapping_layer, use_fc_layers, fc_scaling)

    def freeze_mean(self):
        self.__model_mean.freeze_model()

    def freeze_uncertainty(self):
        self.__model_uncertainty.freeze_model()

    def unfreeze_mean(self):
        self.__model_mean.unfreeze_model()

    def unfreeze_uncertainty(self):
        self.__model_uncertainty.unfreeze_model()

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
