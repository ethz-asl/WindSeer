import torch
import torch.nn as nn

'''
Base Model
'''
class ModelBase(nn.Module):
    def __init__(self, **kwargs):
        super(ModelBase, self).__init__()
        
        self.__num_inputs = None
        self.__num_outputs = None

    def new_epoch_callback(self, epoch):
        # nothing to do here
        return

    def freeze_model(self):
        def freeze_weights(m):
            for params in m.parameters():
                params.requires_grad = False

        self.apply(freeze_weights)

    def unfreeze_model(self):
        def unfreeze_weights(m):
            for params in m.parameters():
                params.requires_grad = True

        self.apply(unfreeze_weights)

    def num_inputs(self):
        return self.__num_inputs

    def num_outputs(self):
        return self.__num_outputs

    def init_params(self):
        def init_weights(m):
            if (type(m) != type(self)):
                try:
                    torch.nn.init.xavier_normal_(m.weight.data)
                except:
                    pass
                try:
                    torch.nn.init.normal_(m.bias, mean=0.0, std=0.02)
                except:
                    pass

        self.apply(init_weights)
