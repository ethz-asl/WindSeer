import torch
from torch.nn import Module
import nn_wind_prediction.utils as utils
import sys
import warnings
import re

class CombinedLoss(Module):
    '''
    Loss which can combine all of the different loss functions into one using learnable homoscedastic uncertainty factors.
    Based on Geometric Loss Functions for Camera Pose Regression with Deep Learning by Alex Kendall, Roberto Cipolla.
    Link: https://arxiv.org/abs/1704.00390
    '''

    def __init__(self, loss_components, learn_scaling, **kwargs):
        super(CombinedLoss, self).__init__()

        self.learn_scaling = learn_scaling
        self.loss_component_names = loss_components
        self.loss_components = []
        self.loss_factors = []
        self.last_computed_loss_components = dict()

        if 'GaussianLogLikelihoodLoss' in self.loss_component_names and len(self.loss_component_names)>1:
            raise ValueError('Sorry, for now GaussianLogLikelihoodLoss can only be on its own!')

        # if the scaling must be learnt, use a ParameterList for the scaling factors
        if self.learn_scaling and len(self.loss_component_names)>1:
            self.loss_factors = torch.nn.ParameterList(self.loss_factors)

        for i, loss_component in enumerate(self.loss_component_names):
            # get kwargs of the loss_component
            loss_component_kwargs = kwargs[loss_component + '_kwargs']

            # initialize buffer for this component
            self.last_computed_loss_components[loss_component] = 0.0

            # handling for LPLoss (L1Loss, L2Loss, L3Loss etc...)
            if re.match(r"L+\d+Loss", loss_component):
                # get order p from name
                p = re.search(r'\d+', loss_component).group()
                # add p to kwargs
                loss_component_kwargs['p'] = int(p)
                # replace name with LPLoss for initialization, replace allows for multiple instances of same loss
                loss_component = loss_component.replace(loss_component[0:len(p)+5], 'LPLoss')

            # account for multiple losses of the same class when initializing the loss component
            loss_multiple = re.search(r'\d+$', loss_component)
            if loss_multiple is not None:
                loss_component_fn = getattr(sys.modules[__name__], loss_component[0:-len(loss_multiple.group())])
            else:
                loss_component_fn = getattr(sys.modules[__name__], loss_component)
            self.loss_components += [loss_component_fn(**loss_component_kwargs)]

            # if there are more than one components to the overall loss, apply scaling factor that can be learnt
            if len(loss_components) > 1:
                loss_factor_init = torch.Tensor([loss_component_kwargs['loss_factor_init']])
                if self.learn_scaling:
                    self.loss_factors += [torch.nn.Parameter(loss_factor_init, requires_grad=True)]
                else:
                    self.loss_factors += [loss_factor_init]

            else:
                # using a factor of 0.0 == no loss scaling, due to the homoscedastic uncertainty expression. see below.
                self.loss_factors += [torch.Tensor([0.0])]
                self.learn_scaling = False

    def forward(self, predicted, target, input, W=1.0):
        if (len(predicted.shape) != 5):
            raise ValueError('CombinedLoss: the loss is only defined for 5D data. Unsqueeze single samples!')

        return self.compute_loss(predicted, target, input, W)

    def compute_loss(self, predicted, target, input, W):
        loss = 0

        # compute all of the loss components of the combined loss, store them in the last computed dict for retrieval
        for k in range(len(self.loss_components)):
            # move the loss factor to the same device as the prediction tensor
            factor = self.loss_factors[k].to(predicted.device)

            # compute the value individual component
            loss_component = self.loss_components[k](predicted, target, input, W)

            # store this value in the buffer dict
            self.last_computed_loss_components[self.loss_component_names[k]] = loss_component.item()

            # apply the homoscedastic uncertainty factor: L += L_q*exp(-q)+q
            loss += loss_component *torch.exp(-factor) + factor

        return loss

#------------------------------------------- Classic Losses ------------------------------------------------------------
class LPLoss(Module):
    '''
    Loss based on the on the p-distance where p is the order of the norm used to compute the distance. If p=1, it is
    equivalent to the L1 loss. If p=2, it is equivalent to the L2/MSE loss. A correction factor can be applied to
    account for the amount of terrain data in the samples.

    Init params:
        p: order of the loss. Use 1 for L1, 2 for MSE.
        kwargs:
            exclude_terrain: bool indicating if a correction factor should be applied to make loss independent of the
                                amount of terrain.
    '''
    __default_exclude_terrain = True

    def __init__(self, p, **kwargs):
        super(LPLoss, self).__init__()

        self.__p = p

        # validity check and warning for p
        if self.__p <= 0:
            raise ValueError('LPLoss: loss order p should be greater than 0, p = {} was used!'.format(self.__p))
        if self.__p < 1:
            warnings.warn('LPLoss: loss order p is fractional, p = {}'.format(self.__p))

        try:
            self.__exclude_terrain = kwargs['exclude_terrain']
        except KeyError:
            self.__exclude_terrain = self.__default_exclude_terrain
            print('LPLoss: exclude_terrain not present in kwargs, using default value:',
                  self.__default_exclude_terrain)

    def forward(self, predicted, target, input, W=1.0):
        if (predicted.shape != target.shape):
            raise ValueError('LPLoss: predicted and target do not have the same shape, pred:{}, target:{}'
                             .format(predicted.shape, target.shape))

        if (len(predicted.shape) != 5) or (len(input.shape) != 5):
            raise ValueError('LPLoss: the loss is only defined for 5D data. Unsqueeze single samples!')

        return self.compute_loss(predicted, target, input, W)

    def compute_loss(self, predicted, target, input, W):
        # first compute the p-loss for each sample in the batch
        loss = (abs(predicted - target)) ** self.__p

        # weight the loss with the pixel-wise weighting matrix
        loss = (loss*W).mean(-1).mean(-1).mean(-1).mean(-1)

        # compute terrain correction factor for each sample in batch
        if self.__exclude_terrain:
            terrain = input[:, 0:1]
            terrain_correction_factors = utils.compute_terrain_factor(predicted, terrain)

            # apply terrain correction factor to loss of each sample in batch
            loss *= terrain_correction_factors

        # return batchwise mean of loss
        return loss.mean()

#--------------------------------------------- Scaled Loss  ------------------------------------------------------------
class ScaledLoss(Module):
    __default_loss_type = 'MSE'
    __default_max_scale = 4.0
    __default_norm_threshold = 0.5
    __default_exclude_terrain = True
    __default_no_scaling = False

    def __init__(self, **kwargs):
        super(ScaledLoss, self).__init__()

        try:
            self.__loss_type = kwargs['loss_type']
        except KeyError:
            self.__loss_type = self.__default_loss_type
            print('ScaledLoss: loss_type not present in kwargs, using default value:', self.__default_loss_type)

        try:
            self.__max_scale = kwargs['max_scale']
        except KeyError:
            self.__max_scale = self.__default_max_scale
            print('ScaledLoss: max_scale not present in kwargs, using default value:', self.__default_max_scale)

        try:
            self.__norm_threshold = kwargs['norm_threshold']
        except KeyError:
            self.__norm_threshold = self.__default_norm_threshold
            print('ScaledLoss: norm_threshold not present in kwargs, using default value:', self.__default_norm_threshold)

        try:
            self.__exclude_terrain = kwargs['exclude_terrain']
        except KeyError:
            self.__exclude_terrain = self.__default_exclude_terrain
            print('ScaledLoss: exclude_terrain not present in kwargs, using default value:', self.__default_exclude_terrain)

        try:
            self.__no_scaling = kwargs['no_scaling']
        except KeyError:
            self.__no_scaling = self.__default_no_scaling
            print('ScaledLoss: no_scaling not present in kwargs, using default value:', self.__default_no_scaling)


        if (self.__loss_type == 'MSE'):
            self.__loss = torch.nn.MSELoss(reduction='none')
        elif (self.__loss_type == 'L1'):
            self.__loss = torch.nn.L1Loss(reduction='none')
        else:
            raise ValueError('Unknown loss type: ', self.__loss_type)

        # input sanity checks
        if (self.__max_scale < 1.0):
            raise ValueError('max_scale must be larger than 1.0')

        if (self.__norm_threshold <= 0.0):
            raise ValueError('max_scale must be larger than 0.0')

    def forward(self, prediction, label, input, W=1.0):
        # compute the loss per pixel
        loss = (self.__loss(prediction, label)*W).mean(dim=1)

        if self.__no_scaling:
            scale = torch.ones_like(loss)

        else:
            # compute the scale
            scale = label[:,0:3]-input[:,1:4]
            scale = (scale.norm(dim=1) / label[:,0:3].norm(dim=1))

            # set nans to 0.0
            scale = torch.where(torch.isnan(scale), torch.zeros_like(scale), scale)

            # map to a range of 1.0 to max_scale
            scale = scale.clamp(min=0.0, max=self.__norm_threshold) * (1.0 / self.__norm_threshold) * (self.__max_scale - 1.0) + 1.0

        if self.__exclude_terrain:
            # batchwise scaled loss
            loss = (loss*scale).sum(-1).sum(-1).sum(-1)

            # normalization factor per batch
            factor = torch.where(input[:,0,:,:,:]==0, torch.zeros_like(scale), scale).sum(-1).sum(-1).sum(-1)

            # normalize and compute the mean over the batches
            return (loss / factor.clamp(min=1.0)).mean()

        else:
            return (loss*scale).sum() / scale.sum()

#------------------------------------------- Divergence Free Loss  -----------------------------------------------------
class DivergenceFreeLoss(Module):
    '''
    Loss function predicted divergence field is compared to 0 divergence field.

    Init params:
        loss_type: whether to use a L1 or a MSE based method.
        grid_size: list containing the grid spacing in directions X, Y and Z of the dataset. [m]
    '''
    __default_loss_type = 'MSE'
    __default_grid_size = [1, 1, 1]
    __default_exclude_terrain = True

    def __init__(self, **kwargs):
        super(DivergenceFreeLoss, self).__init__()
        try:
            self.__loss_type = kwargs['loss_type']
        except KeyError:
            self.__loss_type = self.__default_loss_type
            print('DivergenceFreeLoss: loss_type not present in kwargs, using default value:', self.__default_loss_type)

        try:
            self.__grid_size = kwargs['grid_size']
        except KeyError:
            self.__grid_size = self.__default_grid_size
            print('DivergenceFreeLoss: grid_size not present in kwargs, using default value:', self.__default_grid_size)

        try:
           self.__exclude_terrain =  kwargs['exclude_terrain']
        except KeyError:
            self.__exclude_terrain = self.__default_exclude_terrain
            print('DivergenceFreeLoss: exclude_terrain not present in kwargs, using default value:',
                  self.__default_exclude_terrain)

        if (self.__loss_type == 'MSE'):
            self.__loss = torch.nn.MSELoss(reduction='none')
        elif (self.__loss_type == 'L1'):
            self.__loss = torch.nn.L1Loss(reduction='none')
        else:
            raise ValueError('DivergenceFreeLoss: unknown loss type ', self.__loss_type)


    def forward(self, predicted, target, input, W=1):
        if (predicted.shape != target.shape):
            raise ValueError('DivergenceFreeLoss: predicted and target do not have the same shape, pred:{}, target:{}'
                    .format(predicted.shape, target.shape))

        if (len(predicted.shape) != 5):
            raise ValueError('DivergenceFreeLoss: the loss is only defined for 5D data. Unsqueeze single samples!')

        return self.compute_loss(predicted, target, input, W)

    def compute_loss(self, predicted, target, input, W):
        target = torch.zeros_like(target).to(predicted.device)
        terrain = input[:, 0:1]

        # compute divergence of the prediction
        predicted_divergence = utils.divergence(predicted, self.__grid_size, terrain.squeeze(1)).unsqueeze(1)

        # compute physics loss for all elements and average losses over each sample in batch
        loss =(self.__loss(predicted_divergence,target)*W).mean(tuple(range(1, len(predicted.shape))))

        # compute terrain correction factor for each sample in batch
        if self.__exclude_terrain:
            terrain_correction_factors = utils.compute_terrain_factor(predicted, terrain)

            # apply terrain correction factor to loss of each sample in batch
            loss *= terrain_correction_factors

        # return batchwise mean of loss
        return loss.mean()

#------------------------------------------- Velocity Gradient Loss  ---------------------------------------------------
class VelocityGradientLoss(Module):
    '''
    Upgraded version of Stream Function Loss according to https://arxiv.org/abs/1806.02071.
    Loss function in which predicted flow field spatial gradient is compared to target flow spatial gradient.

    Init params:
        loss_type: whether to use a L1 or a MSE based method.
        grid_size: list containing the grid spacing in directions X, Y and Z of the dataset. [m]
    '''
    __default_loss_type = 'MSE'
    __default_grid_size = [1, 1, 1]
    __default_exclude_terrain = True

    def __init__(self, **kwargs):
        super(VelocityGradientLoss, self).__init__()
        try:
            self.__loss_type = kwargs['loss_type']
        except KeyError:
            self.__loss_type = self.__default_loss_type
            print('VelocityGradientLoss: loss_type not present in kwargs, using default value:', self.__default_loss_type)

        try:
            self.__grid_size = kwargs['grid_size']
        except KeyError:
            self.__grid_size = self.__default_grid_size
            print('VelocityGradientLoss: grid_size not present in kwargs, using default value:', self.__default_grid_size)

        try:
           self.__exclude_terrain =  kwargs['exclude_terrain']
        except KeyError:
            self.__exclude_terrain = self.__default_exclude_terrain
            print('VelocityGradientLoss: exclude_terrain not present in kwargs, using default value:',
                  self.__default_exclude_terrain)

        if (self.__loss_type == 'MSE'):
            self.__loss = torch.nn.MSELoss(reduction='none')
        elif (self.__loss_type == 'L1'):
            self.__loss = torch.nn.L1Loss(reduction='none')
        else:
            raise ValueError('VelocityGradientLoss: unknown loss type ', self.__loss_type)

    def forward(self, predicted, target, input, W=1.0):
        if (predicted.shape != target.shape):
            raise ValueError('VelocityGradientLoss: predicted and target do not have the same shape, pred:{}, target:{}'
                             .format(predicted.shape, target.shape))

        if (len(predicted.shape) != 5):
            raise ValueError('VelocityGradientLoss: the loss is only defined for 5D data. Unsqueeze single samples!')

        return self.compute_loss(predicted, target, input, W)

    def compute_loss(self, predicted, target, input, W):
        terrain = input[:, 0:1]

        # compute gradients of prediction and target
        target_grad = utils.gradient(target,self.__grid_size,terrain.squeeze(1))
        predicted_grad = utils.gradient(predicted,self.__grid_size,terrain.squeeze(1))

        # compute physics loss for all elements and average losses over each sample in batch
        loss = (self.__loss(predicted_grad, target_grad)*W).mean(tuple(range(1, len(predicted.shape))))

        # compute terrain correction factor for each sample in batch
        if self.__exclude_terrain:
            terrain_correction_factors = utils.compute_terrain_factor(predicted, terrain)

            # apply terrain correction factor to loss of each sample in batch
            loss *= terrain_correction_factors

        # return batchwise mean of loss
        return loss.mean()

#------------------------------------------ Gaussian Log Likelihood Loss  ----------------------------------------------
class GaussianLogLikelihoodLoss(Module):
    '''
    Gaussian Log Likelihood Loss according to https://arxiv.org/pdf/1705.07115.pdf
    '''

    __default_eps = 1e-8
    __default_exclude_terrain = True

    def __init__(self, **kwargs):
        super(GaussianLogLikelihoodLoss, self).__init__()

        try:
            self.__eps = kwargs['uncertainty_loss_eps']
        except KeyError:
            self.__eps = self.__default_eps
            print('GaussianLogLikelihoodLoss: uncertainty_loss_eps not present in kwargs, using default value:', self.__eps)

        try:
           self.__exclude_terrain =  kwargs['exclude_terrain']
        except KeyError:
            self.__exclude_terrain = self.__default_exclude_terrain
            print('DivergenceFreeLoss: exclude_terrain not present in kwargs, using default value:',
                  self.__default_exclude_terrain)

    def forward(self, output, label, input, W=1.0):
        num_channels = output.shape[1]

        if (output.shape[1] != 2 * label.shape[1]) and (num_channels % 2 != 0):
            raise ValueError('The output has to have twice the number of channels of the labels')

        return self.compute_loss(output[:,:int(num_channels/2),:], output[:,int(num_channels/2):,:], label, input, W)

    def compute_loss(self, mean, log_variance, label, input, W):
        if (mean.shape[1] != log_variance.shape[1]):
            raise ValueError('The variance and the mean need to have the same number of channels')

        mean_error =  mean - label

        # compute loss for all elements
        loss = log_variance + (mean_error * mean_error) / log_variance.exp().clamp(self.__eps)

        # average loss over each sample in batch
        loss = (loss*W).mean(tuple(range(1, len(mean_error.shape))))

        # compute terrain correction factor for each sample in batch
        if self.__exclude_terrain:
            terrain = input[:, 0:1]
            terrain_correction_factors = utils.compute_terrain_factor(mean_error, terrain)

            # apply terrain correction factor to loss of each sample in batch
            loss *= terrain_correction_factors

        # return batchwise mean of loss
        return loss.mean()