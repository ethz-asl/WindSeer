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

        try:
            self.auto_channel_scaling = bool(kwargs['auto_channel_scaling'])
        except KeyError:
            self.auto_channel_scaling = False
            print('CombinedLoss: auto_channel_scaling not present in kwargs, using default value:', False)

        if self.auto_channel_scaling:
            self.step_counter = 0

            try:
                self.eps_scaling = float(kwargs['eps_scaling'])
            except KeyError:
                self.eps_scaling = 1E-2
                print('CombinedLoss: eps_scaling not present in kwargs, using default value:', 1E-2)

            try:
                self.eps_scheduling_mode = kwargs['eps_scheduling_mode']
            except KeyError:
                self.eps_scheduling_mode = 'None'
                print('CombinedLoss: eps_scheduling_mode not present in kwargs, using default value:', 'None')

            try:
                self.eps_scheduling_kwargs = kwargs['eps_scheduling_kwargs']
            except KeyError:
                self.eps_scheduling_kwargs = {}
                print('CombinedLoss: eps_scheduling_mode not present in kwargs, using default value: {}')

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

    def step(self):
        if self.auto_channel_scaling:
            if self.eps_scheduling_mode == 'None':
                pass

            elif self.eps_scheduling_mode == 'step':
                if self.step_counter > 0 and self.step_counter % self.eps_scheduling_kwargs['step_size'] == 0:
                    self.eps_scaling *= self.eps_scheduling_kwargs['gamma']

            elif self.eps_scheduling_mode == 'decay':
                self.eps_scaling *= self.eps_scheduling_kwargs['gamma']

            else:
                print('CombinedLoss: Unknown eps scheduling mode: ', self.eps_scheduling_mode)

            self.step_counter += 1

    def forward(self, predicted, target, input, W = None):
        if (len(predicted['pred'].shape) != 5):
            raise ValueError('CombinedLoss: the loss is only defined for 5D data. Unsqueeze single samples!')

        # make sure weighting matrix is not empty or None
        if W is None or W.shape[-1] ==0:
            W = None
        else:
            # send weighting matrix to same device as target
            W = W.to(target.device)

        # compute the terrain factors
        terrain = input[:, 0:1]
        terrain_correction_factors = utils.compute_terrain_factor(predicted['pred'], terrain)

        # scale the predictions and labels by the average value of each channel
        if self.auto_channel_scaling:
            channel_scaling = target.abs().mean(tuple(range(2, len(target.shape))))

            channel_scaling /= terrain_correction_factors.unsqueeze(-1)

            channel_scaling = channel_scaling.clamp(min=self.eps_scaling)

            channel_scaling = channel_scaling.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            channel_scaling = channel_scaling.expand_as(target)

            target /= channel_scaling
            predicted['pred'] /= channel_scaling

        return self.compute_loss(predicted, target, input, W, terrain_correction_factors)

    def compute_loss(self, predicted, target, input, W = None, terrain_correction_factors = None):
        loss = 0

        # compute all of the loss components of the combined loss, store them in the last computed dict for retrieval
        for k in range(len(self.loss_components)):
            # move the loss factor to the same device as the prediction tensor
            factor = self.loss_factors[k].to(predicted['pred'].device)

            # compute the value individual component
            loss_component = self.loss_components[k](predicted, target, input, W, terrain_correction_factors)

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

    def forward(self, predicted, target, input, W = None, terrain_correction_factors = None):
        if (predicted['pred'].shape != target.shape):
            raise ValueError('LPLoss: predicted and target do not have the same shape, pred:{}, target:{}'
                             .format(predicted['pred'].shape, target.shape))

        if (len(predicted['pred'].shape) != 5) or (len(input.shape) != 5):
            raise ValueError('LPLoss: the loss is only defined for 5D data. Unsqueeze single samples!')

        return self.compute_loss(predicted['pred'], target, input, W, terrain_correction_factors)

    def compute_loss(self, predicted, target, input, W = None, terrain_correction_factors = None):
        # first compute the p-loss for each sample in the batch
        loss = (abs(predicted - target)) ** self.__p

        # weight the loss with the pixel-wise weighting matrix and take the mean over the volume
        if W is not None:
            loss *= W

        loss = loss.mean(-1).mean(-1).mean(-1).mean(-1)

        # compute terrain correction factor for each sample in batch
        if self.__exclude_terrain and terrain_correction_factors is not None:
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

    def forward(self, prediction, label, input, W = None, terrain_correction_factors = None):
        # compute the loss per pixel, apply weighting and take the mean over the channels
        loss = (self.__loss(prediction['pred'], label))

        if W is not None:
            loss *= W

        loss = loss.mean(dim=1)

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


    def forward(self, predicted, target, input, W = None, terrain_correction_factors = None):
        if (predicted['pred'].shape != target.shape):
            raise ValueError('DivergenceFreeLoss: predicted and target do not have the same shape, pred:{}, target:{}'
                    .format(predicted['pred'].shape, target.shape))

        if (len(predicted['pred'].shape) != 5):
            raise ValueError('DivergenceFreeLoss: the loss is only defined for 5D data. Unsqueeze single samples!')

        return self.compute_loss(predicted['pred'], target, input, W, terrain_correction_factors)

    def compute_loss(self, predicted, target, input, W = None, terrain_correction_factors = None):
        target = torch.zeros_like(target).to(predicted.device)
        terrain = input[:, 0:1]

        # compute divergence of the prediction
        predicted_divergence = utils.divergence(predicted, self.__grid_size, terrain.squeeze(1)).unsqueeze(1)

        # compute physics loss for all elements, weight it by pixel, and average losses over each sample in batch
        loss = self.__loss(predicted_divergence,target)

        if W is not None:
            loss *= W

        loss = loss.mean(tuple(range(1, len(predicted.shape))))
        # compute terrain correction factor for each sample in batch
        if self.__exclude_terrain and terrain_correction_factors is not None:
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

    def forward(self, predicted, target, input, W = None, terrain_correction_factors = None):
        if (predicted['pred'].shape != target.shape):
            raise ValueError('VelocityGradientLoss: predicted and target do not have the same shape, pred:{}, target:{}'
                             .format(predicted['pred'].shape, target.shape))

        if (len(predicted['pred'].shape) != 5):
            raise ValueError('VelocityGradientLoss: the loss is only defined for 5D data. Unsqueeze single samples!')

        return self.compute_loss(predicted['pred'], target, input, W, terrain_correction_factors)

    def compute_loss(self, predicted, target, input, W = None, terrain_correction_factors = None):
        terrain = input[:, 0:1]

        # compute gradients of prediction and target
        target_grad = utils.gradient(target,self.__grid_size,terrain.squeeze(1))
        predicted_grad = utils.gradient(predicted,self.__grid_size,terrain.squeeze(1))

        # compute physics loss for all elements, weight it by pixel, and average losses over each sample in batch
        loss = self.__loss(predicted_grad, target_grad)

        if W is not None:
            loss *= W

        loss = loss.mean(tuple(range(1, len(predicted.shape))))

        # compute terrain correction factor for each sample in batch
        if self.__exclude_terrain and terrain_correction_factors is not None:
            # apply terrain correction factor to loss of each sample in batch
            loss *= terrain_correction_factors

        # return batchwise mean of loss
        return loss.mean()

#-------------------------------------- Kullback–Leibler Divergence Loss -----------------------------------------------
class KLDivLoss(Module):
    '''
    Kullback-Leibler Divergence Loss according to https://arxiv.org/pdf/1312.6114.pdf
    '''

    def __init__(self, **kwargs):
        super(KLDivLoss, self).__init__()

    @staticmethod
    def kld_gaussian(mean1, logvar1, mean2, logvar2):
        # TODO: This could/should probably be an external function, since it is static and could be used elsewhere
        # Assuming p and q are *univariate* gaussian (no cross-correlation terms):
        # KL(p || q) = log(\sigma_q/\sigma_p) + (\sigma_p^2 + (\mu_p - \mu_q)^2)/(2\sigma_q^2) - 1/2
        # Recall: log (x^2) = 2 log (x), so we can take the 1/2 out of the first term if we are given log variances
        return -0.5 * torch.sum(1.0 + logvar1 - logvar2 - (logvar1.exp() + (mean1 - mean2).pow(2)) / logvar2.exp())

    def forward(self, predicted, target, input, W = None, terrain_correction_factors = None):
        if ('distribution_mean' not in predicted.keys()):
            raise ValueError('KLDivLoss: distribution_mean needs to be in the prediction dict')

        if ('distribution_logvar' not in predicted.keys()):
            raise ValueError('KLDivLoss: distribution_mean needs to be in the prediction dict')

        if (predicted['distribution_mean'].shape != predicted['distribution_logvar'].shape):
            raise ValueError('KLDivLoss: the mean and logvar need to have the same shape')

        return self.compute_loss(predicted['distribution_mean'], predicted['distribution_logvar'])

    def compute_loss(self, mean, logvar):
        # We assume the loss is being computed as KLD with respect to zero mean, unit variance (log(1) = 0)
        return self.kld_gaussian(mean, logvar, mean2=torch.zeros(1, device=mean.device), logvar2=torch.zeros(1, device=mean.device))


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
            self.__eps = float(kwargs['uncertainty_loss_eps'])
        except KeyError:
            self.__eps = self.__default_eps
            print('GaussianLogLikelihoodLoss: uncertainty_loss_eps not present in kwargs, using default value:', self.__eps)

        try:
           self.__exclude_terrain =  kwargs['exclude_terrain']
        except KeyError:
            self.__exclude_terrain = self.__default_exclude_terrain
            print('GaussianLogLikelihoodLoss: exclude_terrain not present in kwargs, using default value:',
                  self.__default_exclude_terrain)

    def forward(self, predicted, target, input, W = None, terrain_correction_factors = None):
        if (predicted['pred'].shape != target.shape):
            raise ValueError('GaussianLogLikelihoodLoss: predicted and target do not have the same shape, pred:{}, target:{}'
                             .format(predicted['pred'].shape, target.shape))

        if 'logvar' not in predicted.keys():
            raise ValueError('GaussianLogLikelihoodLoss: the uncertainty needs to be predicted and present in the output dict (logvar)')

        if (len(predicted['pred'].shape) != 5):
            raise ValueError('GaussianLogLikelihoodLoss: the loss is only defined for 5D data. Unsqueeze single samples!')

        if (predicted['pred'].shape != predicted['logvar'].shape):
            raise ValueError('The variance and the mean need to have the same shape')

        return self.compute_loss(predicted['pred'], predicted['logvar'], target, input, W, terrain_correction_factors)

    def compute_loss(self, mean, log_variance, target, input, W = None, terrain_correction_factors = None):
        mean_error =  mean - target

        # compute loss for all elements
        loss = 0.5*log_variance + (mean_error * mean_error) / log_variance.exp().clamp(min=self.__eps, max=1e10)

        if W is not None:
            loss *= W

        # average weighted loss over each sample in batch
        loss = loss.mean(tuple(range(1, len(mean_error.shape))))

        # compute terrain correction factor for each sample in batch
        if self.__exclude_terrain and terrain_correction_factors is not None:
            # apply terrain correction factor to loss of each sample in batch
            loss *= terrain_correction_factors

        # return batchwise mean of loss
        return loss.mean()
