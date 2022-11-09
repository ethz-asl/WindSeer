import torch
from torch.nn import Module
import windseer.utils as utils
import sys
import warnings
import re
import copy


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

        parser = utils.KwargsParser(kwargs, 'CombinedLoss')
        self.auto_channel_scaling = parser.get_safe(
            'auto_channel_scaling', False, bool, True
            )

        if self.auto_channel_scaling:
            self.step_counter = 0

            self.eps_scaling = parser.get_safe(
                'auto_channel_scaling', 1E-2, float, True
                )
            self.eps_scheduling_mode = parser.get_safe(
                'auto_channel_scaling', 'None', str, True
                )
            self.eps_scheduling_kwargs = parser.get_safe(
                'eps_scheduling_kwargs', {}, dict, True
                )

        # if the scaling must be learnt, use a ParameterList for the scaling factors
        if self.learn_scaling and len(self.loss_component_names) > 1:
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
                loss_component = loss_component.replace(
                    loss_component[0:len(p) + 5], 'LPLoss'
                    )

            # account for multiple losses of the same class when initializing the loss component
            loss_multiple = re.search(r'\d+$', loss_component)
            if loss_multiple is not None:
                loss_component_fn = getattr(
                    sys.modules[__name__], loss_component[0:-len(loss_multiple.group())]
                    )
            else:
                loss_component_fn = getattr(sys.modules[__name__], loss_component)
            self.loss_components += [loss_component_fn(**loss_component_kwargs)]

            # if there are more than one components to the overall loss, apply scaling factor that can be learnt
            if len(loss_components) > 1:
                loss_factor_init = torch.Tensor([
                    loss_component_kwargs['loss_factor_init']
                    ])
                if self.learn_scaling:
                    self.loss_factors += [
                        torch.nn.Parameter(loss_factor_init, requires_grad=True)
                        ]
                else:
                    self.loss_factors += [loss_factor_init]

            else:
                # using a factor of 0.0 == no loss scaling, due to the homoscedastic uncertainty expression. see below.
                self.loss_factors += [torch.Tensor([0.0])]
                self.learn_scaling = False

    def step(self):
        '''
        Decreases the eps value used in the auto channel scaling depending on the settings
        either stepwise or continuously in a exponential decay.
        '''
        if self.auto_channel_scaling:
            if self.eps_scheduling_mode == 'None':
                pass

            elif self.eps_scheduling_mode == 'step':
                if self.step_counter > 0 and self.step_counter % self.eps_scheduling_kwargs[
                    'step_size'] == 0:
                    self.eps_scaling *= self.eps_scheduling_kwargs['gamma']

            elif self.eps_scheduling_mode == 'decay':
                self.eps_scaling *= self.eps_scheduling_kwargs['gamma']

            else:
                print(
                    'CombinedLoss: Unknown eps scheduling mode: ',
                    self.eps_scheduling_mode
                    )

            self.step_counter += 1

    def forward(self, predicted, target, input, W=None):
        '''
        Loss forward function.
        Computes the combined loss according to the class settings.
        
        Each individual cell can be scaled according to the weighting tensor if set.
        Each sample in the batch can be scaled according to the number of flow cells
        in that sample to balance samples with different terrain sizes.

        Parameters
        ----------
        predicted : dict
            Prediction dictionary
        target : torch.Tensor
            Target (label) tensor
        input : torch.Tensor
            Input tensor
        W : torch.Tensor or None, default: None
            Weighting tensor to give different weights for each cell individually.

        Returns
        -------
        loss : torch.Tensor
            Computed loss value
        '''
        if (len(predicted['pred'].shape) != 5):
            raise ValueError(
                'CombinedLoss: the loss is only defined for 5D data. Unsqueeze single samples!'
                )

        # make sure weighting matrix is not empty or None
        if W is None or W.shape[-1] == 0:
            W = None
        else:
            # send weighting matrix to same device as target
            W = W.to(target.device)

        # compute the terrain factors
        terrain = input[:, 0:1]
        terrain_correction_factors = utils.compute_terrain_factor(
            predicted['pred'], terrain
            )

        # scale the predictions and labels by the average value of each channel
        if self.auto_channel_scaling:
            channel_scaling = target.abs().mean(tuple(range(2, len(target.shape))))

            channel_scaling /= terrain_correction_factors.unsqueeze(-1)

            channel_scaling = channel_scaling.clamp(min=self.eps_scaling)

            channel_scaling = channel_scaling.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            channel_scaling = channel_scaling.expand_as(target)

            predicted_scaled = copy.deepcopy(predicted)
            predicted_scaled['pred'] /= channel_scaling

            return self.compute_loss(
                predicted_scaled, target / channel_scaling, input, W,
                terrain_correction_factors
                )

        else:
            return self.compute_loss(
                predicted, target, input, W, terrain_correction_factors
                )

    def compute_loss(
            self, predicted, target, input, W=None, terrain_correction_factors=None
        ):
        '''
        Compute the loss according to the class settings.

        Parameters
        ----------
        predicted : dict
            Prediction dictionary
        target : torch.Tensor
            Target (label) tensor
        input : torch.Tensor
            Input tensor
        W : torch.Tensor or None, default: None
            Weighting tensor to give different weights for each cell individually.
        terrain_correction_factors : torch.Tensor or None, default: None
            Tensor with a correction factor for each sample in the batch.

        Returns
        -------
        loss : torch.Tensor
            Computed loss value
        '''
        loss = 0

        # compute all of the loss components of the combined loss, store them in the last computed dict for retrieval
        for k in range(len(self.loss_components)):
            # move the loss factor to the same device as the Prediction dictionary
            factor = self.loss_factors[k].to(predicted['pred'].device)

            # compute the value individual component
            loss_component = self.loss_components[k](
                predicted, target, input, W, terrain_correction_factors
                )

            # store this value in the buffer dict
            self.last_computed_loss_components[self.loss_component_names[k]
                                               ] = loss_component.item()

            # apply the homoscedastic uncertainty factor: L += L_q*exp(-q)+q
            loss += loss_component * torch.exp(-factor) + factor

        return loss


#------------------------------------------- Classic Losses ------------------------------------------------------------
class LPLoss(Module):
    '''
    Loss based on the on the p-distance where p is the order of the norm used to compute the distance. If p=1, it is
    equivalent to the L1 loss. If p=2, it is equivalent to the L2/MSE loss. A correction factor can be applied to
    account for the amount of terrain data in the samples.
    '''

    def __init__(self, p, **kwargs):
        '''
        Parameters
        ----------
        p : int
            Order of the loss. Use 1 for L1, 2 for MSE.
        exclude_terrain : bool, default: True
            If True the loss is only averaged over flow cells.
        '''
        super(LPLoss, self).__init__()

        self._p = p

        # validity check and warning for p
        if self._p <= 0:
            raise ValueError(
                'LPLoss: loss order p should be greater than 0, p = {} was used!'
                .format(self._p)
                )
        if self._p < 1:
            warnings.warn('LPLoss: loss order p is fractional, p = {}'.format(self._p))

        parser = utils.KwargsParser(kwargs, 'LPLoss')
        self._exclude_terrain = parser.get_safe('exclude_terrain', True, bool, True)

    def forward(
            self, predicted, target, input, W=None, terrain_correction_factors=None
        ):
        '''
        Loss forward function.

        Parameters
        ----------
        predicted : dict
            Prediction dictionary
        target : torch.Tensor
            Target (label) tensor
        input : torch.Tensor
            Input tensor
        W : torch.Tensor or None, default: None
            Weighting tensor to give different weights for each cell individually.
        terrain_correction_factors : torch.Tensor or None, default: None
            Tensor with a correction factor for each sample in the batch.

        Returns
        -------
        loss : torch.Tensor
            Computed loss value
        '''
        if (predicted['pred'].shape != target.shape):
            raise ValueError(
                'LPLoss: predicted and target do not have the same shape, pred:{}, target:{}'
                .format(predicted['pred'].shape, target.shape)
                )

        if (len(predicted['pred'].shape) != 5) or (len(input.shape) != 5):
            raise ValueError(
                'LPLoss: the loss is only defined for 5D data. Unsqueeze single samples!'
                )

        return self.compute_loss(
            predicted['pred'], target, input, W, terrain_correction_factors
            )

    def compute_loss(
            self, predicted, target, input, W=None, terrain_correction_factors=None
        ):
        '''
        Compute the loss according to the class settings.

        Parameters
        ----------
        predicted : dict
            Prediction dictionary
        target : torch.Tensor
            Target (label) tensor
        input : torch.Tensor
            Input tensor
        W : torch.Tensor or None, default: None
            Weighting tensor to give different weights for each cell individually.
        terrain_correction_factors : torch.Tensor or None, default: None
            Tensor with a correction factor for each sample in the batch.

        Returns
        -------
        loss : torch.Tensor
            Computed loss value
        '''
        # first compute the p-loss for each sample in the batch
        loss = (abs(predicted - target))**self._p

        # weight the loss with the pixel-wise weighting matrix and take the mean over the volume
        if W is not None:
            loss *= W

        loss = loss.mean(-1).mean(-1).mean(-1).mean(-1)

        # compute terrain correction factor for each sample in batch
        if self._exclude_terrain and terrain_correction_factors is not None:
            # apply terrain correction factor to loss of each sample in batch
            loss *= terrain_correction_factors

        # return batchwise mean of loss
        return loss.mean()


#--------------------------------------------- Scaled Loss  ------------------------------------------------------------
class ScaledLoss(Module):
    '''
    Scaling the loss for each cell depending on the difference of the input and label values of that cell.
    '''

    def __init__(self, **kwargs):
        '''
        Parameters
        ----------
        exclude_terrain : bool, default: True
            If True the loss is only averaged over flow cells.
        loss_type : str, default: MSE
            Whether to use a L1 or a MSE based method.
        norm_threshold : float, default: 0.5
            Minimum velocity norm per cell to avoid numerical issues [m/s]
        max_scale : float, default: 4.0
            Maximum scale value
        no_scaling : bool, default: False
            If True no individual scale per cell is used
        '''
        super(ScaledLoss, self).__init__()

        parser = utils.KwargsParser(kwargs, 'ScaledLoss')
        self._exclude_terrain = parser.get_safe('exclude_terrain', True, bool, True)
        self._loss_type = parser.get_safe('loss_type', 'MSE', str, True)
        self._norm_threshold = parser.get_safe('norm_threshold', 0.5, float, True)
        self._max_scale = parser.get_safe('max_scale', 4.0, float, True)
        self._no_scaling = parser.get_safe('no_scaling', False, bool, True)

        if (self._loss_type == 'MSE'):
            self._loss = torch.nn.MSELoss(reduction='none')
        elif (self._loss_type == 'L1'):
            self._loss = torch.nn.L1Loss(reduction='none')
        else:
            raise ValueError('Unknown loss type: ', self._loss_type)

        # input sanity checks
        if (self._max_scale < 1.0):
            raise ValueError('max_scale must be larger than 1.0')

        if (self._norm_threshold <= 0.0):
            raise ValueError('max_scale must be larger than 0.0')

    def forward(
            self, prediction, label, input, W=None, terrain_correction_factors=None
        ):
        '''
        Loss forward function.

        Parameters
        ----------
        predicted : dict
            Prediction dictionary
        target : torch.Tensor
            Target (label) tensor
        input : torch.Tensor
            Input tensor
        W : torch.Tensor or None, default: None
            Weighting tensor to give different weights for each cell individually.
        terrain_correction_factors : torch.Tensor or None, default: None
            Tensor with a correction factor for each sample in the batch.

        Returns
        -------
        loss : torch.Tensor
            Computed loss value
        '''
        # compute the loss per pixel, apply weighting and take the mean over the channels
        loss = (self._loss(prediction['pred'], label))

        if W is not None:
            loss *= W

        loss = loss.mean(dim=1)

        if self._no_scaling:
            scale = torch.ones_like(loss)

        else:
            # compute the scale
            scale = label[:, 0:3] - input[:, 1:4]
            scale = (scale.norm(dim=1) / label[:, 0:3].norm(dim=1))

            # set nans to 0.0
            scale = torch.where(torch.isnan(scale), torch.zeros_like(scale), scale)

            # map to a range of 1.0 to max_scale
            scale = scale.clamp(
                min=0.0, max=self._norm_threshold
                ) * (1.0 / self._norm_threshold) * (self._max_scale - 1.0) + 1.0

        if self._exclude_terrain:
            # batchwise scaled loss
            loss = (loss * scale).sum(-1).sum(-1).sum(-1)

            # normalization factor per batch
            factor = torch.where(
                input[:, 0, :, :, :] == 0, torch.zeros_like(scale), scale
                ).sum(-1).sum(-1).sum(-1)

            # normalize and compute the mean over the batches
            return (loss / factor.clamp(min=1.0)).mean()

        else:
            return (loss * scale).sum() / scale.sum()


#------------------------------------------- Divergence Free Loss  -----------------------------------------------------
class DivergenceFreeLoss(Module):
    '''
    Loss function predicted divergence field is compared to 0 divergence field.
    '''

    def __init__(self, **kwargs):
        '''
        Parameters
        ----------
        exclude_terrain : bool, default: True
            If True the loss is only averaged over flow cells.
        loss_type : str, default: MSE
            Whether to use a L1 or a MSE based method.
        grid_size : list, default: [1,1,1]
            List containing the grid spacing in directions X, Y and Z of the dataset. [m]
        '''
        super(DivergenceFreeLoss, self).__init__()

        parser = utils.KwargsParser(kwargs, 'DivergenceFreeLoss')
        self._loss_type = parser.get_safe('loss_type', 'MSE', str, True)
        self._grid_size = parser.get_safe('grid_size', [1, 1, 1], list, True)
        self._exclude_terrain = parser.get_safe('exclude_terrain', True, bool, True)

        if (self._loss_type == 'MSE'):
            self._loss = torch.nn.MSELoss(reduction='none')
        elif (self._loss_type == 'L1'):
            self._loss = torch.nn.L1Loss(reduction='none')
        else:
            raise ValueError('DivergenceFreeLoss: unknown loss type ', self._loss_type)

    def forward(
            self, predicted, target, input, W=None, terrain_correction_factors=None
        ):
        '''
        Loss forward function.

        Parameters
        ----------
        predicted : dict
            Prediction dictionary
        target : torch.Tensor
            Target (label) tensor
        input : torch.Tensor
            Input tensor
        W : torch.Tensor or None, default: None
            Weighting tensor to give different weights for each cell individually.
        terrain_correction_factors : torch.Tensor or None, default: None
            Tensor with a correction factor for each sample in the batch.

        Returns
        -------
        loss : torch.Tensor
            Computed loss value
        '''
        if (predicted['pred'].shape != target.shape):
            raise ValueError(
                'DivergenceFreeLoss: predicted and target do not have the same shape, pred:{}, target:{}'
                .format(predicted['pred'].shape, target.shape)
                )

        if (len(predicted['pred'].shape) != 5):
            raise ValueError(
                'DivergenceFreeLoss: the loss is only defined for 5D data. Unsqueeze single samples!'
                )

        return self.compute_loss(
            predicted['pred'], target, input, W, terrain_correction_factors
            )

    def compute_loss(
            self, predicted, target, input, W=None, terrain_correction_factors=None
        ):
        '''
        Compute the loss according to the class settings.

        Parameters
        ----------
        predicted : dict
            Prediction dictionary
        target : torch.Tensor
            Target (label) tensor
        input : torch.Tensor
            Input tensor
        W : torch.Tensor or None, default: None
            Weighting tensor to give different weights for each cell individually.
        terrain_correction_factors : torch.Tensor or None, default: None
            Tensor with a correction factor for each sample in the batch.

        Returns
        -------
        loss : torch.Tensor
            Computed loss value
        '''
        terrain = input[:, 0:1]

        # compute divergence of the prediction
        predicted_divergence = utils.divergence(
            predicted, self._grid_size, terrain.squeeze(1)
            ).unsqueeze(1)

        target_divergence = torch.zeros_like(predicted_divergence).to(predicted.device)

        # compute physics loss for all elements, weight it by pixel, and average losses over each sample in batch
        loss = self._loss(predicted_divergence, target_divergence)

        if W is not None:
            loss *= W

        loss = loss.mean(tuple(range(1, len(predicted.shape))))
        # compute terrain correction factor for each sample in batch

        if self._exclude_terrain and terrain_correction_factors is not None:
            # apply terrain correction factor to loss of each sample in batch
            loss *= terrain_correction_factors

        # return batchwise mean of loss
        return loss.mean()


#------------------------------------------- Velocity Gradient Loss  ---------------------------------------------------
class VelocityGradientLoss(Module):
    '''
    Upgraded version of Stream Function Loss according to https://arxiv.org/abs/1806.02071.
    Loss function in which predicted flow field spatial gradient is compared to target flow spatial gradient.
    '''

    def __init__(self, **kwargs):
        '''
        Parameters
        ----------
        exclude_terrain : bool, default: True
            If True the loss is only averaged over flow cells.
        loss_type : str, default: MSE
            Whether to use a L1 or a MSE based method.
        grid_size : list, default: [1,1,1]
            List containing the grid spacing in directions X, Y and Z of the dataset. [m]
        '''
        super(VelocityGradientLoss, self).__init__()

        parser = utils.KwargsParser(kwargs, 'VelocityGradientLoss')
        self._exclude_terrain = parser.get_safe('exclude_terrain', True, bool, True)
        self._loss_type = parser.get_safe('loss_type', 'MSE', str, True)
        self._grid_size = parser.get_safe('grid_size', [1, 1, 1], list, True)

        if (self._loss_type == 'MSE'):
            self._loss = torch.nn.MSELoss(reduction='none')
        elif (self._loss_type == 'L1'):
            self._loss = torch.nn.L1Loss(reduction='none')
        else:
            raise ValueError(
                'VelocityGradientLoss: unknown loss type ', self._loss_type
                )

    def forward(
            self, predicted, target, input, W=None, terrain_correction_factors=None
        ):
        '''
        Loss forward function.

        Parameters
        ----------
        predicted : dict
            Prediction dictionary
        target : torch.Tensor
            Target (label) tensor
        input : torch.Tensor
            Input tensor
        W : torch.Tensor or None, default: None
            Weighting tensor to give different weights for each cell individually.
        terrain_correction_factors : torch.Tensor or None, default: None
            Tensor with a correction factor for each sample in the batch.

        Returns
        -------
        loss : torch.Tensor
            Computed loss value
        '''
        if (predicted['pred'].shape != target.shape):
            raise ValueError(
                'VelocityGradientLoss: predicted and target do not have the same shape, pred:{}, target:{}'
                .format(predicted['pred'].shape, target.shape)
                )

        if (len(predicted['pred'].shape) != 5):
            raise ValueError(
                'VelocityGradientLoss: the loss is only defined for 5D data. Unsqueeze single samples!'
                )

        return self.compute_loss(
            predicted['pred'], target, input, W, terrain_correction_factors
            )

    def compute_loss(
            self, predicted, target, input, W=None, terrain_correction_factors=None
        ):
        '''
        Compute the loss according to the class settings.

        Parameters
        ----------
        predicted : dict
            Prediction dictionary
        target : torch.Tensor
            Target (label) tensor
        input : torch.Tensor
            Input tensor
        W : torch.Tensor or None, default: None
            Weighting tensor to give different weights for each cell individually.
        terrain_correction_factors : torch.Tensor or None, default: None
            Tensor with a correction factor for each sample in the batch.

        Returns
        -------
        loss : torch.Tensor
            Computed loss value
        '''
        terrain = input[:, 0:1]

        # compute gradients of prediction and target
        target_grad = utils.gradient(target, self._grid_size, terrain.squeeze(1))
        predicted_grad = utils.gradient(predicted, self._grid_size, terrain.squeeze(1))

        # compute physics loss for all elements, weight it by pixel, and average losses over each sample in batch
        loss = self._loss(predicted_grad, target_grad)

        if W is not None:
            loss *= W

        loss = loss.mean(tuple(range(1, len(predicted.shape))))

        # compute terrain correction factor for each sample in batch
        if self._exclude_terrain and terrain_correction_factors is not None:
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
        '''
        Computing the Kullback-Leibler divergence between two distributions.
        
        Assuming p and q are *univariate* gaussian (no cross-correlation terms):
        KL(p || q) = log(\sigma_q/\sigma_p) + (\sigma_p^2 + (\mu_p - \mu_q)^2)/(2\sigma_q^2) - 1/2
        Recall: log (x^2) = 2 log (x), so we can take the 1/2 out of the first term if we are given log variances
        
        Parameters
        ----------
        mean1 : torch.Tensor
            Mean of the first distribution
        logvar1 : torch.Tensor
            Logarithmic variance of the first distribution
        mean2 : torch.Tensor
            Mean of the second distribution
        logvar2 : torch.Tensor
            Logarithmic variance of the second distribution

        Returns
        -------
        loss : torch.Tensor
            Computed loss value
        '''
        return -0.5 * torch.sum(
            1.0 + logvar1 - logvar2 - (logvar1.exp() +
                                       (mean1 - mean2).pow(2)) / logvar2.exp()
            )

    def forward(
            self, predicted, target, input, W=None, terrain_correction_factors=None
        ):
        '''
        Loss forward function.

        Parameters
        ----------
        predicted : dict
            Prediction dictionary
        target : torch.Tensor
            Target (label) tensor
        input : torch.Tensor
            Input tensor
        W : torch.Tensor or None, default: None
            Weighting tensor to give different weights for each cell individually.
        terrain_correction_factors : torch.Tensor or None, default: None
            Tensor with a correction factor for each sample in the batch.

        Returns
        -------
        loss : torch.Tensor
            Computed loss value
        '''
        if ('distribution_mean' not in predicted.keys()):
            raise ValueError(
                'KLDivLoss: distribution_mean needs to be in the prediction dict'
                )

        if ('distribution_logvar' not in predicted.keys()):
            raise ValueError(
                'KLDivLoss: distribution_mean needs to be in the prediction dict'
                )

        if (
            predicted['distribution_mean'].shape !=
            predicted['distribution_logvar'].shape
            ):
            raise ValueError(
                'KLDivLoss: the mean and logvar need to have the same shape'
                )

        return self.compute_loss(
            predicted['distribution_mean'], predicted['distribution_logvar']
            )

    def compute_loss(self, mean, logvar):
        '''
        Compute the loss kldiv loss to a zero mean unit variance distribution.

        Parameters
        ----------
        mean : torch.Tensor
            Prediction dictionary
        logvar : torch.Tensor
            Target (label) tensor

        Returns
        -------
        loss : torch.Tensor
            Computed loss value
        '''
        # We assume the loss is being computed as KLD with respect to zero mean, unit variance (log(1) = 0)
        return self.kld_gaussian(
            mean,
            logvar,
            mean2=torch.zeros(1, device=mean.device),
            logvar2=torch.zeros(1, device=mean.device)
            )


#------------------------------------------ Gaussian Log Likelihood Loss  ----------------------------------------------
class GaussianLogLikelihoodLoss(Module):
    '''
    Gaussian Log Likelihood Loss according to https://arxiv.org/pdf/1705.07115.pdf
    '''

    def __init__(self, **kwargs):
        '''
        Parameters
        ----------
        exclude_terrain : bool, default: True
            If True the loss is only averaged over flow cells.
        eps : float, default: 1e-8
            Minimum absolute uncertainty value to avoid numerical issues
        '''
        super(GaussianLogLikelihoodLoss, self).__init__()

        parser = utils.KwargsParser(kwargs, 'GaussianLogLikelihoodLoss')
        self._exclude_terrain = parser.get_safe('exclude_terrain', True, bool, True)
        self._eps = parser.get_safe('uncertainty_loss_eps', 1e-8, float, True)

    def forward(
            self, predicted, target, input, W=None, terrain_correction_factors=None
        ):
        '''
        Loss forward function.

        Parameters
        ----------
        predicted : dict
            Prediction dictionary
        target : torch.Tensor
            Target (label) tensor
        input : torch.Tensor
            Input tensor
        W : torch.Tensor or None, default: None
            Weighting tensor to give different weights for each cell individually.
        terrain_correction_factors : torch.Tensor or None, default: None
            Tensor with a correction factor for each sample in the batch.

        Returns
        -------
        loss : torch.Tensor
            Computed loss value
        '''
        if (predicted['pred'].shape != target.shape):
            raise ValueError(
                'GaussianLogLikelihoodLoss: predicted and target do not have the same shape, pred:{}, target:{}'
                .format(predicted['pred'].shape, target.shape)
                )

        if 'logvar' not in predicted.keys():
            raise ValueError(
                'GaussianLogLikelihoodLoss: the uncertainty needs to be predicted and present in the output dict (logvar)'
                )

        if (len(predicted['pred'].shape) != 5):
            raise ValueError(
                'GaussianLogLikelihoodLoss: the loss is only defined for 5D data. Unsqueeze single samples!'
                )

        if (predicted['pred'].shape != predicted['logvar'].shape):
            raise ValueError('The variance and the mean need to have the same shape')

        return self.compute_loss(
            predicted['pred'], predicted['logvar'], target, input, W,
            terrain_correction_factors
            )

    def compute_loss(
            self,
            mean,
            log_variance,
            target,
            input,
            W=None,
            terrain_correction_factors=None
        ):
        '''
        Compute the loss according to the class settings.

        Parameters
        ----------
        mean : torch.Tensor
            Predicted mean tensor
        log_variance : torch.Tensor
            Predicted variance tensor
        target : torch.Tensor
            Target (label) tensor
        input : torch.Tensor
            Input tensor
        W : torch.Tensor or None, default: None
            Weighting tensor to give different weights for each cell individually.
        terrain_correction_factors : torch.Tensor or None, default: None
            Tensor with a correction factor for each sample in the batch.

        Returns
        -------
        loss : torch.Tensor
            Computed loss value
        '''
        mean_error = mean - target

        # compute loss for all elements
        loss = 0.5 * log_variance + (mean_error *
                                     mean_error) / log_variance.exp().clamp(
                                         min=self._eps, max=1e10
                                         )

        if W is not None:
            loss *= W

        # average weighted loss over each sample in batch
        loss = loss.mean(tuple(range(1, len(mean_error.shape))))

        # compute terrain correction factor for each sample in batch
        if self._exclude_terrain and terrain_correction_factors is not None:
            # apply terrain correction factor to loss of each sample in batch
            loss *= terrain_correction_factors

        # return batchwise mean of loss
        return loss.mean()
