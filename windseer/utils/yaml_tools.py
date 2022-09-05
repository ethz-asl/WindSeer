from __future__ import print_function

import os
import yaml


class WindseerParams(object):
    '''
    Class handling and loading the parameter for windseer
    from a yaml file.
    '''

    def __init__(self, yaml_config, verbose=True):
        '''
        Parameters
        ----------
        yaml_config : str
            Filename of the yaml file
        verbose : bool, default: True
            Print additional information to the console
        '''
        self.yaml_file = yaml_config
        run_parameters = self.load_yaml(self.yaml_file, verbose)
        self.model = run_parameters['model']
        self.data = run_parameters['data']
        self.run = run_parameters['run']
        self.loss = run_parameters['loss']
        self.name = self._build_name()

    @staticmethod
    def load_yaml(file, verbose=True):
        '''
        Loading the parameter from a yaml file.

        Parameters
        ----------
        file : str
            Filename of the yaml file
        verbose : bool, default: True
            Print additional information to the console

        Returns
        -------
        params : dict
            Loaded parameter dictionary
        '''
        if verbose:
            print("Using YAML config: {0}".format(file))
        with open(file, 'rt') as fh:
            run_parameters = yaml.safe_load(fh)

        label_channels = run_parameters['data']['label_channels']

        run_parameters['model']['model_args']['use_turbulence'
                                              ] = 'turb' in label_channels
        run_parameters['model']['model_args']['use_pressure'] = 'p' in label_channels
        run_parameters['model']['model_args']['use_epsilon'
                                              ] = 'epsilon' in label_channels
        run_parameters['model']['model_args']['use_nut'] = 'nut' in label_channels
        run_parameters['model']['model_args']['n_epochs'] = run_parameters['run'][
            'n_epochs']

        return run_parameters

    def _build_name(self):
        '''
        Get the name of the configuration

        Returns
        -------
        name : str
            Configuration name
        '''
        name = self.model['name_prefix']
        return name

    def Dataset_kwargs(self):
        '''
        Get the dataset kwargs.

        Returns
        -------
        kwargs : dict
            Dataset kwargs
        '''
        kwargs = {
            'stride_hor': self.data['stride_hor'],
            'stride_vert': self.data['stride_vert'],
            'input_channels': self.data['input_channels'],
            'label_channels': self.data['label_channels'],
            'scaling_ux': self.data['ux_scaling'],
            'scaling_uy': self.data['uy_scaling'],
            'scaling_uz': self.data['uz_scaling'],
            'scaling_turb': self.data['turb_scaling'],
            'scaling_p': self.data['p_scaling'],
            'scaling_epsilon': self.data['epsilon_scaling'],
            'scaling_nut': self.data['nut_scaling'],
            'scaling_terrain': self.data['terrain_scaling'],
            'input_mode': self.data['input_mode'],
            'nx': self.model['model_args']['n_x'],
            'ny': self.model['model_args']['n_y'],
            'nz': self.model['model_args']['n_z'],
            'autoscale': self.data['autoscale'],
            'loss_weighting_fn': self.loss['loss_weighting_fn'],
            'loss_weighting_clamp': self.loss['loss_weighting_clamp']
            }

        if 'verbose' in self.data.keys():
            kwargs['verbose'] = self.data['verbose']
        else:
            kwargs['verbose'] = True

        if 'return_name' in self.data.keys():
            kwargs['return_name'] = self.data['return_name']
        else:
            kwargs['return_name'] = False

        if 'device' in self.data.keys():
            kwargs['device'] = self.data['device']
        else:
            kwargs['device'] = 'cpu'

        # check if the keys exist for the more recently introduced parameter
        # to keep backwards compatibility
        if 'additive_gaussian_noise' in self.data.keys():
            kwargs['additive_gaussian_noise'] = self.data['additive_gaussian_noise']

        if 'max_gaussian_noise_std' in self.data.keys():
            kwargs['max_gaussian_noise_std'] = self.data['max_gaussian_noise_std']

        if 'n_turb_fields' in self.data.keys():
            kwargs['n_turb_fields'] = self.data['n_turb_fields']

        if 'max_normalized_turb_scale' in self.data.keys():
            kwargs['max_normalized_turb_scale'] = self.data['max_normalized_turb_scale']

        if 'max_normalized_bias_scale' in self.data.keys():
            kwargs['max_normalized_bias_scale'] = self.data['max_normalized_bias_scale']

        if 'only_z_velocity_bias' in self.data.keys():
            kwargs['only_z_velocity_bias'] = self.data['only_z_velocity_bias']

        if 'max_fraction_of_sparse_data' in self.data.keys():
            kwargs['max_fraction_of_sparse_data'] = self.data[
                'max_fraction_of_sparse_data']

        if 'use_system_random' in self.data.keys():
            kwargs['use_system_random'] = self.data['use_system_random']

        if 'trajectory_min_length' in self.data.keys():
            kwargs['trajectory_min_length'] = self.data['trajectory_min_length']

        if 'trajectory_max_length' in self.data.keys():
            kwargs['trajectory_max_length'] = self.data['trajectory_max_length']

        if 'trajectory_min_segment_length' in self.data.keys():
            kwargs['trajectory_min_segment_length'] = self.data[
                'trajectory_min_segment_length']

        if 'trajectory_max_segment_length' in self.data.keys():
            kwargs['trajectory_max_segment_length'] = self.data[
                'trajectory_max_segment_length']

        if 'trajectory_step_size' in self.data.keys():
            kwargs['trajectory_step_size'] = self.data['trajectory_step_size']

        if 'trajectory_max_iter' in self.data.keys():
            kwargs['trajectory_max_iter'] = self.data['trajectory_max_iter']

        if 'trajectory_start_weighting_mode' in self.data.keys():
            kwargs['trajectory_start_weighting_mode'] = self.data[
                'trajectory_start_weighting_mode']

        if 'trajectory_length_short_focus' in self.data.keys():
            kwargs['trajectory_length_short_focus'] = self.data[
                'trajectory_length_short_focus']

        if 'input_smoothing' in self.data.keys():
            kwargs['input_smoothing'] = self.data['input_smoothing']

        if 'input_smoothing_interpolation' in self.data.keys():
            kwargs['input_smoothing_interpolation'] = self.data[
                'input_smoothing_interpolation']

        if 'input_smoothing_interpolation_linear' in self.data.keys():
            kwargs['input_smoothing_interpolation_linear'] = self.data[
                'input_smoothing_interpolation_linear']

        return kwargs

    def model_kwargs(self):
        '''
        Get the model kwargs.

        Returns
        -------
        kwargs : dict
            Model kwargs
        '''
        return self.model['model_args']

    def pass_grid_size_to_loss(self, grid_size):
        '''
        Pass the grid size to the kwargs of the loss functions that need it.


        Parameters
        ----------
        grid_size : list of int
            Grid size in meter
        '''
        for i, loss_component in enumerate(self.loss['loss_components']):
            if 'DivergenceFree' in loss_component or 'VelocityGradient' in loss_component:
                self.loss[loss_component + '_kwargs']['grid_size'] = grid_size

    def save(self, dir=None):
        '''
        Save the parameter to a yaml file.

        Parameters
        ----------
        dir : str or None, default: None
            Folder name, if not set the model folder is used
        '''
        if dir is None:
            dir = self.name
        with open(os.path.join(dir, 'params.yaml'), 'wt') as fh:
            yaml.safe_dump({
                'run': self.run,
                'loss': self.loss,
                'data': self.data,
                'model': self.model
                }, fh)

    def print(self):
        '''
        Print the parameter to the console.
        '''
        print('Train Settings:')
        print('\tWarm start:\t\t', self.run['warm_start'])
        print('\tLearning rate step size:', self.run['learning_rate_decay_step_size'])
        print('\tLearning rate decay:\t', self.run['learning_rate_decay'])
        print('\tBatchsize:\t\t', self.run['batchsize'])
        print('\tEpochs:\t\t\t', self.run['n_epochs'])
        print('\tMinibatch epoch loss:\t', self.run['minibatch_epoch_loss'])
        print(' ')
        print('Loss Settings:')
        print('\tLoss weighting fn:\t', self.loss['loss_weighting_fn'])
        print('\tLoss component(s):\t', self.loss['loss_components'])
        if len(self.loss['loss_components']) > 1:
            print('\tLearn loss scaling factors:\t', self.loss['learn_scaling'])
        for i, loss_component in enumerate(self.loss['loss_components']):
            print(
                '\t' + loss_component, 'kwargs :', self.loss[loss_component + '_kwargs']
                )
        print(' ')
        print('Model Settings:')
        print('\t Model prefix:\t\t', self.model['name_prefix'])
        print('\t Model type:\t\t', self.model['model_type'])
        print('\t Model args:')
        print('\t\t', self.model['model_args'])
        print(' ')
        print('Optimizer Settings:')
        print('\t Optimizer type:\t', self.run['optimizer_type'])
        print('\t Optimizer args:')
        print('\t\t', self.run['optimizer_kwargs'])
        print(' ')
        print('Dataset Settings:')
        print('\tInput channels:\t', self.data['input_channels'])
        print('\tLabel channels:\t', self.data['label_channels'])
        print('\tUx scaling:\t\t', self.data['ux_scaling'])
        print('\tUy scaling:\t\t', self.data['uy_scaling'])
        print('\tUz scaling:\t\t', self.data['uz_scaling'])
        print('\tTurbulence scaling:\t', self.data['turb_scaling'])
        print('\tPressure scaling:\t', self.data['p_scaling'])
        print('\tEpsilon scaling:\t', self.data['epsilon_scaling'])
        print('\tNut scaling:\t\t', self.data['nut_scaling'])
        print('\tHorizontal stride:\t', self.data['stride_hor'])
        print('\tVertical stride:\t', self.data['stride_vert'])
        print('\tAugmentation mode:\t', self.data['augmentation_mode'])
        print('\tAugmentation params:\t', self.data['augmentation_kwargs'])


class BasicParameters(object):
    '''
    Base class for handling yaml parameter files.
    '''

    def __init__(self, yaml_config, subdict=None):
        '''
        Loading the parameter from a yaml file.

        Parameters
        ----------
        yaml_config : str
            Filename of the yaml file
        subdict : str or None, default: None
            Only extract a subgroup of the parameter file
        '''
        self.subdict = subdict
        self.yaml_file = yaml_config
        if subdict is None:
            self.params = self._load_yaml(self.yaml_file)
        else:
            self.params = self._load_yaml(self.yaml_file)[subdict]

    @staticmethod
    def _load_yaml(file, str='Using YAML config: '):
        '''
        Loading the parameter from a yaml file.

        Parameters
        ----------
        file : str
            Filename of the yaml file
        str : str
            Information string displayed when loading the file

        Returns
        -------
        params : dict
            Loaded parameter dictionary
        '''
        print("{0} {1}".format(str, file))
        with open(file, 'rt') as fh:
            run_parameters = yaml.safe_load(fh)
        return run_parameters

    def _save(self, dir, file='params.yaml'):
        '''
        Saving the parameter to a yaml file.

        Parameters
        ----------
        dir : str
            Directory name where to store the params
        file : str, default: params.yaml
            Filename
        '''
        with open(os.path.join(dir, file), 'wt') as fh:
            if self.subdict is None:
                yaml.safe_dump(fh)
            else:
                yaml.safe_dump({self.subdict: self.params}, fh)

    def _print(self, header_str='Parameters:'):
        print(header_str)
        for key, item in self.params.items():
            print('\t{0}:\t{1}'.format(key, item))


class COSMOParameters(BasicParameters):
    '''
    Class handling yaml file containing the COSMO params.
    '''

    def __init__(self, yaml_config):
        '''
        Parameters
        ----------
        yaml_config : str
            Filename of the yaml file
        '''
        super(COSMOParameters, self).__init__(yaml_config, subdict='cosmo')
        try:
            auto_time = self.params['time'].lower() == 'auto'
        except:
            auto_time = False

        if ('time' not in self.params) or (auto_time):
            try:
                # Assume time is last two digits of filename
                bn = os.path.splitext(os.path.basename(self.params['file']))[0]
                self.params['time'] = int(bn[-2:])
            except:
                print(
                    'Automatic time extraction failed on file: {0}. Setting time to 00'
                    .format(self.params['file'])
                    )
                self.params['time'] = 0

    def load_yaml(self, file):
        '''
        Loading the parameter from a yaml file.

        Parameters
        ----------
        file : str
            Filename of the yaml file

        Returns
        -------
        params : dict
            Loaded parameter dictionary
        '''
        return self._load_yaml(file, "Using YAML COSMO config: ")

    def save(self, dir=None):
        '''
        Saving the parameter to a yaml file.
        The filename is set to cosmo.yaml

        Parameters
        ----------
        dir : str
            Directory name where to store the params
        '''
        self._save(dir, 'cosmo.yaml')

    def print(self):
        '''
        Print the parameter to the console.
        '''
        self._print('COSMO parameters:')

    def get_cosmo_time(self, target_time):
        '''
        Get the relative time of the requested COSMO data

        Parameters
        ----------
        target_time : int
            Target time

        Returns
        -------
        delta_t : int
            Relative timestamp
        '''
        delta_t = target_time - self.params['time']
        if delta_t < 0:
            print(
                'WARNING: Requested time {0} is before COSMO time {1}, returning time index 0'
                .format(target_time, self.params['time'])
                )
        return delta_t


class UlogParameters(BasicParameters):
    '''
    Class handling yaml file containing the ulog params.
    '''

    def __init__(self, yaml_file):
        '''
        Parameters
        ----------
        yaml_file : str
            Filename of the yaml file
        '''
        super(UlogParameters, self).__init__(yaml_file, subdict='ulog')

    def load_yaml(self, file):
        '''
        Loading the parameter from a yaml file.

        Parameters
        ----------
        file : str
            Filename of the yaml file

        Returns
        -------
        params : dict
            Loaded parameter dictionary
        '''
        return self._load_yaml(file, "Using YAML ulog config: ")

    def save(self, dir=None):
        '''
        Saving the parameter to a yaml file.
        The filename is set to ulog.yaml

        Parameters
        ----------
        dir : str
            Directory name where to store the params
        '''
        self._save(dir, 'ulog.yaml')

    def print(self):
        '''
        Print the parameter to the console.
        '''
        self._print('Ulog parameters:')
