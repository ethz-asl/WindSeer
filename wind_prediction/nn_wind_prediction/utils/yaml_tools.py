from __future__ import print_function

import os
import yaml

class EDNNParameters(object):

    def __init__(self, yaml_config):
        self.yaml_file = yaml_config
        run_parameters = self.load_yaml(self.yaml_file)
        self.model = run_parameters['model']
        self.data = run_parameters['data']
        self.run = run_parameters['run']
        self.loss = run_parameters['loss']
        self.name = self._build_name()


    @staticmethod
    def load_yaml(file):
        print("Using YAML config: {0}".format(file))
        with open(file, 'rt') as fh:
            run_parameters = yaml.safe_load(fh)

        label_channels = run_parameters['data']['label_channels']

        run_parameters['model']['model_args']['use_turbulence'] = 'turb' in label_channels
        run_parameters['model']['model_args']['use_pressure'] = 'p' in label_channels
        run_parameters['model']['model_args']['use_epsilon'] = 'epsilon' in label_channels
        run_parameters['model']['model_args']['use_nut'] = 'nut' in label_channels
        run_parameters['model']['model_args']['n_epochs'] = run_parameters['run']['n_epochs']

        return run_parameters

    @staticmethod
    def _letter_switch(value, letter=None):
        if isinstance(value, str):
            letter=value[0]
            value=letter.isupper()
        if value:
            return letter.upper()
        else:
            return letter.lower()

    def _build_name(self):
        name = self.model['name_prefix']

        # don't use it for now as the order is random
#         name = self.model['name_prefix']+'_'
#         for key in self.model['model_args']:
#             print(key)
#             try:
#                 if isinstance(self.model['model_args'][key], bool):
#                     name += self._letter_switch(self.model['model_args'][key], key[0])
#
#                 else:
#                     val = float(self.model['model_args'][key])
#                     name += self._letter_switch(False, key[0])
#                     name += '{0:d}'.format(self.model['model_args'][key])
#
#             except:
#                 if isinstance(self.model['model_args'][key], str):
#                     name += self._letter_switch(self.model['model_args'][key])

        return name

    def Dataset_kwargs(self):
        return {'stride_hor': self.data['stride_hor'],
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
                'autoscale': self.data['autoscale']}

    def model_kwargs(self):
        return self.model['model_args']

    def pass_grid_size_to_loss(self, grid_size):
        '''
        Small function to pass the grid size to the kwargs of the loss functions that need it.
        '''
        for i, loss_component in enumerate(self.loss['loss_components']):
            if 'DivergenceFree' in loss_component or 'VelocityGradient' in loss_component:
                self.loss[loss_component + '_kwargs']['grid_size'] = grid_size

    def save(self, dir=None):
        if dir is None:
            dir = self.name
        with open(os.path.join(dir, 'params.yaml'), 'wt') as fh:
            yaml.safe_dump({'run': self.run, 'loss': self.loss, 'data': self.data, 'model': self.model}, fh)

    def print(self):
        print('Train Settings:')
        print('\tWarm start:\t\t', self.run['warm_start'])
        print('\tLearning rate initial:\t', self.run['learning_rate_initial'])
        print('\tLearning rate step size:', self.run['learning_rate_decay_step_size'])
        print('\tLearning rate decay:\t', self.run['learning_rate_decay'])
        print('\tBatchsize:\t\t', self.run['batchsize'])
        print('\tEpochs:\t\t\t', self.run['n_epochs'])
        print('\tMinibatch epoch loss:\t', self.run['minibatch_epoch_loss'])
        print(' ')
        print('Loss Settings:')
        print('\tLoss component(s):\t', self.loss['loss_components'])
        if len(self.loss['loss_components']) > 1:
            print('\tLearn loss scaling factors:\t', self.loss['learn_scaling'])
        for i, loss_component in enumerate(self.loss['loss_components']):
            print('\t'+loss_component, 'kwargs :',self.loss[loss_component + '_kwargs'])
        print(' ')
        print('Model Settings:')
        print('\t Model prefix:\t\t', self.model['name_prefix'])
        print('\t Model type:\t\t', self.model['model_type'])
        print('\t Model args:')
        print('\t\t', self.model['model_args'])
        print(' ')
        print('Dataset Settings:')
        print('\tInput channels:\t',self.data['input_channels'])
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
    def __init__(self, yaml_config, subdict=None):
        self.subdict = subdict
        self.yaml_file = yaml_config
        if subdict is None:
            self.params = self._load_yaml(self.yaml_file)
        else:
            self.params = self._load_yaml(self.yaml_file)[subdict]

    @staticmethod
    def _load_yaml(file, str='Using YAML config: '):
        print("{0} {1}".format(str, file))
        with open(file, 'rt') as fh:
            run_parameters = yaml.safe_load(fh)
        return run_parameters

    def _save(self, dir=None, file='params.yaml'):
        if dir is None:
            dir = self.name
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
    def __init__(self, yaml_config):
        super(COSMOParameters, self).__init__(yaml_config, subdict='cosmo')
        if ('time' not in self.params) or (self.params['time'].lower() == 'auto'):
            try:
                # Assume time is last two digits of filename
                bn = os.path.splitext(os.path.basename(self.params['file']))[0]
                self.params['time'] = int(bn[-2:])
            except:
                print('Automatic time extraction failed on file: {0}. Setting time to 00'.format(self.params['file']))
                self.params['time'] = 0

    def load_yaml(self, file):
        return self._load_yaml(file, "Using YAML COSMO config: ")

    def save(self, dir=None):
        self._save(dir, 'cosmo.yaml')

    def print(self):
        self._print('COSMO parameters:')

    def get_cosmo_time(self, target_time):
        delta_t = target_time - self.params['time']
        if delta_t < 0:
            print('WARNING: Requested time {0} is before COSMO time {1}, returning time index 0'.format(target_time, self.params['time']))
        return delta_t


class UlogParameters(BasicParameters):

    def __init__(self, yaml_file):
        super(UlogParameters, self).__init__(yaml_file, subdict='ulog')

    def load_yaml(self, file):
        return self._load_yaml(file, "Using YAML ulog config: ")

    def save(self, dir=None):
        self._save(dir, 'ulog.yaml')

    def print(self):
        self._print('Ulog parameters:')
