import yaml
import os

class EDNNParameters(object):

    def __init__(self, yaml_config):
        self.yaml_file = yaml_config
        run_parameters = self.load_yaml(self.yaml_file)
        self.model = run_parameters['model']
        self.data = run_parameters['data']
        self.run = run_parameters['run']
        self.name = self._build_name()


    @staticmethod
    def load_yaml(file):
        print("Using YAML config: {0}".format(file))
        with open(file, 'rt') as fh:
            run_parameters = yaml.safe_load(fh)

        # decide if turbulence is used (somewhat a hack maybe there is something better in the future)
        if run_parameters['model']['d3']:
            if run_parameters['model']['n_output_layers'] > 3:
                run_parameters['model']['use_turbulence'] = True
            else:
                run_parameters['model']['use_turbulence'] = False
        else:
            if run_parameters['model']['n_output_layers'] > 2:
                run_parameters['model']['use_turbulence'] = True
            else:
                run_parameters['model']['use_turbulence'] = False

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
        name = self.model['name_prefix']+'_'
        name += self._letter_switch(self.model['interpolation_mode'])
        name += self._letter_switch(self.model['align_corners'], 'a')
        name += self._letter_switch(self.model['skipping'], 'k')
        name += 'd{0:d}'.format(self.model['n_downsample_layers'])
        name += self._letter_switch(self.model['pooling_method'])
        name += self._letter_switch(self.model['use_fc_layers'], 'f')
        name += '{0:d}'.format(self.model['fc_scaling'])
        name += self._letter_switch(self.model['use_mapping_layer'], 'm')
        name += self._letter_switch(self.model['skipping'], 'k')
        return name


    def MyDataset_kwargs(self):
        return {'stride_hor': self.data['stride_hor'],
                    'stride_vert': self.data['stride_vert'],
                    'turbulence_label': self.model['use_turbulence'],
                    'scaling_uhor': self.data['uhor_scaling'],
                    'scaling_uz': self.data['uz_scaling'],
                    'scaling_nut': self.data['turbulence_scaling']}

    def model3d_kwargs(self):
        return {'n_input_layers': self.model['n_input_layers'],
                'n_output_layers': self.model['n_output_layers'],
                'n_x': self.model['n_x'],
                'n_y': self.model['n_y'],
                'n_z': self.model['n_z'],
                'n_downsample_layers': self.model['n_downsample_layers'],
                'interpolation_mode': self.model['interpolation_mode'],
                'align_corners': self.model['align_corners'],
                'skipping': self.model['skipping'],
                'use_terrain_mask': self.model['use_terrain_mask'],
                'pooling_method': self.model['pooling_method'],
                'use_mapping_layer': self.model['use_mapping_layer'],
                'use_fc_layers': self.model['use_fc_layers'],
                'fc_scaling': self.model['fc_scaling']}

    def model2d_kwargs(self):
        return {'n_input_layers': self.model['n_input_layers'],
                'interpolation_mode': self.model['interpolation_mode'],
                'align_corners': self.model['align_corners'],
                'skipping': self.model['skipping'],
                'predict_turbulence': self.model['use_turbulence']}

    def save(self, dir=None):
        if dir is None:
            dir = self.name
        with open(os.path.join(dir, 'params.yaml'), 'wt') as fh:
            yaml.safe_dump({'run': self.run, 'data': self.data, 'model': self.model}, fh)

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
        print('Model Settings:')
        print('\tModel name:\t\t', self.name)
        print('\t3D:\t\t\t', self.model['d3'])
        print('\tNumber of inputs:\t', self.model['n_input_layers'])
        print('\tNumber of outputs:\t', self.model['n_output_layers'])
        print('\tNx:\t\t\t', self.model['n_x'])
        print('\tNy:\t\t\t', self.model['n_y'])
        print('\tNz:\t\t\t', self.model['n_z'])
        print('\tNumber conv layers:\t', self.model['n_downsample_layers'])
        print('\tInterpolation mode:\t', self.model['interpolation_mode'])
        print('\tAlign corners:\t\t', self.model['align_corners'])
        print('\tSkip connection:\t', self.model['skipping'])
        print('\tUse terrain mask:\t', self.model['use_terrain_mask'])
        print('\tPooling method:\t\t', self.model['pooling_method'])
        print('\tUse fc layers:\t\t', self.model['use_fc_layers'])
        print('\tFC layer scaling:\t', self.model['fc_scaling'])
        print('\tUse mapping layer:\t', self.model['use_mapping_layer'])
        print(' ')
        print('Dataset Settings:')
        print('\tUhor scaling:\t\t', self.data['uhor_scaling'])
        print('\tUz scaling:\t\t', self.data['uz_scaling'])
        print('\tTurbulence scaling:\t', self.data['turbulence_scaling'])
        print('\tHorizontal stride:\t', self.data['stride_hor'])
        print('\tVertical stride:\t', self.data['stride_vert'])