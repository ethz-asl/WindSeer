from setuptools import setup, find_packages

setup(
    name='nn_wind_prediction',
    version='0.0.1',
    description='A neural network and the utilities to predict the wind',
    license='MIT',
    packages=['nn_wind_prediction', 'nn_wind_prediction.models', 'nn_wind_prediction.data',
              'nn_wind_prediction.nn', 'nn_wind_prediction.utils'],
    author='Achermann Florian',
    author_email='florian.achermann@mavt.ethz.ch',
    keywords=['neural network', 'pytorch'],
    url='https://github.com/ethz-asl/intel_wind'
)
