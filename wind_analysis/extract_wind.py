import nn_wind_prediction.cosmo as cosmo

out = cosmo.extract_cosmo_data('data/cosmo-1_ethz_fcst_2018112315.nc', 46.947225, 8.693297, 12,
                               terrain_file='data/cosmo-1_ethz_ana_const.nc')

import pdb
pdb.set_trace()
