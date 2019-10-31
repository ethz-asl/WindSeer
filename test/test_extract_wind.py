import nn_wind_prediction.cosmo as cosmo

out = cosmo.extract_cosmo_data('../wind_analysis/data/cosmo-1_ethz_fcst_2018112315.nc', 46.947225, 8.693297, 12,
                               terrain_file='../wind_analysis/data/cosmo-1_ethz_ana_const.nc')

print('Extraction successful')
