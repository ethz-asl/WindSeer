import utils
import time
import matplotlib.pyplot as plt

db = utils.MyDataset('data/train.zip',  scaling_ux = 10.0, scaling_uz = 2.5, scaling_nut = 10.0)

start_time = time.time()
for i in range(len(db)):
    input, label = db[i]
print("INFO: Time to get all samples in the dataset took %s seconds" % (time.time() - start_time))

fh_in, ah_in = plt.subplots(3, 2)
fh_in.set_size_inches([6.2, 10.2])

h_ux_in = ah_in[0][0].imshow(input[1,:,:], origin='lower', vmin=label[0,:,:].min(), vmax=label[0,:,:].max())
h_uz_in = ah_in[0][1].imshow(input[2,:,:], origin='lower', vmin=label[1,:,:].min(), vmax=label[1,:,:].max())
ah_in[0][0].set_title('Ux in')
ah_in[0][1].set_title('Uz in')
fh_in.colorbar(h_ux_in, ax=ah_in[0][0])
fh_in.colorbar(h_uz_in, ax=ah_in[0][1])

h_ux_in = ah_in[1][0].imshow(label[0,:,:], origin='lower', vmin=label[0,:,:].min(), vmax=label[0,:,:].max())
h_uz_in = ah_in[1][1].imshow(label[1,:,:], origin='lower', vmin=label[1,:,:].min(), vmax=label[1,:,:].max())
ah_in[1][0].set_title('Ux label')
ah_in[1][1].set_title('Uz label')
fh_in.colorbar(h_ux_in, ax=ah_in[1][0])
fh_in.colorbar(h_uz_in, ax=ah_in[1][1])

h_ux_in = ah_in[2][0].imshow(input[0,:,:], origin='lower', vmin=input[0,:,:].min(), vmax=input[0,:,:].max())
h_uz_in = ah_in[2][1].imshow(label[2,:,:], origin='lower', vmin=label[2,:,:].min(), vmax=label[2,:,:].max())
ah_in[2][0].set_title('Terrain')
ah_in[2][1].set_title('Turbulence viscosity label')
fh_in.colorbar(h_ux_in, ax=ah_in[2][0])
fh_in.colorbar(h_uz_in, ax=ah_in[2][1])

plt.show()
