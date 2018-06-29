import numpy as np
import matplotlib.pyplot as plt

U_ref = np.array([1.0, 10.0])
Z_ref = np.array([20.0, 20.0])
z_0 = np.array([0.1, 0.1])
z_ground = np.array([0.0, 0.0])

kappa = 0.41
C_mu = 0.09

z = np.linspace(0.1, 500.0, 100)
z = np.tile(z, (len(U_ref), 1)).T

U_star = kappa*U_ref/(np.log((Z_ref+z_0)/z_0))

U = U_star/kappa*np.log((z-z_ground+z_0)/z_0)
k = (U_star**2)/np.sqrt(C_mu)
epsilon = (U_star**3)/(kappa*(z - z_ground + z_0))

fh, ah = plt.subplots(1, 3)
ah[0].plot(U, z)
ah[0].plot([U_ref], [Z_ref], 'r.')
ah[1].plot(epsilon, z)
ah[1].set_x
ah[0].set_ylabel('Height above ground (m)')
ah[0].set_xlabel('Wind speed (m/s)')
ah[1].set_xlabel('Epsilon')
print 'U* = {1}, k = {0}'.format(k, U_star)
plt.show(block=False)

