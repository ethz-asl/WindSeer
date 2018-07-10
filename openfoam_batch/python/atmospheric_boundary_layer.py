from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt


U_ref = np.array([10.0, 10.0])
Z_ref = np.array([10.0, 10.0])
z_0 = np.array([3e-4, 0.01])
z_ground = np.array([0.0, 0.0])

kappa = 0.41
C_mu = 0.09

z = np.linspace(0.1, 100.0, 100)
z = np.tile(z, (len(U_ref), 1)).T

U_star = kappa*U_ref/(np.log((Z_ref+z_0)/z_0))

U = U_star/kappa*np.log((z-z_ground+z_0)/z_0)
k = (U_star**2)/np.sqrt(C_mu)
epsilon = (U_star**3)/(kappa*(z - z_ground + z_0))

fh, ah = plt.subplots(1, 2)
ah[0].plot(U, z)
ah[0].plot([U_ref], [Z_ref], 'r.')
ah[1].plot(epsilon, z)
ah[0].set_ylabel('Height above ground (m)')
ah[0].set_xlabel('Wind speed (m/s)')
ah[1].set_xlabel('$\epsilon$')
print('U* = {1}, k = {0}'.format(k, U_star))


# Stuff for Bolund case:
b_kappa = 0.4
b_Z_ref = 10.0
b_U_star = 0.4
b_z_0 = 6.4e-4      # 3.5e-4,  3.2e-4,  3.0e-4
b_U_ref = b_U_star/b_kappa*np.log((b_Z_ref+b_z_0)/(b_z_0))
b_k = 5.8*b_U_star**2
b_Cmu = (b_U_star**2/b_k)**2
print("BOLUND: U_ref: {0:0.6f}, TKE: {1:0.6f}, Cmu: {2:0.6f}".format(b_U_ref, b_k, b_Cmu))
b_z = np.append(np.logspace(-3, 1, 50), np.linspace(10.5, 20, 50))
b_s = b_U_star/b_kappa*np.log((b_z+b_z_0)/b_z_0)
fhb, ahb = plt.subplots(1, 2)
b_epsilon = (b_U_star**3)/(b_kappa*(b_z + b_z_0))

b_L = 1.0 # Turbulence length. 1,0m? 10m?
b_eps_bound = np.power(b_Cmu, 0.75) * np.power(b_k, 1.5) / b_L
ahb[0].plot(b_s/b_U_star, b_z)
ahb[0].plot([b_U_ref/b_U_star], [b_Z_ref], 'r.')
ahb[1].plot(b_epsilon, b_z)
ahb[0].set_ylabel('$z_{agl}$ (m)')
ahb[0].set_xlabel('$s/u_{*0}$ (m/s)')
ahb[1].set_xlabel('$\epsilon$')
ahb[0].set_xlim([0, 30.0])
ahb[1].set_xscale('log')

plt.show(block=False)

