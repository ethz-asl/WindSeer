import numpy as np
import argparse

def abl_k_eps(U_ref, z=100.0, Z_ref=20.0, z_0=0.1, z_ground=0.0, kappa=0.41, C_mu=0.09):
    # This is from the boundary layer calculations, but the numbers seem low...
    U_star = kappa*U_ref/(np.log((Z_ref+z_0)/z_0))
    k = (U_star**2)/np.sqrt(C_mu)
    epsilon = (U_star**3)/(kappa*(z - z_ground + z_0))

    return k, epsilon

# def default_k_eps()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate k, epsilon estimates for atmospheric boundary layer')
    parser.add_argument('-U', '--U-ref', required=True, type=float, help='U_ref wind speed (m/s)')
    args = parser.parse_args()

    k, eps = abl_k_eps(args.U_ref)
    print '{0:0.6f} {1:0.6f}'.format(k, eps)