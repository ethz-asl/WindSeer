import numpy as np
import argparse


def abl_k_eps(U_ref, z=100.0, Z_ref=20.0, z_0=0.1, z_ground=0.0, kappa=0.41, C_mu=0.09):
    # This is from the boundary layer calculations, but the numbers seem low...
    U_star = kappa*U_ref/(np.log((Z_ref+z_0)/z_0))
    k = (U_star**2)/np.sqrt(C_mu)
    epsilon = (U_star**3)/(kappa*(z - z_ground + z_0))
    return k, epsilon


def default_k_eps(U_ref, I=0.01, l=10.0, z=100.0, Z_ref=20.0, z_0=0.1, z_ground=0.0, kappa=0.41, C_mu=0.09):
    U_star = kappa * U_ref / (np.log((Z_ref + z_0) / z_0))
    U_z = U_star/kappa * np.log((z - z_ground + z_0) / z_0)
    k = 1.5*(U_z*I)**2
    epsilon = C_mu*(k**(1.5)/l)
    return k, epsilon


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate k, epsilon estimates for atmospheric boundary layer')
    parser.add_argument('-U', '--U-ref', required=True, type=float, help='U_ref wind speed (m/s)')
    parser.add_argument('-z', required=False, default=50.0, type=float, help='z height for epsilon')
    parser.add_argument('--Z-ref', required=False, default=20.0, type=float, help='Z_ref (m)')
    parser.add_argument('-z0', required=False, default=20.0, type=float, help='z_0')
    parser.add_argument('-zg', '--z-ground', required=False, default=0.0, type=float, help='z_ground')
    parser.add_argument('-k', '--kappa', required=False, default=0.41, type=float, help='kappa')
    parser.add_argument('-C', '--C-mu', required=False, default=0.09, type=float, help='C_mu')
    args = parser.parse_args()

    k, eps = abl_k_eps(args.U_ref, z=args.z, Z_ref=args.Z_ref, z_0=args.z0, z_ground=args.z_ground, kappa=args.kappa,
                       C_mu=args.C_mu)
    print '{0:0.6f} {1:0.6f}'.format(k, eps)