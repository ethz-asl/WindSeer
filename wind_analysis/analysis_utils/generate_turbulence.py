import numpy as np
import time
import control


def karman_E(k, L, sigma):
    E = 1.4528 * L * sigma**2 * (L*np.linalg.norm(k))**4 / (1+(L*np.linalg.norm(k))**2)**(17/6)
    return E


def spec_tens_iso_inc(k, L, sigma):
    phi_11 = k[1]**2 + k[2]**2
    phi_12 = -k[0] * k[1]
    phi_13 = -k[0] * k[2]
    phi_21 = -k[1] * k[0]
    phi_22 = k[0]**2 + k[2]**2
    phi_23 = -k[1] * k[2]
    phi_31 = -k[2] * k[0]
    phi_32 = -k[2] * k[1]
    phi_33 = k[0]**2 + k[1]**2

    E_k = karman_E(k, L, sigma)
    phi_mat = [[phi_11, phi_12, phi_13], [phi_21, phi_22, phi_23], [phi_31, phi_32, phi_33]]
    Phi = E_k / (4 * np.pi * np.linalg.norm(k)**4) * np.array(phi_mat)
    return Phi


def generate_turbulence_spectral(use_fft=True, check_statistics=False):
    '''
    Prototyping Turbulent Wind Fields based on Spectral Domain Simulation

    Author: David Rohr, ASL, ETH Zurich, Switzerland, 2019

    Resources: - The Spatial Structure of Neutral Atmospheric Surface-Layer
              Turbulence, J. Mann, J. Fluid Mech., vol. 273, pp. 141-168, 1994

               - Wind Field Simulation, J. Mann, Prop. Engng. Mech,. vol.13
              No.4 pp.269-282, 1998

              - Simulation of Three-Dimensional Turbulent Velocity Fields,
              R. Frehlich & L.Cornman, J. of applied Meteorology, vol.40, 2000
    '''
    lambda_min = 1100/63  # minimal wavelength of turbulence to simulate, [m]  (min 6cm)

    x_range = [0, 15]  # x-grid range [m] (north)
    y_range = [0, 15]  # y-grid range [m] (east)
    z_range = [0, 15]  # z-grid range [m] (down)

    dx = 0.5
    dy = 0.5
    dz = 0.5

    x = np.arange(x_range[0], x_range[1], dx).tolist()
    y = np.arange(y_range[0], y_range[1], dy).tolist()
    z = np.arange(z_range[0], z_range[1], dz).tolist()

    nx = len(x)
    ny = len(y)
    nz = len(z)

    X, Y, Z = np.meshgrid(x, y, z)
    U = np.array(0 * X).astype(complex)
    V = np.array(0 * Y).astype(complex)
    W = np.array(0 * Z).astype(complex)


    ### Spectral parameters
    nk = 91
    nk_x = nk
    nk_y = nk
    nk_z = nk

    # assemble spatial frequency components (wave vector)
    k_x = 2*np.pi/lambda_min/nk_x * np.array(np.arange(-(nk_x-1)/2, (nk_x+1)/2).tolist())
    k_y = 2*np.pi/lambda_min/nk_x * np.array(np.arange(-(nk_y-1)/2, (nk_y+1)/2).tolist())
    k_z = 2*np.pi/lambda_min/nk_x * np.array(np.arange(-(nk_z-1)/2, (nk_z+1)/2).tolist())

    # frequency spacing
    dk_x = k_x[1] - k_x[0]
    dk_y = k_y[1] - k_y[0]
    dk_z = k_z[1] - k_z[0]

    sigma = 1
    L = 725

    ## Fourier Simulation

    # Create random phase for every frequency component
    xi = (np.random.randn(3, nk_x, nk_y, nk_z) + 1j*np.random.randn(3, nk_x, nk_y, nk_z))/np.sqrt(2)

    C_ij = np.zeros((3, 3, nk_x, nk_y, nk_z))
    # Phi_ij = np.zeros((3, 3, nk_x, nk_y, nk_z))
    # E_ij = np.zeros((nk_x, nk_y, nk_z))
    # E_sum = 0

    for ikx in range(nk_x):
        for iky in range(nk_y):
            for ikz in range(nk_z):
                k = np.array([k_x[ikx], k_y[iky], k_z[ikz]])
                k = np.transpose(k)
                if 0 < np.linalg.norm(k) <= k_x[-1]:
                    # Phi_ij[:, :, ikx, iky, ikz] = spec_tens_iso_inc(k, L, sigma)
                    E = karman_E(k, L, sigma)
                    # E_ij[ikx, iky, ikz] = E
                    # E_sum = E_sum + E / (np.linalg.norm(k) ** 2 * 4 * np.pi) * dk_x * dk_y * dk_z

                    A_ij = np.sqrt(E / (4 * np.pi)) / (np.linalg.norm(k) ** 2) * np.array([[0, k[2], -k[1]],
                                                                                             [- k[2], 0, k[0]],
                                                                                             [k[1], -k[0], 0]])
                    C_ij[:, :, ikx, iky, ikz] = np.sqrt(dk_x * dk_y * dk_z) * np.array(A_ij)

    perc = 0
    N = nx * ny * nz

    if not use_fft:
        # Direct computation of turbulence at arbitrary position, expensive
        for ipx in range(nx):
            for ipy in range(ny):
                for ipz in range(nz):

                    print(perc/N)
                    perc = perc + 1

                    r = np.array([x[ipx], y[ipy], z[ipz]])
                    r = np.transpose(r)
                    v_r = np.zeros((3,), dtype=np.complex_)

                    for ikx in range(nk_x):
                        for iky in range(nk_y):
                            for ikz in range(nk_z):
                                k = np.array([k_x[ikx], k_y[iky], k_z[ikz]])
                                k = np.transpose(k)
                                v_r = v_r + np.exp(1j * np.array(np.transpose(k).dot(r))) \
                                      * np.array(C_ij[:, :, ikx, iky, ikz]).dot(np.array(xi[:, ikx, iky, ikz]))

                    U[ipx, ipy, ipz] = v_r[0]
                    V[ipx, ipy, ipz] = v_r[1]
                    W[ipx, ipy, ipz] = v_r[2]

    if use_fft:
        # IFFT
        complex_field = np.zeros((3, nk_x, nk_y, nk_z), dtype=complex)
        for ikx in range(nk_x):
            for iky in range(nk_y):
                for ikz in range(nk_z):
                    complex_field[:, ikx, iky, ikz] = np.array(C_ij[:, :, ikx, iky, ikz]).dot(np.array(xi[:, ikx, iky, ikz]))

        U_c = np.squeeze(complex_field[0, :, :, :])
        V_c = np.squeeze(complex_field[1, :, :, :])
        W_c = np.squeeze(complex_field[2, :, :, :])

        U_c = np.roll(np.roll(np.roll(U_c, int(-(nk_x-1)/2), axis=0), int(-(nk_y-1)/2), axis=1), int(-(nk_z-1)/2), axis=2)
        V_c = np.roll(np.roll(np.roll(V_c, int(-(nk_x-1)/2), axis=0), int(-(nk_y-1)/2), axis=1), int(-(nk_z-1)/2), axis=2)
        W_c = np.roll(np.roll(np.roll(W_c, int(-(nk_x-1)/2), axis=0), int(-(nk_y-1)/2), axis=1), int(-(nk_z-1)/2), axis=2)

        U2 = np.fft.ifft(np.fft.ifft(np.fft.ifft(U_c, n=None, axis=0), n=None, axis=1), n=None, axis=2) * nk_x*nk_y*nk_z
        V2 = np.fft.ifft(np.fft.ifft(np.fft.ifft(V_c, n=None, axis=0), n=None, axis=1), n=None, axis=2) * nk_x*nk_y*nk_z
        W2 = np.fft.ifft(np.fft.ifft(np.fft.ifft(W_c, n=None, axis=0), n=None, axis=1), n=None, axis=2) * nk_x*nk_y*nk_z

        x2 = np.linspace(0, (nk_x-1)*lambda_min, nk_x)
        y2 = np.linspace(0, (nk_y-1)*lambda_min, nk_y)
        z2 = np.linspace(0, (nk_z-1)*lambda_min, nk_z)

        X2, Y2, Z2 = np.meshgrid(x2, y2, z2)

        if check_statistics:
            prsv_pred = 0
            for ikx in range(nk_x):
                for iky in range(nk_y):
                    prsv_pred = prsv_pred + np.array(np.transpose(complex_field[:, ikx, iky, ikz])).dot(complex_field[:, ikx, iky, ikz])

            # check statistics of field
            # turbulent component standard deviation
            std_real = [np.std(np.reshape(np.real(U2), (1, -1))),
                        np.std(np.reshape(np.real(V2), (1, -1))),
                        np.std(np.reshape(np.real(W2), (1, -1)))]
            # turbulent kinetic energy (of real valued field)
            tke_real = 0.5 * np.sum(np.multiply(std_real, std_real))
            # turbulent kinetic energy (of complex valued field)
            tke_complex = 0.5 / (nk_x*nk_y*nk_z) * (np.sum(np.reshape(np.multiply(np.abs(U2), np.abs(U2)), (1, -1)))
                                                        + np.sum(np.reshape(np.multiply(np.abs(V2), np.abs(V2)), (1, -1)))
                                                        + np.sum(np.reshape(np.multiply(np.abs(W2), np.abs(W2)), (1, -1))))
            prsv = 1 / (nk_x*nk_y*nk_z) * (np.sum(np.reshape(np.multiply(np.abs(U2), np.abs(U2)), (1, -1)))
                                              + np.sum(np.reshape(np.multiply(np.abs(V2), np.abs(V2)), (1, -1)))
                                              + np.sum(np.reshape(np.multiply(np.abs(W2), np.abs(W2)), (1, -1))))
            print(prsv)

    if use_fft:
        U = U2
        V = V2
        W = W2
        X = X2
        Y = Y2
        Z = Z2
    # turbulent velocity field matrix
    UVW = np.stack((np.real(U), np.real(V), np.real(W)), axis=0)
    XYZ = np.stack((X, Y, Z), axis=0)

    return UVW, XYZ


def generate_turbulence_dryden(h=20, V=15, b=3, u_20=10, n=10, dt=0.5):
    """
    % FUNCTION:		dryden_TF
    %
    % PURPOSE:		Generate continuous turbulence estimate from Dryden PSD for
    %				low altitude (<1000 ft) flight.
    %
    % SYNTAX:		[UVW, PQR] = dryden_TF(h, V, b, u_20, n, dt)
    %
    % INPUTS:		h		- altitude (m)
    %				V		- airspeed (m/s)
    %				b		- wing span (m)
    %				u_20	- mean wind speed at 20ft (m/s)
    %							Light turbulence: 7.617 m/s
    %							Moderate turbulence: 15.234 m/s
    %							Severe turbulence: 30.468 m/s
    %				n		- number of time steps
    %				dt		- time step increment (s)
    %
    % OUTPUTS:		UVW		- matrix of turbulence magnitudes (m/s) [n×3]
    %				PQR		- matrix of rotational turbulence (rad/s) [n×3]
    %
    % AUTHOR:		Nicholas Lawrance
    %
    % CREATED:		July 2010
    %
    % MODIFIED:     July 2010
    """
    # Units to Imperial(ft, lb, s)
    u_20 = u_20 * 3.281
    h = h * 3.281
    b = b * 3.281
    V = V * 3.281

    # Low altitude model

    # Turbulence scale lengths
    L_u = h / (0.177 + 0.000823 * h) ** 1.2
    L_v = L_u
    L_w = h

    # Turbulence intensities
    sigma_w = 0.1 * u_20
    sigma_u = sigma_w / (0.177 + 0.000823 * h) ** 0.4
    sigma_v = sigma_u

    # Dryden turbulence TFs
    s = control.tf('s')
    dryden_uTF = sigma_u * np.sqrt(2 * L_u / (np.pi * V)) / (1 + s * L_u / V)
    dryden_pTF = sigma_w * np.sqrt(0.8 / V) * (np.pi / (4 * b)) ** (1 / 6) /\
                 (L_w ** (1 / 3) * (1 + s * 4 * b / np.pi / V))

    dryden_vTF = sigma_v * np.sqrt(L_v / np.pi / V) * (1 + s * np.sqrt(3) * L_v / V) / (1 + s * L_v / V) ** 2
    dryden_rTF = s / V / (1 + s * 3 * b / (np.pi * V)) * dryden_vTF

    dryden_wTF = sigma_w * np.sqrt(L_w / np.pi / V) * (1 + s * np.sqrt(3) * L_w / V) / (1 + s * L_w / V) ** 2
    dryden_qTF = s / V / (1 + s * 4 * b / (np.pi * V)) * dryden_wTF

    # Convert to state space
    [num, den] = control.tfdata(dryden_uTF)
    sys = control.tf2ss(num, den)
    Au = sys.A; Bu = sys.B; Cu = sys.C; Du = sys.D

    [num, den] = control.tfdata(dryden_vTF)
    sys = control.tf2ss(num, den)
    Av = sys.A; Bv = sys.B; Cv = sys.C; Dv = sys.D

    [num, den] = control.tfdata(dryden_wTF)
    sys = control.tf2ss(num, den)
    Aw = sys.A; Bw = sys.B; Cw = sys.C; Dw = sys.D

    [num, den] = control.tfdata(dryden_pTF)
    sys = control.tf2ss(num, den)
    Ap = sys.A; Bp = sys.B; Cp = sys.C; Dp = sys.D

    [num, den] = control.tfdata(dryden_qTF)
    sys = control.tf2ss(num, den)
    Aq = sys.A; Bq = sys.B; Cq = sys.C; Dq = sys.D

    [num, den] = control.tfdata(dryden_rTF)
    sys = control.tf2ss(num, den)
    Ar = sys.A; Br = sys.B; Cr = sys.C; Dr = sys.D

    # Generate time series
    rr = np.pi * np.random.randn(6, n)

    # These are the internal state representations, need to be stored but not
    # human useful, they are transfered through C and D to get actual gust values
    xxu = np.zeros((Au.shape[0], 1))
    xxv = np.zeros((Av.shape[0], 1))
    xxw = np.zeros((Aw.shape[0], 1))
    xxp = np.zeros((Ap.shape[0], 1))
    xxq = np.zeros((Aq.shape[0], 1))
    xxr = np.zeros((Ar.shape[0], 1))

    # Prealocate gusts
    UVW = np.zeros((n, 3))
    PQR = np.zeros((n, 3))

    # Numerical integration
    for ii in range(n):
        # Newton lazy integration
        xxu = xxu + (Au * xxu + Bu * rr[0, ii]) * dt
        UVW[ii, 0] = Cu * xxu

        xxv = xxv + (Av * xxv + Bv * rr[1, ii]) * dt
        UVW[ii, 1] = Cv * xxv

        xxw = xxw + (Aw * xxw + Bw * rr[2, ii]) * dt
        UVW[ii, 2] = Cw * xxw

        xxp = xxp + (Ap * xxp + Bp * rr[3, ii]) * dt
        PQR[ii, 0] = Cp * xxp

        xxq = xxq + (Aq * xxq + Bq * rr[4, ii]) * dt
        PQR[ii, 1] = Cq * xxq

        xxr = xxr + (Ar * xxr + Br * rr[5, ii]) * dt
        PQR[ii, 2] = Cr * xxr

    UVW = UVW / 3.281

    return UVW, PQR


# Test
# for i in range(100):
#     t_start = time.time()
#     uvw, xyz = generate_turbulence_spectral()
#     # uvw, pqr = generate_turbulence_dryden()
#     print('Time per function call: {0}'.format(time.time()-t_start))
