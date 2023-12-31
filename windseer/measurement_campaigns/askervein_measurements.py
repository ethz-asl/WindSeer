import numpy as np
import sys


def get_Askervein_measurements(name):
    '''
    Get the measurements form the Askervein experiment from the specified run.
    
    Parameters
    ----------
    name : str
        Name of the requested run

    Returns
    -------
    data : dict
        Dictionary with the measurements from the requested run
    '''
    available_runs = [
        'TU25', 'TU30A', 'TU30B', 'TU01A', 'TU01B', 'TU01C', 'TU01D', 'TU03A', 'TU03B',
        'TU05A', 'TU05B', 'TU05C', 'TU07B'
        ]
    if name in available_runs:
        return globals()[name]()
    else:
        print('Requested run ('**2 + name**2 + ') not valid. Available runs:')
        print(available_runs)
        sys.exit()


def add_sample_upw(data, mast, x, y, z, s, upw_deg, dir_deg, tke=np.nan):
    '''
    Add a measurement to the data dictionary
    
    Parameters
    ----------
    data : dict
        Data dictionary that is updated with the data from the mast
    mast : str
        Mast name/identifier
    x : float
        X-position of the measurements
    y : float
        Y-position of the measurements
    z : float
        Z-position of the measurements
    s : float
        Measured wind magnitude in m/s
    upw_deg : float
        Measured vertical wind direction in degrees
    dir_deg : float
        Measured horizontal wind direction in degrees
    tke : float, default : np.nan
        Measured TKE value in m^2/s^2
    '''
    upw = upw_deg * np.pi / 180.0
    dir = dir_deg * np.pi / 180.0
    sample = {
        'x': x,
        'y': y,
        'z': z,
        's': s,
        'u': -s * np.cos(upw) * np.sin(dir),
        'v': -s * np.cos(upw) * np.cos(dir),
        'w': s * np.sin(upw),
        'tke': tke
        }
    if mast in data.keys():
        data[mast].append(sample)
    else:
        data[mast] = [sample]


def add_sample_w(data, mast, x, y, z, s, w, dir_deg, tke=np.nan):
    '''
    Add a measurement to the data dictionary
    
    Parameters
    ----------
    data : dict
        Data dictionary that is updated with the data from the mast
    mast : str
        Mast name/identifier
    x : float
        X-position of the measurements
    y : float
        Y-position of the measurements
    z : float
        Z-position of the measurements
    s : float
        Measured wind magnitude in m/s
    w : float
        Measured vertical wind magnitude in m/s
    dir_deg : float
        Measured horizontal wind direction in degrees
    tke : float, default : np.nan
        Measured TKE value in m^2/s^2
    '''
    dir = dir_deg * np.pi / 180.0
    sample = {
        'x': x,
        'y': y,
        'z': z,
        's': s,
        'u': -s * np.sin(dir),
        'v': -s * np.cos(dir),
        'w': w,
        'tke': tke
        }
    if mast in data.keys():
        data[mast].append(sample)
    else:
        data[mast] = [sample]


def add_sample_s(data, mast, x, y, z, s):
    '''
    Add a measurement to the data dictionary
    
    Parameters
    ----------
    data : dict
        Data dictionary that is updated with the data from the mast
    mast : str
        Mast name/identifier
    x : float
        X-position of the measurements
    y : float
        Y-position of the measurements
    z : float
        Z-position of the measurements
    s : float
        Measured wind magnitude in m/s
    '''
    sample = {'x': x, 'y': y, 'z': z, 's': s, 'u': np.nan, 'v': np.nan, 'w': np.nan}
    if mast in data.keys():
        data[mast].append(sample)
    else:
        data[mast] = [sample]


def TU25():
    '''
    Get the measurements from the TU25 run.
    
    Returns
    -------
    data : dict
        Dictionary with the measurements from the TU25 run
    '''
    data = {}
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 5.51, 1.3, 208,
        0.5 * (0.931**2 + 0.609**2 + 0.495**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 5.5, 1.8, 210.3,
        0.5 * (0.992**2 + 0.552**2 + 0.255**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 5.60, -2.0, 210.1,
        0.5 * (0.845**2 + 0.587**2 + 0.437**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 20, 6.40, -1.7, 209.0,
        0.5 * (0.806**2 + 0.516**2 + 0.395**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 30, 7.00, -0.2, 212.8,
        0.5 * (0.810**2 + 0.524**2 + 0.428**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 40, 7.10, -1.0, 211.3,
        0.5 * (0.778**2 + 0.511**2 + 0.394**2)
        )
    add_sample_s(data, 'RS', -99, -1062, 4.9, 4.84)
    add_sample_w(data, 'RS', -99, -1062, 10.0, 5.57, np.nan, 210)
    add_sample_s(data, 'RS', -99, -1062, 16.9, 6.26)
    add_sample_s(data, 'RS', -99, -1062, 3, 4.43)
    add_sample_s(data, 'RS', -99, -1062, 5, 4.98)
    add_sample_s(data, 'RS', -99, -1062, 8, 5.39)
    add_sample_s(data, 'RS', -99, -1062, 15, 5.68)
    add_sample_s(data, 'RS', -99, -1062, 24, 6.18)
    add_sample_s(data, 'RS', -99, -1062, 49, 7.47)

    add_sample_upw(
        data, 'ASW85', 414, 1080, 10, 4.8, 3.1, 201.7,
        0.5 * (0.742**2 + 0.469**2 + 0.244**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 6, 4.133, 1.837, 205.339,
        0.5 * (0.842**2 + 0.524**2 + 0.225**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 10, 4.733, 0.301, 206.019,
        0.5 * (0.867**2 + 0.550**2 + 0.240**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 20, 5.088, 0.562, 210.267,
        0.5 * (0.839**2 + 0.529**2 + 0.289**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 31, 5.995, -0.553, 205.073,
        0.5 * (0.806**2 + 0.541**2 + 0.301**2)
        )
    add_sample_upw(
        data, 'ASW50', 651, 1336, 10, 4.0, 2.8, 198.7,
        0.5 * (0.883**2 + 0.522**2 + 0.317**2)
        )
    add_sample_upw(
        data, 'ASW35', 763, 1456, 10, 4.1, 10.5, 201.2,
        0.5 * (0.831**2 + 0.671**2 + 0.299**2)
        )
    add_sample_upw(
        data, 'ASW20', 858, 1562, 10, 6.8, 16.9, 205.7,
        0.5 * (0.711**2 + 0.662**2 + 0.367**2)
        )
    add_sample_upw(
        data, 'ASW10', 920, 1625, 10, 8.3, 16.0, 211.8,
        0.5 * (0.678**2 + 0.747**2 + 0.417**2)
        )

    add_sample_upw(
        data, 'HT', 982, 1703, 10, 9.9, 2.3, 206.2,
        0.5 * (0.728**2 + 0.627**2 + 0.333**2)
        )
    add_sample_s(data, 'HT', 982, 1703, 1, 8.20)
    add_sample_s(data, 'HT', 982, 1703, 3, 9.91)
    add_sample_s(data, 'HT', 982, 1703, 5, 10.29)
    add_sample_s(data, 'HT', 982, 1711, 8, 10.28)
    add_sample_s(data, 'HT', 982, 1711, 15, 10.12)

    add_sample_upw(
        data, 'ANE10', 1055, 1770, 10, 7.4, -10.5, 211.6,
        0.5 * (1.109**2 + 0.660**2 + 0.340**2)
        )
    add_sample_upw(
        data, 'ANE20', 1124, 1842, 10, 3.8, -18.4, 207.7,
        0.5 * (1.138**2 + 0.825**2 + 0.477**2)
        )

    add_sample_w(
        data, 'CP', 1289, 1451, 5, 8.94, 0.47, 220,
        0.5 * (0.859**2 + 0.612**2 + 0.232**2)
        )
    add_sample_w(
        data, 'CP', 1289, 1451, 16, 9.54, 0.42, 217,
        0.5 * (0.701**2 + 0.556**2 + 0.375**2)
        )
    add_sample_w(data, 'CP', 1287, 1415, 10, 8.79, np.nan, 223)

    add_sample_w(
        data, 'AASW10', 1224, 1346, 10, 8.44, 1.41, 217,
        0.5 * (0.734**2 + 0.687**2 + 0.385**2)
        )
    add_sample_w(data, 'AASW10', 1214, 1354, 10, 8.15, np.nan, 219)
    add_sample_w(data, 'AASW20', 1154, 1278, 10, 6.70, np.nan, 220)
    add_sample_w(
        data, 'AASW30', 1094, 1202, 10, 5.31, 1.38, 212,
        0.5 * (0.748**2 + 0.732**2 + 0.437**2)
        )
    add_sample_w(data, 'AASW30', 1084, 1211, 10, 5.49, np.nan, 215)
    add_sample_w(data, 'AASW40', 1018, 1132, 10, 4.43, np.nan, 214)
    add_sample_w(
        data, 'AASW50', 953, 1058, 10, 3.88, -0.23, 210,
        0.5 * (0.969**2 + 0.581**2 + 0.204**2)
        )

    return data


def TU30A():
    '''
    Get the measurements from the TU30A run.
    
    Returns
    -------
    data : dict
        Dictionary with the measurements from the TU30A run
    '''
    data = {}
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 8.06, 2.8, 133,
        0.5 * (1.421**2 + 0.938**2 + 0.746**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 47, 9.87, 0.3, 131,
        0.5 * (1.221**2 + 0.829**2 + 0.649**2)
        )
    add_sample_upw(data, 'RS', -99, -1062, 10, 7.5, 0.0, 137.8, np.nan)  # sigw missing
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 7.8, -0.2, 133.6,
        0.5 * (1.343**2 + 0.886**2 + 0.517**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 20, 8.8, 0.7, 133.6,
        0.5 * (1.267**2 + 0.835**2 + 0.491**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 30, 9.3, 1.5, 137.3,
        0.5 * (1.247**2 + 0.890**2 + 0.587**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 40, 9.9, 0.1, 133.8,
        0.5 * (1.220**2 + 0.862**2 + 0.521**2)
        )
    add_sample_s(data, 'RS', -99, -1062, 4.9, 6.83)
    add_sample_w(data, 'RS', -99, -1062, 10.0, 7.96, np.nan, 131)
    add_sample_s(data, 'RS', -99, -1062, 16.9, 8.72)
    add_sample_s(data, 'RS', -99, -1062, 3, 6.24)
    add_sample_s(data, 'RS', -99, -1062, 5, 6.96)
    add_sample_s(data, 'RS', -99, -1062, 8, 7.64)
    add_sample_s(data, 'RS', -99, -1062, 15, 8.42)
    add_sample_s(data, 'RS', -99, -1062, 24, 9.15)
    add_sample_s(data, 'RS', -99, -1062, 34, 9.58)
    add_sample_s(data, 'RS', -99, -1062, 49, 10.00)

    add_sample_upw(
        data, 'ASW85', 414, 1080, 10, 8.3, 2.2, 135.8,
        0.5 * (1.263**2 + 0.793**2 + 0.369**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 6, 8.680, 1.563, 137.692,
        0.5 * (1.390**2 + 0.896**2 + 0.358**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 10, 8.532, 0.238, 132.073,
        0.5 * (1.229**2 + 0.834**2 + 0.373**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 20, 10.230, 1.447, 136.139,
        0.5 * (1.291**2 + 0.944**2 + 0.440**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 31, 10.932, -1.166, 132.940,
        0.5 * (1.328**2 + 0.933**2 + 0.464**2)
        )
    add_sample_upw(
        data, 'ASW35', 763, 1456, 10, 8.3, 1.3, 130.8,
        0.5 * (1.383**2 + 0.954**2 + 0.282**2)
        )
    add_sample_upw(
        data, 'ASW20', 858, 1562, 10, 8.8, 7.8, 138.4,
        0.5 * (1.533**2 + 1.038**2 + 0.594**2)
        )
    add_sample_upw(
        data, 'ASW10', 920, 1625, 10, 9.8, 3.2, 140.9,
        0.5 * (1.400**2 + 0.934**2 + 0.369**2)
        )

    add_sample_upw(
        data, 'HT', 982, 1703, 10, 10.7, 1.5, 137.4,
        0.5 * (1.363**2 + 0.957**2 + 0.313**2)
        )
    add_sample_s(data, 'HT', 982, 1703, 1, 7.41)
    add_sample_s(data, 'HT', 982, 1703, 3, 9.95)
    add_sample_s(data, 'HT', 982, 1703, 5, 10.18)
    add_sample_s(data, 'HT', 982, 1711, 8, 10.80)
    add_sample_s(data, 'HT', 982, 1711, 15, 11.65)
    add_sample_s(data, 'HT', 982, 1711, 24, 12.25)
    add_sample_s(data, 'HT', 982, 1711, 34, 12.23)

    add_sample_upw(
        data, 'ANE10', 1055, 1770, 10, 9.2, -1.6, 134.7,
        0.5 * (1.363**2 + 0.973**2 + 0.369**2)
        )
    add_sample_upw(
        data, 'ANE20', 1124, 1842, 10, 9.3, 0.9, 138.7,
        0.5 * (1.347**2 + 0.903**2 + 0.397**2)
        )
    add_sample_upw(
        data, 'ANE40', 1262, 1975, 10, 9.3, 0.1, 136.7,
        0.5 * (1.330**2 + 0.861**2 + 0.374**2)
        )

    add_sample_w(data, 'CP', 1289, 1451, 1.6, 8.04, np.nan, np.nan, np.nan)
    add_sample_w(data, 'CP', 1289, 1451, 3.0, 9.23, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 5.0, 8.80, 0.71, np.nan,
        0.5 * (1.250**2 + 1.414**2 + 0.323**2)
        )
    add_sample_w(data, 'CP', 1289, 1451, 7.4, 10.95, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 10.0, 10.39, 0.83, 137,
        0.5 * (1.228**2 + 0.85**2 + 0.343**2)
        )
    add_sample_w(
        data, 'CP', 1289, 1451, 16.0, 11.18, 0.89, 137,
        0.5 * (1.174**2 + 0.85**2 + 0.383**2)
        )
    add_sample_w(data, 'CP', 1287, 1415, 10, 10.21, np.nan, 138)

    add_sample_w(
        data, 'AASW10', 1224, 1346, 10, 10.26, 0.61, 137,
        0.5 * (1.327**2 + 0.799**2 + 0.349**2)
        )
    add_sample_w(data, 'AASW10', 1214, 1354, 10, 10.29, np.nan, 133)
    add_sample_w(data, 'AASW20', 1154, 1278, 10, 9.97, np.nan, 141)
    add_sample_w(
        data, 'AASW30', 1094, 1202, 10, 9.40, 0.53, 136,
        0.5 * (1.235**2 + 0.843**2 + 0.422**2)
        )
    add_sample_w(data, 'AASW30', 1084, 1211, 10, 9.61, np.nan, 133)
    add_sample_w(data, 'AASW40', 1018, 1132, 10, 8.86, np.nan, 139)
    add_sample_w(
        data, 'AASW50', 953, 1058, 10, 8.95, 0.22, 134,
        0.5 * (1.333**2 + 0.748**2 + 0.367**2)
        )

    return data


def TU30B():
    '''
    Get the measurements from the TU30B run.
    
    Returns
    -------
    data : dict
        Dictionary with the measurements from the TU30B run
    '''
    data = {}
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 13.06, 2.5, 125,
        0.5 * (2.098**2 + 1.562**2 + 1.140**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 47, 16.61, 0.4, 125,
        0.5 * (1.641**2 + 1.312**2 + 1.000**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 12.10, 1.5, 129.4,
        0.5 * (1.880**2 + 1.195**2 + 0.530**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 12.7, 0.0, 127.0,
        0.5 * (1.950**2 + 1.330**2 + 0.778**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 20, 14.4, 2.0, 125.0,
        0.5 * (1.905**2 + 1.295**2 + 0.763**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 30, 15.3, 1.0, 129.3,
        0.5 * (1.795**2 + 1.220**2 + 0.788**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 40, 16.8, 0.7, 126.2,
        0.5 * (1.790**2 + 1.340**2 + 0.783**2)
        )
    add_sample_s(data, 'RS', -99, -1062, 4.9, 11.23)
    add_sample_w(data, 'RS', -99, -1062, 10.0, 12.90, np.nan, 124)
    add_sample_s(data, 'RS', -99, -1062, 16.9, 14.14)
    add_sample_s(data, 'RS', -99, -1062, 3, 9.94)
    add_sample_s(data, 'RS', -99, -1062, 5, 10.95)
    add_sample_s(data, 'RS', -99, -1062, 8, 12.00)
    add_sample_s(data, 'RS', -99, -1062, 15, 13.63)
    add_sample_s(data, 'RS', -99, -1062, 24, 14.93)
    add_sample_s(data, 'RS', -99, -1062, 34, 15.77)
    add_sample_s(data, 'RS', -99, -1062, 49, 16.72)

    add_sample_upw(
        data, 'ASW85', 414, 1080, 10, 11.9, 2.0, 122.7,
        0.5 * (1.805**2 + 1.420**2 + 0.511**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 6, 12.047, 2.117, 124.893,
        0.5 * (1.864**2 + 1.342**2 + 0.535**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 10, 11.741, 1.213, 117.347,
        0.5 * (1.768**2 + 1.079**2 + 0.590**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 20, 14.171, 2.805, 122.941,
        0.5 * (1.658**2 + 1.338**2 + 0.687**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 31, 15.424, -1.811, 120.682,
        0.5 * (1.777**2 + 1.390**2 + 0.766**2)
        )
    add_sample_upw(
        data, 'ASW35', 763, 1456, 10, 11.7, -0.4, 115.8,
        0.5 * (1.950**2 + 1.205**2 + 0.371**2)
        )
    add_sample_upw(
        data, 'ASW20', 858, 1562, 10, 12.6, 3.5, 125.1,
        0.5 * (1.850**2 + 1.380**2 + 0.780**2)
        )
    add_sample_upw(
        data, 'ASW10', 920, 1625, 10, 14.2, -0.4, 122.6,
        0.5 * (1.910**2 + 1.410**2 + 0.589**2)
        )

    add_sample_upw(
        data, 'HT', 982, 1703, 10, 14.8, 1.6, 117.4,
        0.5 * (2.155**2 + 1.520**2 + 0.501**2)
        )
    add_sample_s(data, 'HT', 982, 1703, 1, 10.94)
    add_sample_s(data, 'HT', 982, 1703, 3, 13.36)
    add_sample_s(data, 'HT', 982, 1703, 5, 14.34)
    add_sample_s(data, 'HT', 982, 1711, 8, 15.11)
    add_sample_s(data, 'HT', 982, 1711, 15, 16.61)
    add_sample_s(data, 'HT', 982, 1711, 24, 17.65)
    add_sample_s(data, 'HT', 982, 1711, 34, 17.62)
    add_sample_s(data, 'HT', 982, 1711, 49, 17.74)

    add_sample_upw(
        data, 'ANE10', 1055, 1770, 10, 13.4, 1.1, 116.7,
        0.5 * (1.930**2 + 1.225**2 + 0.520**2)
        )
    add_sample_upw(
        data, 'ANE20', 1124, 1842, 10, 13.1, 5.5, 125.5,
        0.5 * (1.845**2 + 1.055**2 + 0.758**2)
        )
    add_sample_upw(
        data, 'ANE40', 1262, 1975, 10, 12.7, 1.8, 123.3,
        0.5 * (2.010**2 + 1.265**2 + 0.550**2)
        )

    add_sample_w(data, 'CP', 1289, 1451, 1.6, 12.08, np.nan, np.nan, np.nan)
    add_sample_w(data, 'CP', 1289, 1451, 3.0, 13.75, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 5.0, 13.41, 1.12, np.nan,
        0.5 * (1.964**2 + 1.237**2 + 0.416**2)
        )
    add_sample_w(data, 'CP', 1289, 1451, 7.4, 16.80, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 10.0, 15.10, 1.18, 119,
        0.5 * (1.924**2 + 1.292**2 + 0.494**2)
        )
    add_sample_w(
        data, 'CP', 1289, 1451, 16.0, 16.16, 1.41, 120,
        0.5 * (1.868**2 + 1.318**2 + 0.563**2)
        )
    add_sample_w(data, 'CP', 1287, 1415, 10, 15.25, np.nan, 118)

    add_sample_w(
        data, 'AASW10', 1224, 1346, 10, 15.02, 0.36, 121,
        0.5 * (1.999**2 + 1.365**2 + 0.481**2)
        )
    add_sample_w(data, 'AASW10', 1214, 1354, 10, 15.04, np.nan, 115)
    add_sample_w(data, 'AASW20', 1154, 1278, 10, 14.72, np.nan, 128)
    add_sample_w(
        data, 'AASW30', 1094, 1202, 10, 14.11, 0.09, 121,
        0.5 * (1.901**2 + 1.276**2 + 0.894**2)
        )
    add_sample_w(data, 'AASW30', 1084, 1211, 10, 13.87, np.nan, 123)
    add_sample_w(data, 'AASW40', 1018, 1132, 10, 12.51, np.nan, 124)
    add_sample_w(
        data, 'AASW50', 953, 1058, 10, 12.55, 0.12, 123,
        0.5 * (1.908**2 + 1.143**2 + 0.494**2)
        )

    return data


def TU01A():
    '''
    Get the measurements from the TU01A run.
    
    Returns
    -------
    data : dict
        Dictionary with the measurements from the TU01A run
    '''
    data = {}
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 9.44, 2.7, 174,
        0.5 * (1.562**2 + 1.191**2 + 0.909**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 47, 11.71, 0.2, 177,
        0.5 * (1.408**2 + 1.060**2 + 0.909**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 9.40, 2.6, 171.5,
        0.5 * (1.485**2 + 0.845**2 + 0.475**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 9.5, -0.9, 170.6,
        0.5 * (1.508**2 + 1.077**2 + 0.704**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 20, 10.6, -0.5, 170.2,
        0.5 * (1.442**2 + 0.934**2 + 0.674**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 30, 11.0, 1.5, 172.6,
        0.5 * (1.352**2 + 1.016**2 + 0.825**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 40, 11.7, -0.1, 170.3,
        0.5 * (1.413**2 + 0.925**2 + 0.735**2)
        )
    add_sample_s(data, 'RS', -99, -1062, 4.9, 8.38)
    add_sample_w(data, 'RS', -99, -1062, 10.0, 9.60, np.nan, 170)
    add_sample_s(data, 'RS', -99, -1062, 16.9, 10.45)
    add_sample_s(data, 'RS', -99, -1062, 3, 7.69)
    add_sample_s(data, 'RS', -99, -1062, 5, 8.54)
    add_sample_s(data, 'RS', -99, -1062, 8, 9.23)
    add_sample_s(data, 'RS', -99, -1062, 15, 10.07)
    add_sample_s(data, 'RS', -99, -1062, 24, 10.94)
    add_sample_s(data, 'RS', -99, -1062, 34, 11.41)
    add_sample_s(data, 'RS', -99, -1062, 49, 12.19)

    add_sample_upw(
        data, 'ASW85', 414, 1080, 10, 10.0, 3.1, 165.7,
        0.5 * (1.545**2 + 0.981**2 + 0.490**2)
        )
    add_sample_upw(
        data, 'ASW50', 651, 1336, 10, 9.9, 1.2, 157.2,
        0.5 * (1.503**2 + 1.006**2 + 0.461**2)
        )
    add_sample_upw(
        data, 'ASW35', 763, 1456, 10, 10.0, 6.7, 157.4,
        0.5 * (1.478**2 + 1.100**2 + 0.548**2)
        )
    add_sample_upw(
        data, 'ASW20', 858, 1562, 10, 12.6, 14.0, 166.7,
        0.5 * (1.405**2 + 1.197**2 + 0.579**2)
        )
    add_sample_upw(
        data, 'ASW10', 920, 1625, 10, 14.4, 10.3, 175.8,
        0.5 * (1.300**2 + 1.202**2 + 0.437**2)
        )

    add_sample_upw(
        data, 'ASW60', 571, 1253, 6, 9.872, 1.848, 163.162,
        0.5 * (1.589**2 + 1.029**2 + 0.446**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 10, 9.395, 1.025, 160.523,
        0.5 * (1.453**2 + 0.983**2 + 0.526**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 20, 11.694, 1.974, 162.846,
        0.5 * (1.628**2 + 1.035**2 + 0.626**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 31, 12.088, -3.983, 161.181,
        0.5 * (1.592**2 + 0.998**2 + 0.831**2)
        )

    add_sample_upw(
        data, 'HT', 982, 1703, 10, 17.6, 1.8, 176.2,
        0.5 * (1.340**2 + 1.100**2 + 0.366**2)
        )
    add_sample_s(data, 'HT', 982, 1703, 1, 12.48)
    add_sample_s(data, 'HT', 982, 1703, 3, 15.18)
    add_sample_s(data, 'HT', 982, 1703, 5, 16.33)
    add_sample_s(data, 'HT', 982, 1711, 8, 16.73)
    add_sample_s(data, 'HT', 982, 1711, 15, 17.19)
    add_sample_s(data, 'HT', 982, 1711, 24, 17.07)
    add_sample_s(data, 'HT', 982, 1711, 49, 16.88)

    add_sample_upw(
        data, 'ANE10', 1055, 1770, 10, 14.1, -7.7, 176.7,
        0.5 * (1.658**2 + 0.983**2 + 0.435**2)
        )
    add_sample_upw(
        data, 'ANE20', 1124, 1842, 10, 8.1, -4.8, 166.0,
        0.5 * (2.263**2 + 1.363**2 + 0.765**2)
        )
    add_sample_upw(
        data, 'ANE40', 1262, 1975, 10, 9.0, -3.0, 156.8,
        0.5 * (1.750**2 + 1.202**2 + 0.687**2)
        )

    add_sample_w(data, 'CP', 1289, 1451, 1.6, 12.32, np.nan, np.nan, np.nan)
    add_sample_w(data, 'CP', 1289, 1451, 3.0, 13.80, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 5.0, 14.12, 1.05, 186,
        0.5 * (1.426**2 + 1.023**2 + 0.323**2)
        )
    add_sample_w(data, 'CP', 1289, 1451, 7.4, 16.11, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 10.0, 14.96, 1.54, 180,
        0.5 * (1.280**2 + 1.036**2 + 0.439**2)
        )
    add_sample_w(
        data, 'CP', 1289, 1451, 16.0, 15.60, 1.13, 179,
        0.5 * (1.217**2 + 1.075**2 + 0.561**2)
        )
    add_sample_w(data, 'CP', 1287, 1415, 10, 14.36, np.nan, 187)

    add_sample_w(data, 'AASW10', 1214, 1354, 10, 13.76, np.nan, 178)
    add_sample_w(
        data, 'AASW10', 1224, 1346, 10, 14.05, 2.02, 177,
        0.5 * (1.341**2 + 1.158**2 + 0.495**2)
        )
    add_sample_w(data, 'AASW20', 1154, 1278, 10, 12.40, np.nan, 177)
    add_sample_w(data, 'AASW30', 1084, 1211, 10, 11.37, np.nan, 169)
    add_sample_w(
        data, 'AASW30', 1094, 1202, 10, 11.50, 2.11, 166,
        0.5 * (1.382**2 + 0.965**2 + 0.574**2)
        )
    add_sample_w(data, 'AASW40', 1018, 1132, 10, 9.69, np.nan, 167)
    add_sample_w(
        data, 'AASW50', 953, 1058, 10, 9.22, 0.57, 160,
        0.5 * (1.505**2 + 0.958**2 + 0.551**2)
        )

    return data


def TU01B():
    '''
    Get the measurements from the TU01B run.
    
    Returns
    -------
    data : dict
        Dictionary with the measurements from the TU01B run
    '''
    data = {}
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 9.25, 2.8, 180,
        0.5 * (1.511**2 + 1.003**2 + 0.843**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 47, 11.42, 0.0, 183,
        0.5 * (1.272**2 + 0.891**2 + 0.823**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 9.00, 2.7, 177.6,
        0.5 * (1.408**2 + 0.786**2 + 0.473**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 9.0, -1.5, 177.2,
        0.5 * (1.358**2 + 0.863**2 + 0.648**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 20, 10.1, -0.9, 176.7,
        0.5 * (1.313**2 + 0.771**2 + 0.631**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 30, 10.5, 1.0, 180.0,
        0.5 * (1.223**2 + 0.890**2 + 0.760**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 40, 11.1, -0.5, 177.4,
        0.5 * (1.243**2 + 0.761**2 + 0.669**2)
        )

    add_sample_s(data, 'RS', -99, -1062, 4.9, 7.94)
    add_sample_w(data, 'RS', -99, -1062, 10.0, 9.12, np.nan, 177)
    add_sample_s(data, 'RS', -99, -1062, 16.9, 9.99)

    add_sample_s(data, 'RS', -99, -1062, 3, 7.43)
    add_sample_s(data, 'RS', -99, -1062, 5, 8.25)
    add_sample_s(data, 'RS', -99, -1062, 8, 8.86)
    add_sample_s(data, 'RS', -99, -1062, 15, 9.62)
    add_sample_s(data, 'RS', -99, -1062, 24, 10.43)
    add_sample_s(data, 'RS', -99, -1062, 34, 10.90)
    add_sample_s(data, 'RS', -99, -1062, 49, 11.73)

    add_sample_upw(
        data, 'ASW85', 414, 1080, 10, 9.5, 3.5, 171.7,
        0.5 * (1.298**2 + 0.845**2 + 0.455**2)
        )
    add_sample_upw(
        data, 'ASW50', 651, 1336, 10, 8.7, 1.3, 161.7,
        0.5 * (1.367**2 + 1.056**2 + 0.443**2)
        )
    add_sample_upw(
        data, 'ASW35', 763, 1456, 10, 8.8, 7.4, 162.3,
        0.5 * (1.370**2 + 1.140**2 + 0.494**2)
        )
    add_sample_upw(
        data, 'ASW20', 858, 1562, 10, 11.4, 14.8, 172.4,
        0.5 * (1.233**2 + 1.064**2 + 0.570**2)
        )
    add_sample_upw(
        data, 'ASW10', 920, 1625, 10, 13.4, 11.9, 182.0,
        0.5 * (1.240**2 + 1.108**2 + 0.431**2)
        )

    add_sample_upw(
        data, 'ASW60', 571, 1253, 6, 8.710, 1.584, 168.226,
        0.5 * (1.443**2 + 0.979**2 + 0.410**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 10, 8.099, 0.747, 165.979,
        0.5 * (1.287**2 + 0.948**2 + 0.486**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 20, 10.289, 1.663, 168.643,
        0.5 * (1.471**2 + 0.996**2 + 0.579**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 31, 10.598, -3.011, 167.189,
        0.5 * (1.385**2 + 0.984**2 + 0.715**2)
        )

    add_sample_upw(
        data, 'HT', 982, 1703, 10, 16.3, 1.8, 182.1,
        0.5 * (1.260**2 + 1.070**2 + 0.385**2)
        )
    add_sample_s(data, 'HT', 982, 1703, 1, 11.57)
    add_sample_s(data, 'HT', 982, 1703, 3, 14.13)
    add_sample_s(data, 'HT', 982, 1703, 5, 15.20)
    add_sample_s(data, 'HT', 982, 1711, 8, 15.51)
    add_sample_s(data, 'HT', 982, 1711, 15, 15.79)
    add_sample_s(data, 'HT', 982, 1711, 24, 15.62)
    add_sample_s(data, 'HT', 982, 1711, 49, 15.45)

    add_sample_upw(
        data, 'ANE10', 1055, 1770, 10, 12.5, -8.3, 182.9,
        0.5 * (1.587**2 + 0.898**2 + 0.401**2)
        )
    add_sample_upw(
        data, 'ANE20', 1124, 1842, 10, 7.1, -6.8, 170.8,
        0.5 * (2.255**2 + 1.370**2 + 0.754**2)
        )
    add_sample_upw(
        data, 'ANE40', 1262, 1975, 10, 7.4, -2.4, 159.0,
        0.5 * (1.565**2 + 1.188**2 + 0.673**2)
        )

    add_sample_w(data, 'CP', 1289, 1451, 1.6, 11.32, np.nan, np.nan, np.nan)
    add_sample_w(data, 'CP', 1289, 1451, 3.0, 12.74, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 5.0, 13.11, 0.97, 192,
        0.5 * (1.290**2 + 0.964**2 + 0.321**2)
        )
    add_sample_w(data, 'CP', 1289, 1451, 7.4, 14.57, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 10.0, 13.81, 1.41, 186,
        0.5 * (1.150**2 + 0.985**2 + 0.433**2)
        )
    add_sample_w(
        data, 'CP', 1289, 1451, 16.0, 14.37, 0.98, 185,
        0.5 * (1.090**2 + 0.954**2 + 0.546**2)
        )
    add_sample_w(data, 'CP', 1287, 1415, 10, 13.24, np.nan, 192)

    add_sample_w(data, 'AASW10', 1214, 1354, 10, 12.43, np.nan, 184)
    add_sample_w(
        data, 'AASW10', 1224, 1346, 10, 12.73, 1.95, 183,
        0.5 * (1.182**2 + 1.041**2 + 0.473**2)
        )
    add_sample_w(data, 'AASW20', 1154, 1278, 10, 10.97, np.nan, 181)
    add_sample_w(data, 'AASW30', 1084, 1211, 10, 9.79, np.nan, 173)
    add_sample_w(
        data, 'AASW30', 1094, 1202, 10, 9.86, 1.95, 171,
        0.5 * (1.200**2 + 1.000**2 + 0.555**2)
        )
    add_sample_w(data, 'AASW40', 1018, 1132, 10, 8.49, np.nan, 175)
    add_sample_w(
        data, 'AASW50', 953, 1058, 10, 7.96, 0.50, 165,
        0.5 * (1.410**2 + 0.907**2 + 0.486**2)
        )

    return data


def TU01C():
    '''
    Get the measurements from the TU01C run.
    
    Returns
    -------
    data : dict
        Dictionary with the measurements from the TU01C run
    '''
    data = {}
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 7.81, 2.7, 185,
        0.5 * (1.322**2 + 0.892**2 + 0.727**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 47, 10.34, 0.2, 189,
        0.5 * (1.191**2 + 0.837**2 + 0.697**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 7.4, 2.7, 181.9,
        0.5 * (1.200**2 + 0.658**2 + 0.396**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 7.4, -1.3, 180.8,
        0.5 * (1.150**2 + 0.727**2 + 0.552**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 20, 8.6, -1.1, 181.0,
        0.5 * (1.157**2 + 0.675**2 + 0.530**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 30, 9.2, 1.1, 184.7,
        0.5 * (1.137**2 + 0.822**2 + 0.632**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 40, 9.7, -0.4, 182.4,
        0.5 * (1.157**2 + 0.721**2 + 0.559**2)
        )

    add_sample_s(data, 'RS', -99, -1062, 4.9, 6.47)
    add_sample_w(data, 'RS', -99, -1062, 10.0, 7.55, np.nan, 181)
    add_sample_s(data, 'RS', -99, -1062, 16.9, 8.38)

    add_sample_s(data, 'RS', -99, -1062, 3, 6.01)
    add_sample_s(data, 'RS', -99, -1062, 5, 6.68)
    add_sample_s(data, 'RS', -99, -1062, 8, 7.20)
    add_sample_s(data, 'RS', -99, -1062, 15, 8.00)
    add_sample_s(data, 'RS', -99, -1062, 24, 9.25)
    add_sample_s(data, 'RS', -99, -1062, 34, 9.46)
    add_sample_s(data, 'RS', -99, -1062, 49, 10.45)

    add_sample_upw(
        data, 'ASW85', 414, 1080, 10, 7.4, 3.9, 173.4,
        0.5 * (1.213**2 + 0.608**2 + 0.374**2)
        )
    add_sample_upw(
        data, 'ASW50', 651, 1336, 10, 6.8, 1.0, 160.7,
        0.5 * (1.142**2 + 0.797**2 + 0.331**2)
        )
    add_sample_upw(
        data, 'ASW35', 763, 1456, 10, 7.2, 6.9, 161.7,
        0.5 * (1.167**2 + 0.842**2 + 0.403**2)
        )
    add_sample_upw(
        data, 'ASW20', 858, 1562, 10, 9.9, 14.2, 174.8,
        0.5 * (1.123**2 + 0.772**2 + 0.468**2)
        )
    add_sample_upw(
        data, 'ASW10', 920, 1625, 10, 12.0, 12.5, 185.8,
        0.5 * (1.113**2 + 0.945**2 + 0.415**2)
        )

    add_sample_upw(
        data, 'ASW60', 571, 1253, 6, 6.684, 1.184, 167.736,
        0.5 * (1.094**2 + 0.708**2 + 0.285**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 10, 6.352, 0.467, 166.792,
        0.5 * (0.961**2 + 0.706**2 + 0.370**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 20, 8.143, 1.204, 170.050,
        0.5 * (1.144**2 + 0.759**2 + 0.415**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 31, 8.646, -1.578, 169.657,
        0.5 * (1.146**2 + 0.746**2 + 0.464**2)
        )

    add_sample_upw(
        data, 'HT', 982, 1703, 10, 15.0, 1.3, 186.9,
        0.5 * (1.140**2 + 0.723**2 + 0.209**2)
        )
    add_sample_s(data, 'HT', 982, 1703, 1, 10.71)
    add_sample_s(data, 'HT', 982, 1703, 3, 13.19)
    add_sample_s(data, 'HT', 982, 1703, 5, 14.12)
    add_sample_s(data, 'HT', 982, 1711, 8, 14.25)
    add_sample_s(data, 'HT', 982, 1711, 15, 14.43)
    add_sample_s(data, 'HT', 982, 1711, 24, 14.59)
    add_sample_s(data, 'HT', 982, 1711, 49, 14.82)

    add_sample_upw(
        data, 'ANE10', 1055, 1770, 10, 11.2, -8.5, 186.9,
        0.5 * (1.497**2 + 0.682**2 + 0.362**2)
        )
    add_sample_upw(
        data, 'ANE20', 1124, 1842, 10, 5.4, -9.0, 167.6,
        0.5 * (1.993**2 + 1.407**2 + 0.759**2)
        )
    add_sample_upw(
        data, 'ANE40', 1262, 1975, 10, 6.0, -2.1, 154.9,
        0.5 * (1.280**2 + 1.160**2 + 0.690**2)
        )

    add_sample_w(data, 'CP', 1289, 1451, 1.6, 10.62, np.nan, np.nan, np.nan)
    add_sample_w(data, 'CP', 1289, 1451, 3.0, 11.96, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 5.0, 12.14, 0.86, 197,
        0.5 * (1.182**2 + 0.741**2 + 0.280**2)
        )
    add_sample_w(data, 'CP', 1289, 1451, 7.4, 13.80, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 10.0, 12.80, 1.26, 192,
        0.5 * (1.071**2 + 0.759**2 + 0.380**2)
        )
    add_sample_w(
        data, 'CP', 1289, 1451, 16.0, 13.40, 0.85, 191,
        0.5 * (1.049**2 + 0.701**2 + 0.470**2)
        )
    add_sample_w(data, 'CP', 1287, 1415, 10, 12.24, np.nan, 191)

    add_sample_w(data, 'AASW10', 1214, 1354, 10, 11.32, np.nan, 191)
    add_sample_w(
        data, 'AASW10', 1224, 1346, 10, 11.63, 1.82, 189,
        0.5 * (1.065**2 + 0.796**2 + 0.442**2)
        )
    add_sample_w(data, 'AASW20', 1154, 1278, 10, 9.62, np.nan, 186)
    add_sample_w(data, 'AASW30', 1084, 1211, 10, 8.23, np.nan, 177)
    add_sample_w(
        data, 'AASW30', 1094, 1202, 10, 8.30, 1.67, 174,
        0.5 * (1.042**2 + 0.799**2 + 0.468**2)
        )
    add_sample_w(data, 'AASW40', 1018, 1132, 10, 7.02, np.nan, 175)
    add_sample_w(
        data, 'AASW50', 953, 1058, 10, 6.34, 0.39, 166,
        0.5 * (1.110**2 + 0.722**2 + 0.433**2)
        )

    return data


def TU01D():
    '''
    Get the measurements from the TU01D run.
    
    Returns
    -------
    data : dict
        Dictionary with the measurements from the TU01D run
    '''
    data = {}
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 7.70, 2.5, 203,
        0.5 * (1.144**2 + 0.922**2 + 0.737**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 47, 10.57, 0.9, 206,
        0.5 * (1.103**2 + 0.859**2 + 0.665**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 7.7, 2.1, 198.7,
        0.5 * (1.140**2 + 0.688**2 + 0.342**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 7.6, -1.7, 198.6,
        0.5 * (1.200**2 + 0.868**2 + 0.598**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 20, 8.8, -1.3, 198.8,
        0.5 * (1.200**2 + 0.812**2 + 0.574**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 30, 9.6, 0.4, 202.6,
        0.5 * (1.200**2 + 0.852**2 + 0.632**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 40, 10.2, -0.8, 200.5,
        0.5 * (1.100**2 + 0.825**2 + 0.577**2)
        )

    add_sample_s(data, 'RS', -99, -1062, 4.9, 6.46)
    add_sample_w(data, 'RS', -99, -1062, 10.0, 7.56, np.nan, 197)
    add_sample_s(data, 'RS', -99, -1062, 16.9, 8.44)

    add_sample_s(data, 'RS', -99, -1062, 3, 6.33)
    add_sample_s(data, 'RS', -99, -1062, 5, 6.97)
    add_sample_s(data, 'RS', -99, -1062, 8, 7.54)
    add_sample_s(data, 'RS', -99, -1062, 15, 8.11)
    add_sample_s(data, 'RS', -99, -1062, 24, 9.52)
    add_sample_s(data, 'RS', -99, -1062, 34, 9.85)
    add_sample_s(data, 'RS', -99, -1062, 49, 11.04)

    add_sample_upw(
        data, 'ASW85', 414, 1080, 10, 7.4, 3.3, 192.1,
        0.5 * (1.200**2 + 0.731**2 + 0.388**2)
        )
    add_sample_upw(
        data, 'ASW50', 651, 1336, 10, 5.5, 2.4, 182.4,
        0.5 * (1.170**2 + 0.837**2 + 0.431**2)
        )
    add_sample_upw(
        data, 'ASW35', 763, 1456, 10, 6.1, 8.9, 185.0,
        0.5 * (1.210**2 + 0.839**2 + 0.419**2)
        )
    add_sample_upw(
        data, 'ASW20', 858, 1562, 10, 9.2, 16.1, 193.2,
        0.5 * (1.140**2 + 0.990**2 + 0.516**2)
        )
    add_sample_upw(
        data, 'ASW10', 920, 1625, 10, 11.6, 14.5, 201.7,
        0.5 * (1.100**2 + 1.090**2 + 0.492**2)
        )

    add_sample_upw(
        data, 'ASW60', 571, 1253, 6, 6.267, 1.127, 187.951,
        0.5 * (1.289**2 + 0.731**2 + 0.323**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 10, 6.047, 0.768, 190.632,
        0.5 * (1.109**2 + 0.817**2 + 0.372**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 20, 8.199, 1.710, 190.620,
        0.5 * (1.314**2 + 0.859**2 + 0.443**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 31, 8.772, -1.571, 190.310,
        0.5 * (1.265**2 + 0.890**2 + 0.506**2)
        )

    add_sample_upw(
        data, 'HT', 982, 1703, 10, 14.4, 2.6, 197.5,
        0.5 * (1.340**2 + 0.994**2 + 0.462**2)
        )
    add_sample_s(data, 'HT', 982, 1703, 1, 11.12)
    add_sample_s(data, 'HT', 982, 1703, 3, 13.57)
    add_sample_s(data, 'HT', 982, 1703, 5, 14.34)
    add_sample_s(data, 'HT', 982, 1711, 8, 14.31)
    add_sample_s(data, 'HT', 982, 1711, 15, 14.30)
    add_sample_s(data, 'HT', 982, 1711, 24, 14.46)
    add_sample_s(data, 'HT', 982, 1711, 34, 14.37)
    add_sample_s(data, 'HT', 982, 1711, 49, 14.67)

    add_sample_upw(
        data, 'ANE10', 1055, 1770, 10, 10.0, -9.6, 199.7,
        0.5 * (1.880**2 + 0.973**2 + 0.499**2)
        )
    add_sample_upw(
        data, 'ANE20', 1124, 1842, 10, 3.5, -13.3, 178.8,
        0.5 * (2.090**2 + 1.590**2 + 0.866**2)
        )
    add_sample_upw(
        data, 'ANE40', 1262, 1975, 10, 3.7, -3.4, 156.6,
        0.5 * (1.620**2 + 1.460**2 + 1.040**2)
        )

    add_sample_w(data, 'CP', 1289, 1451, 1.6, 10.85, np.nan, np.nan, np.nan)
    add_sample_w(data, 'CP', 1289, 1451, 3.0, 11.89, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 5.0, 12.02, 0.82, 211,
        0.5 * (1.602**2 + 0.890**2 + 0.295**2)
        )
    add_sample_w(data, 'CP', 1289, 1451, 7.4, 13.63, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 10.0, 12.40, 1.16, 207,
        0.5 * (1.559**2 + 0.895**2 + 0.458**2)
        )
    add_sample_w(
        data, 'CP', 1289, 1451, 16.0, 12.90, 0.72, 205,
        0.5 * (1.602**2 + 0.853**2 + 0.575**2)
        )
    add_sample_w(data, 'CP', 1287, 1415, 10, 11.88, np.nan, 213)

    add_sample_w(data, 'AASW10', 1214, 1354, 10, 10.85, np.nan, 208)
    add_sample_w(
        data, 'AASW10', 1224, 1346, 10, 11.15, 1.83, 206,
        0.5 * (1.462**2 + 0.913**2 + 0.501**2)
        )
    add_sample_w(data, 'AASW20', 1154, 1278, 10, 8.77, np.nan, 207)
    add_sample_w(data, 'AASW30', 1084, 1211, 10, 7.14, np.nan, 197)
    add_sample_w(
        data, 'AASW30', 1094, 1202, 10, 6.90, 1.66, 194,
        0.5 * (1.178**2 + 0.903**2 + 0.553**2)
        )
    add_sample_w(data, 'AASW40', 1018, 1132, 10, 5.58, np.nan, 195)
    add_sample_w(
        data, 'AASW50', 953, 1058, 10, 4.75, 0.35, 187,
        0.5 * (1.259**2 + 0.820**2 + 0.446**2)
        )

    return data


def TU03A():
    '''
    Get the measurements from the TU03A run.
    
    Returns
    -------
    data : dict
        Dictionary with the measurements from the TU03A run
    '''
    data = {}
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 9.90, 2.0, 210,
        0.5 * (1.500**2 + 1.037**2 + 0.907**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 47, 13.08, 0.1, 208,
        0.5 * (1.254**2 + 0.878**2 + 0.779**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 9.3, 2.7, 205.9,
        0.5 * (1.325**2 + 0.665**2 + 0.432**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 9.8, -2.0, 207.8,
        0.5 * (1.455**2 + 0.957**2 + 0.722**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 20, 11.3, -2.0, 206.4,
        0.5 * (1.095**2 + 0.732**2 + 0.574**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 30, 12.0, -0.6, 210.6,
        0.5 * (1.260**2 + 0.873**2 + 0.717**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 40, 12.5, -1.2, 208.7,
        0.5 * (1.230**2 + 0.854**2 + 0.654**2)
        )

    add_sample_s(data, 'RS', -99, -1062, 4.9, 8.73)
    add_sample_w(data, 'RS', -99, -1062, 10.0, 10.01, np.nan, 206)
    add_sample_s(data, 'RS', -99, -1062, 16.9, 10.95)

    add_sample_s(data, 'RS', -99, -1062, 3, 7.78)
    add_sample_s(data, 'RS', -99, -1062, 5, 8.65)
    add_sample_s(data, 'RS', -99, -1062, 8, 9.28)
    add_sample_s(data, 'RS', -99, -1062, 15, 10.26)
    add_sample_s(data, 'RS', -99, -1062, 24, 11.27)
    add_sample_s(data, 'RS', -99, -1062, 34, 12.11)
    add_sample_s(data, 'RS', -99, -1062, 49, 13.39)

    add_sample_upw(
        data, 'ASW85', 414, 1080, 10, 8.6, 3.7, 202.9,
        0.5 * (1.250**2 + 0.867**2 + 0.491**2)
        )
    add_sample_upw(
        data, 'ASW50', 651, 1336, 10, 7.4, 2.9, 192.4,
        0.5 * (1.440**2 + 0.706**2 + 0.516**2)
        )
    add_sample_upw(
        data, 'ASW35', 763, 1456, 10, 7.9, 11.2, 195.0,
        0.5 * (1.456**2 + 1.080**2 + 0.637**2)
        )
    add_sample_upw(
        data, 'ASW20', 858, 1562, 10, 11.5, 15.9, 199.6,
        0.5 * (1.290**2 + 1.205**2 + 0.623**2)
        )
    add_sample_upw(
        data, 'ASW10', 920, 1625, 10, 14.7, 14.9, 207.4,
        0.5 * (1.155**2 + 1.360**2 + 0.669**2)
        )

    add_sample_upw(
        data, 'ASW60', 571, 1253, 6, 7.867, 1.627, 198.653,
        0.5 * (1.351**2 + 0.931**2 + 0.394**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 10, 8.424, 1.221, 199.698,
        0.5 * (1.350**2 + 0.955**2 + 0.453**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 20, 8.834, 2.417, 202.792,
        0.5 * (1.276**2 + 0.947**2 + 0.547**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 31, 10.340, -2.180, 199.424,
        0.5 * (1.280**2 + 0.947**2 + 0.588**2)
        )

    add_sample_upw(
        data, 'HT', 982, 1703, 10, 17.9, 2.9, 203.0,
        0.5 * (1.120**2 + 1.115**2 + 0.615**2)
        )
    add_sample_s(data, 'HT', 982, 1703, 1, 14.31)
    add_sample_s(data, 'HT', 982, 1703, 3, 17.34)
    add_sample_s(data, 'HT', 982, 1703, 5, 18.12)
    add_sample_s(data, 'HT', 982, 1711, 8, 18.01)
    add_sample_s(data, 'HT', 982, 1711, 15, 18.29)
    add_sample_s(data, 'HT', 982, 1711, 24, 17.95)
    add_sample_s(data, 'HT', 982, 1711, 34, 17.54)

    add_sample_upw(
        data, 'ANE10', 1055, 1770, 10, 13.3, -10.7, 206.0,
        0.5 * (1.790**2 + 1.110**2 + 0.575**2)
        )
    add_sample_upw(
        data, 'ANE20', 1124, 1842, 10, 5.9, -12.4, 194.0,
        0.5 * (2.820**2 + 1.800**2 + 0.975**2)
        )
    add_sample_upw(
        data, 'ANE40', 1262, 1975, 10, 3.1, -7.3, 184.6,
        0.5 * (2.260**2 + 2.095**2 + 1.335**2)
        )

    add_sample_w(data, 'CP', 1289, 1451, 1.6, 14.63, np.nan, np.nan, np.nan)
    add_sample_w(data, 'CP', 1289, 1451, 3.0, 15.80, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 5.0, 15.99, 1.08, 217,
        0.5 * (1.357**2 + 1.064**2 + 0.363**2)
        )
    add_sample_w(data, 'CP', 1289, 1451, 7.4, 18.17, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 10.0, 16.45, 1.49, 213,
        0.5 * (1.119**2 + 1.057**2 + 0.557**2)
        )
    add_sample_w(
        data, 'CP', 1289, 1451, 16.0, 16.92, 0.85, 211,
        0.5 * (1.119**2 + 1.007**2 + 0.685**2)
        )
    add_sample_w(data, 'CP', 1287, 1415, 10, 15.59, np.nan, 220)

    add_sample_w(data, 'AASW10', 1214, 1354, 10, 14.48, np.nan, 213)
    add_sample_w(
        data, 'AASW10', 1224, 1346, 10, 14.88, 2.57, 212,
        0.5 * (1.141**2 + 1.180**2 + 0.631**2)
        )
    add_sample_w(data, 'AASW20', 1154, 1278, 10, 11.81, np.nan, 214)
    add_sample_w(data, 'AASW30', 1084, 1211, 10, 9.68, np.nan, 207)
    add_sample_w(
        data, 'AASW30', 1094, 1202, 10, 9.68, 2.49, 206,
        0.5 * (1.254**2 + 1.208**2 + 0.684**2)
        )
    add_sample_w(data, 'AASW40', 1018, 1132, 10, 7.70, np.nan, 209)
    add_sample_w(
        data, 'AASW50', 953, 1058, 10, 6.65, 0.57, 201,
        0.5 * (1.525**2 + 1.011**2 + 0.593**2)
        )

    return data


def TU03B():
    '''
    Get the measurements from the TU03B run.
    
    Returns
    -------
    data : dict
        Dictionary with the measurements from the TU03B run
    '''
    data = {}
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 9.11, 2.2, 211,
        0.5 * (1.409**2 + 0.965**2 + 0.826**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 47, 11.66, 0.1, 209,
        0.5 * (1.166**2 + 0.796**2 + 0.710**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 8.6, 2.5, 207.3,
        0.5 * (1.223**2 + 0.704**2 + 0.413**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 8.9, -2.0, 207.9,
        0.5 * (1.327**2 + 0.859**2 + 0.643**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 20, 10.1, -1.8, 207.1,
        0.5 * (1.212**2 + 0.764**2 + 0.597**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 30, 10.6, -0.1, 210.5,
        0.5 * (1.167**2 + 0.810**2 + 0.650**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 40, 11.2, -1.3, 208.9,
        0.5 * (1.123**2 + 0.776**2 + 0.593**2)
        )

    add_sample_s(data, 'RS', -99, -1062, 4.9, 7.88)
    add_sample_w(data, 'RS', -99, -1062, 10.0, 9.03, np.nan, 207)
    add_sample_s(data, 'RS', -99, -1062, 16.9, 9.81)

    add_sample_s(data, 'RS', -99, -1062, 3, 7.10)
    add_sample_s(data, 'RS', -99, -1062, 5, 7.86)
    add_sample_s(data, 'RS', -99, -1062, 8, 8.44)
    add_sample_s(data, 'RS', -99, -1062, 15, 9.35)
    add_sample_s(data, 'RS', -99, -1062, 24, 10.19)
    add_sample_s(data, 'RS', -99, -1062, 34, 10.84)
    add_sample_s(data, 'RS', -99, -1062, 49, 11.96)

    add_sample_upw(
        data, 'ASW85', 414, 1080, 10, 7.8, 3.9, 201.6,
        0.5 * (1.200**2 + 0.762**2 + 0.463**2)
        )
    add_sample_upw(
        data, 'ASW50', 651, 1336, 10, 6.7, 2.8, 192.9,
        0.5 * (1.350**2 + 0.683**2 + 0.475**2)
        )
    add_sample_upw(
        data, 'ASW35', 763, 1456, 10, 7.2, 11.5, 196.0,
        0.5 * (1.243**2 + 1.038**2 + 0.580**2)
        )
    add_sample_upw(
        data, 'ASW20', 858, 1562, 10, 10.5, 16.0, 200.6,
        0.5 * (1.115**2 + 1.126**2 + 0.565**2)
        )
    add_sample_upw(
        data, 'ASW10', 920, 1625, 10, 13.2, 14.5, 207.9,
        0.5 * (1.059**2 + 1.232**2 + 0.577**2)
        )

    add_sample_upw(
        data, 'ASW60', 571, 1253, 6, 7.060, 1.358, 199.461,
        0.5 * (1.378**2 + 0.889**2 + 0.363**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 10, 7.754, 0.927, 200.608,
        0.5 * (1.379**2 + 0.893**2 + 0.421**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 20, 8.089, 2.108, 203.288,
        0.5 * (1.285**2 + 0.839**2 + 0.497**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 31, 9.422, -1.821, 199.579,
        0.5 * (1.280**2 + 0.841**2 + 0.556**2)
        )

    add_sample_upw(
        data, 'HT', 982, 1711, 47, 15.9, 3.0, np.nan,
        0.5 * (1.20**2 + 0.88**2 + 0.83**2)
        )
    add_sample_upw(
        data, 'HT', 982, 1711, 6, 15.8, 1.2, np.nan,
        0.5 * (1.41**2 + 1.11**2 + 0.68**2)
        )
    add_sample_upw(
        data, 'HT', 982, 1711, 4, 16.0, -0.6, np.nan,
        0.5 * (1.60**2 + 1.22**2 + 0.65**2)
        )
    add_sample_upw(
        data, 'HT', 982, 1711, 2, 15.7, 1.2, np.nan,
        0.5 * (1.99**2 + 1.35**2 + 0.83**2)
        )

    add_sample_upw(
        data, 'HT', 982, 1703, 10, 16.2, 2.7, 203.4,
        0.5 * (1.100**2 + 1.034**2 + 0.531**2)
        )
    add_sample_s(data, 'HT', 982, 1703, 1, 12.99)
    add_sample_s(data, 'HT', 982, 1703, 3, 15.71)
    add_sample_s(data, 'HT', 982, 1703, 5, 16.38)
    add_sample_s(data, 'HT', 982, 1711, 8, 16.30)
    add_sample_s(data, 'HT', 982, 1711, 15, 16.63)
    add_sample_s(data, 'HT', 982, 1711, 24, 16.15)
    add_sample_s(data, 'HT', 982, 1711, 34, 15.77)

    add_sample_upw(
        data, 'ANE10', 1055, 1770, 10, 12.0, -11.1, 206.5,
        0.5 * (1.758**2 + 1.012**2 + 0.531**2)
        )
    add_sample_upw(
        data, 'ANE20', 1124, 1842, 10, 5.6, -13.0, 195.9,
        0.5 * (2.560**2 + 1.502**2 + 0.881**2)
        )
    add_sample_upw(
        data, 'ANE40', 1262, 1975, 10, 3.0, -5.7, 188.1,
        0.5 * (1.983**2 + 1.798**2 + 1.192**2)
        )

    add_sample_w(data, 'CP', 1289, 1451, 1.6, 13.01, np.nan, np.nan, np.nan)
    add_sample_w(data, 'CP', 1289, 1451, 3.0, 14.27, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 5.0, 14.57, 1.00, 217,
        0.5 * (1.287**2 + 0.986**2 + 0.330**2)
        )
    add_sample_w(data, 'CP', 1289, 1451, 7.4, 16.35, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 10.0, 15.05, 1.37, 213,
        0.5 * (1.093**2 + 0.947**2 + 0.483**2)
        )
    add_sample_w(
        data, 'CP', 1289, 1451, 16.0, 15.49, 0.80, 212,
        0.5 * (1.061**2 + 0.901**2 + 0.616**2)
        )
    add_sample_w(data, 'CP', 1287, 1415, 10, 14.17, np.nan, 218)

    add_sample_w(data, 'AASW10', 1214, 1354, 10, 13.15, np.nan, 211)
    add_sample_w(
        data, 'AASW10', 1224, 1346, 10, 13.62, 2.35, 212,
        0.5 * (1.133**2 + 1.061**2 + 0.584**2)
        )
    add_sample_w(data, 'AASW20', 1154, 1278, 10, 10.69, np.nan, 214)
    add_sample_w(data, 'AASW30', 1084, 1211, 10, 8.86, np.nan, 208)
    add_sample_w(
        data, 'AASW30', 1094, 1202, 10, 8.91, 2.30, 205,
        0.5 * (1.143**2 + 1.139**2 + 0.619**2)
        )
    add_sample_w(data, 'AASW40', 1018, 1132, 10, 7.06, np.nan, 210)
    add_sample_w(
        data, 'AASW50', 953, 1058, 10, 6.14, 0.53, 202,
        0.5 * (1.387**2 + 1.022**2 + 0.563**2)
        )

    return data


def TU05A():
    '''
    Get the measurements from the TU05A run.
    
    Returns
    -------
    data : dict
        Dictionary with the measurements from the TU05A run
    '''
    data = {}
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 10.19, 1.5, 285,
        0.5 * (1.655**2 + 1.246**2 + 0.974**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 47, 13.20, 0.3, 281,
        0.5 * (1.236**2 + 0.922**2 + 0.563**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 9.2, 1.2, 285.3,
        0.5 * (1.275**2 + 0.967**2 + 0.426**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 9.7, -0.4, 286.5,
        0.5 * (1.635**2 + 1.095**2 + 0.778**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 20, 11.8, -3.6, 283.0,
        0.5 * (1.595**2 + 0.913**2 + 0.557**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 30, 12.3, -0.7, 286.0,
        0.5 * (1.360**2 + 0.935**2 + 0.595**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 40, 13.2, -1.6, 286.9,
        0.5 * (1.280**2 + 0.914**2 + 0.493**2)
        )

    add_sample_s(data, 'RS', -99, -1062, 4.9, 8.71)
    add_sample_w(data, 'RS', -99, -1062, 10.0, 9.78, np.nan, 283)
    add_sample_s(data, 'RS', -99, -1062, 16.9, 11.03)

    add_sample_s(data, 'RS', -99, -1062, 3, 8.11)
    add_sample_s(data, 'RS', -99, -1062, 5, 8.84)
    add_sample_s(data, 'RS', -99, -1062, 8, 9.45)
    add_sample_s(data, 'RS', -99, -1062, 15, 10.70)
    add_sample_s(data, 'RS', -99, -1062, 24, 12.06)
    add_sample_s(data, 'RS', -99, -1062, 34, 12.57)
    add_sample_s(data, 'RS', -99, -1062, 49, 13.22)

    add_sample_upw(
        data, 'ASW85', 414, 1080, 10, 9.5, -0.3, 286.3,
        0.5 * (1.305**2 + 0.907**2 + 0.321**2)
        )
    add_sample_upw(
        data, 'ASW50', 651, 1336, 10, 9.6, 2.9, 289.0,
        0.5 * (1.280**2 + 1.005**2 + 0.456**2)
        )
    add_sample_upw(
        data, 'ASW35', 763, 1456, 10, 9.6, 4.3, 287.1,
        0.5 * (1.475**2 + 1.061**2 + 0.508**2)
        )
    add_sample_upw(
        data, 'ASW10', 920, 1625, 10, 12.9, 4.6, 282.1,
        0.5 * (1.620**2 + 1.230**2 + 0.319**2)
        )

    add_sample_upw(
        data, 'ASW60', 571, 1253, 6, 8.744, 1.189, 291.439,
        0.5 * (1.509**2 + 0.919**2 + 0.380**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 10, 8.921, 2.866, 281.507,
        0.5 * (1.522**2 + 0.681**2 + 0.478**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 20, 10.588, 3.516, 289.882,
        0.5 * (1.562**2 + 1.036**2 + 0.524**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 31, 11.400, 1.218, 291.875,
        0.5 * (1.617**2 + 0.989**2 + 0.606**2)
        )

    add_sample_upw(
        data, 'HT', 982, 1703, 10, 14.3, 1.4, 274.6,
        0.5 * (1.445**2 + 1.389**2 + 0.268**2)
        )
    add_sample_s(data, 'HT', 982, 1703, 1, 9.88)
    add_sample_s(data, 'HT', 982, 1703, 3, 12.14)
    add_sample_s(data, 'HT', 982, 1703, 5, 13.34)
    add_sample_s(data, 'HT', 982, 1711, 8, 13.70)
    add_sample_s(data, 'HT', 982, 1711, 24, 14.83)
    add_sample_s(data, 'HT', 982, 1711, 34, 14.64)

    add_sample_upw(
        data, 'ANE10', 1055, 1770, 10, 11.6, -3.7, 282.0,
        0.5 * (1.680**2 + 1.164**2 + 0.374**2)
        )
    add_sample_upw(
        data, 'ANE20', 1124, 1842, 10, 11.1, -8.5, 291.5,
        0.5 * (1.690**2 + 0.965**2 + 0.526**2)
        )
    add_sample_upw(
        data, 'ANE40', 1262, 1975, 10, 10.1, -1.7, 291.3,
        0.5 * (1.585**2 + 0.955**2 + 0.365**2)
        )

    add_sample_w(data, 'CP', 1289, 1451, 1.6, 10.33, np.nan, np.nan, np.nan)
    add_sample_w(data, 'CP', 1289, 1451, 3.0, 11.80, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 5.0, 11.56, 0.14, 279,
        0.5 * (1.346**2 + 1.036**2 + 0.246**2)
        )
    add_sample_w(data, 'CP', 1289, 1451, 7.4, 13.84, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 10.0, 12.78, 0.16, 279,
        0.5 * (1.357**2 + 1.133**2 + 0.305**2)
        )
    add_sample_w(
        data, 'CP', 1289, 1451, 16.0, 13.31, -0.03, 279,
        0.5 * (1.318**2 + 1.166**2 + 0.361**2)
        )
    add_sample_w(data, 'CP', 1287, 1415, 10, 12.82, np.nan, 285)

    add_sample_w(data, 'AASW10', 1214, 1354, 10, 12.63, np.nan, 284)
    add_sample_w(
        data, 'AASW10', 1224, 1346, 10, 12.63, 1.13, 280,
        0.5 * (1.388**2 + 1.151**2 + 0.416**2)
        )
    add_sample_w(data, 'AASW20', 1154, 1278, 10, 11.76, np.nan, 282)
    add_sample_w(data, 'AASW30', 1084, 1211, 10, 10.89, np.nan, 289)
    add_sample_w(
        data, 'AASW30', 1094, 1202, 10, 11.02, 1.09, 286,
        0.5 * (1.355**2 + 0.981**2 + 0.496**2)
        )
    add_sample_w(data, 'AASW40', 1018, 1132, 10, 9.87, np.nan, 298)
    add_sample_w(
        data, 'AASW50', 953, 1058, 10, 9.42, 0.25, 289,
        0.5 * (1.501**2 + 0.890**2 + 0.446**2)
        )

    return data


def TU05B():
    '''
    Get the measurements from the TU05B run.
    
    Returns
    -------
    data : dict
        Dictionary with the measurements from the TU05B run
    '''
    data = {}
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 8.69, 0.4, 303,
        0.5 * (1.630**2 + 1.153**2 + 0.948**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 47, 11.97, -0.6, 303,
        0.5 * (1.277**2 + 0.835**2 + 0.613**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 8.2, 0.7, 306.1,
        0.5 * (1.313**2 + 0.842**2 + 0.422**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 8.6, -1.5, 307.2,
        0.5 * (1.553**2 + 0.971**2 + 0.824**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 20, 10.5, -3.6, 303.7,
        0.5 * (1.428**2 + 0.814**2 + 0.600**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 30, 11.0, -1.5, 307.3,
        0.5 * (1.276**2 + 0.718**2 + 0.599**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 40, 12.0, -1.5, 307.9,
        0.5 * (1.254**2 + 0.724**2 + 0.581**2)
        )

    add_sample_s(data, 'RS', -99, -1062, 4.9, 7.21)
    add_sample_w(data, 'RS', -99, -1062, 10.0, 8.28, np.nan, 303)
    add_sample_s(data, 'RS', -99, -1062, 16.9, 9.54)

    add_sample_s(data, 'RS', -99, -1062, 3, 6.98)
    add_sample_s(data, 'RS', -99, -1062, 5, 7.63)
    add_sample_s(data, 'RS', -99, -1062, 8, 8.21)
    add_sample_s(data, 'RS', -99, -1062, 15, 9.57)
    add_sample_s(data, 'RS', -99, -1062, 24, 10.81)
    add_sample_s(data, 'RS', -99, -1062, 34, 11.28)
    add_sample_s(data, 'RS', -99, -1062, 49, 12.00)

    add_sample_upw(
        data, 'ASW85', 414, 1080, 10, 8.9, 0.0, 305.0,
        0.5 * (1.275**2 + 0.894**2 + 0.216**2)
        )
    add_sample_upw(
        data, 'ASW50', 651, 1336, 10, 9.3, 2.5, 305.3,
        0.5 * (1.215**2 + 0.820**2 + 0.403**2)
        )
    add_sample_upw(
        data, 'ASW35', 763, 1456, 10, 9.2, 1.4, 303.6,
        0.5 * (1.473**2 + 0.816**2 + 0.401**2)
        )
    add_sample_upw(
        data, 'ASW10', 920, 1625, 10, 11.4, 0.8, 305.9,
        0.5 * (1.472**2 + 0.909**2 + 0.364**2)
        )

    add_sample_upw(
        data, 'ASW60', 571, 1253, 6, 9.302, 1.423, 309.123,
        0.5 * (1.538**2 + 0.986**2 + 0.359**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 10, 8.941, 4.031, 303.284,
        0.5 * (1.414**2 + 1.065**2 + 0.415**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 20, 10.766, 4.226, 309.379,
        0.5 * (1.385**2 + 0.978**2 + 0.472**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 31, 11.302, 0.325, 311.416,
        0.5 * (1.230**2 + 0.938**2 + 0.575**2)
        )

    add_sample_upw(
        data, 'HT', 982, 1703, 10, 11.6, 1.1, 303.5,
        0.5 * (1.345**2 + 1.038**2 + 0.253**2)
        )
    add_sample_s(data, 'HT', 982, 1703, 1, 7.39)
    add_sample_s(data, 'HT', 982, 1703, 3, 9.41)
    add_sample_s(data, 'HT', 982, 1703, 5, 10.51)
    add_sample_s(data, 'HT', 982, 1711, 8, 11.36)
    add_sample_s(data, 'HT', 982, 1711, 24, 12.78)
    add_sample_s(data, 'HT', 982, 1711, 34, 13.31)

    add_sample_upw(
        data, 'ANE10', 1055, 1770, 10, 10.9, -0.2, 307.5,
        0.5 * (1.490**2 + 0.943**2 + 0.270**2)
        )
    add_sample_upw(
        data, 'ANE20', 1124, 1842, 10, 10.7, -3.4, 313.2,
        0.5 * (1.482**2 + 0.849**2 + 0.387**2)
        )

    add_sample_w(data, 'CP', 1289, 1451, 1.6, 8.33, np.nan, np.nan, np.nan)
    add_sample_w(data, 'CP', 1289, 1451, 3.0, 9.51, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 5.0, 9.23, -0.06, 305,
        0.5 * (1.182**2 + 0.873**2 + 0.271**2)
        )
    add_sample_w(data, 'CP', 1289, 1451, 7.4, 11.33, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 10.0, 10.37, -0.14, 304,
        0.5 * (1.211**2 + 0.946**2 + 0.329**2)
        )
    add_sample_w(
        data, 'CP', 1289, 1451, 16.0, 11.10, -0.15, 305,
        0.5 * (1.227**2 + 0.999**2 + 0.380**2)
        )
    add_sample_w(data, 'CP', 1287, 1415, 10, 10.04, np.nan, 310)

    add_sample_w(data, 'AASW10', 1214, 1354, 10, 10.61, np.nan, 307)
    add_sample_w(
        data, 'AASW10', 1224, 1346, 10, 10.97, 0.29, 303,
        0.5 * (1.232**2 + 0.906**2 + 0.398**2)
        )
    add_sample_w(data, 'AASW20', 1154, 1278, 10, 10.16, np.nan, 303)
    add_sample_w(data, 'AASW30', 1084, 1211, 10, 10.38, np.nan, 307)
    add_sample_w(
        data, 'AASW30', 1094, 1202, 10, 10.29, 0.19, 304,
        0.5 * (1.298**2 + 0.810**2 + 0.385**2)
        )
    add_sample_w(data, 'AASW40', 1018, 1132, 10, 9.61, np.nan, 310)
    add_sample_w(
        data, 'AASW50', 953, 1058, 10, 9.43, 0.07, 305,
        0.5 * (1.234**2 + 0.869**2 + 0.367**2)
        )

    return data


def TU05C():
    '''
    Get the measurements from the TU05C run.
    
    Returns
    -------
    data : dict
        Dictionary with the measurements from the TU05C run
    '''
    data = {}
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 8.15, 0.0, 298,
        0.5 * (1.477**2 + 1.103**2 + 0.871**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 47, 11.11, -0.9, 300,
        0.5 * (1.067**2 + 0.761**2 + 0.560**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 7.7, 0.9, 303.2,
        0.5 * (1.183**2 + 0.799**2 + 0.405**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 8.0, -1.2, 304.0,
        0.5 * (1.363**2 + 0.945**2 + 0.709**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 20, 9.7, -3.7, 300.6,
        0.5 * (1.307**2 + 0.805**2 + 0.567**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 30, 10.2, -1.2, 304.8,
        0.5 * (1.142**2 + 0.738**2 + 0.580**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 40, 11.2, -1.0, 304.9,
        0.5 * (1.078**2 + 0.695**2 + 0.530**2)
        )

    add_sample_s(data, 'RS', -99, -1062, 4.9, 6.86)
    add_sample_w(data, 'RS', -99, -1062, 10.0, 7.85, np.nan, 299)
    add_sample_s(data, 'RS', -99, -1062, 16.9, 8.92)

    add_sample_s(data, 'RS', -99, -1062, 3, 6.64)
    add_sample_s(data, 'RS', -99, -1062, 5, 7.26)
    add_sample_s(data, 'RS', -99, -1062, 8, 7.79)
    add_sample_s(data, 'RS', -99, -1062, 15, 8.84)
    add_sample_s(data, 'RS', -99, -1062, 24, 10.03)
    add_sample_s(data, 'RS', -99, -1062, 34, 10.50)
    add_sample_s(data, 'RS', -99, -1062, 49, 11.15)

    add_sample_upw(
        data, 'ASW85', 414, 1080, 10, 8.6, 0.4, 302.5,
        0.5 * (1.177**2 + 0.742**2 + 0.232**2)
        )
    add_sample_upw(
        data, 'ASW50', 651, 1336, 10, 8.6, 2.7, 303.4,
        0.5 * (1.079**2 + 0.676**2 + 0.382**2)
        )
    add_sample_upw(
        data, 'ASW35', 763, 1456, 10, 8.6, 1.8, 300.4,
        0.5 * (1.293**2 + 0.799**2 + 0.385**2)
        )
    add_sample_upw(
        data, 'ASW10', 920, 1625, 10, 10.4, 0.9, 304.2,
        0.5 * (1.190**2 + 0.754**2 + 0.315**2)
        )

    add_sample_upw(
        data, 'ASW60', 571, 1253, 6, 8.670, 1.371, 306.282,
        0.5 * (1.361**2 + 0.892**2 + 0.370**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 10, 8.127, 4.054, 298.457,
        0.5 * (1.181**2 + 0.847**2 + 0.432**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 20, 10.101, 4.139, 306.634,
        0.5 * (1.194**2 + 0.886**2 + 0.457**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 31, 10.611, 0.502, 308.606,
        0.5 * (1.067**2 + 0.820**2 + 0.547**2)
        )

    add_sample_upw(
        data, 'HT', 982, 1703, 10, 11.0, 1.1, 301.1,
        0.5 * (1.150**2 + 0.772**2 + 0.217**2)
        )

    add_sample_upw(
        data, 'ANE10', 1055, 1770, 10, 10.4, -1.0, 304.5,
        0.5 * (1.273**2 + 0.792**2 + 0.264**2)
        )
    add_sample_upw(
        data, 'ANE20', 1124, 1842, 10, 9.6, -4.4, 311.2,
        0.5 * (1.270**2 + 0.764**2 + 0.368**2)
        )

    add_sample_w(data, 'CP', 1289, 1451, 1.6, 7.97, np.nan, np.nan, np.nan)
    add_sample_w(data, 'CP', 1289, 1451, 3.0, 9.20, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 5.0, 8.94, -0.05, 301,
        0.5 * (1.125**2 + 0.787**2 + 0.255**2)
        )
    add_sample_w(data, 'CP', 1289, 1451, 7.4, 10.85, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 10.0, 10.07, -0.13, 300,
        0.5 * (1.117**2 + 0.802**2 + 0.304**2)
        )
    add_sample_w(
        data, 'CP', 1289, 1451, 16.0, 10.72, -0.19, 301,
        0.5 * (1.113**2 + 0.841**2 + 0.344**2)
        )
    add_sample_w(data, 'CP', 1287, 1415, 10, 9.57, np.nan, 308)

    add_sample_w(data, 'AASW10', 1214, 1354, 10, 9.84, np.nan, 304)
    add_sample_w(
        data, 'AASW10', 1224, 1346, 10, 10.24, 0.39, 299,
        0.5 * (1.156**2 + 0.868**2 + 0.387**2)
        )
    add_sample_w(data, 'AASW20', 1154, 1278, 10, 9.59, np.nan, 299)
    add_sample_w(data, 'AASW30', 1084, 1211, 10, 9.87, np.nan, 304)
    add_sample_w(
        data, 'AASW30', 1094, 1202, 10, 9.80, 0.31, 301,
        0.5 * (1.218**2 + 0.732**2 + 0.368**2)
        )
    add_sample_w(data, 'AASW40', 1018, 1132, 10, 8.99, np.nan, 305)
    add_sample_w(
        data, 'AASW50', 953, 1058, 10, 9.07, 0.09, 302,
        0.5 * (1.166**2 + 0.738**2 + 0.361**2)
        )

    return data


def TU07B():
    '''
    Get the measurements from the TU07B run.
    
    Returns
    -------
    data : dict
        Dictionary with the measurements from the TU07B run
    '''
    data = {}
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 10.27, 0.4, 256,
        0.5 * (1.600**2 + 1.546**2 + 0.822**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 47, 12.56, 0.3, 258,
        0.5 * (1.722**2 + 1.606**2 + 0.648**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 9.8, 1.9, 257.2,
        0.5 * (1.533**2 + 1.395**2 + 0.376**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 10, 9.8, -1.4, 258.0,
        0.5 * (1.607**2 + 1.500**2 + 0.674**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 20, 11.4, -3.3, 257.7,
        0.5 * (1.670**2 + 1.463**2 + 0.575**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 30, 11.6, 0.1, 259.6,
        0.5 * (1.570**2 + 1.535**2 + 0.631**2)
        )
    add_sample_upw(
        data, 'RS', -99, -1062, 40, 12.4, -1.4, 259.8,
        0.5 * (1.703**2 + 1.509**2 + 0.567**2)
        )

    add_sample_s(data, 'RS', -99, -1062, 4.9, 8.79)
    add_sample_w(data, 'RS', -99, -1062, 10.0, 10.15, np.nan, 256)

    add_sample_s(data, 'RS', -99, -1062, 3, 8.05)
    add_sample_s(data, 'RS', -99, -1062, 5, 8.95)
    add_sample_s(data, 'RS', -99, -1062, 8, 9.66)
    add_sample_s(data, 'RS', -99, -1062, 15, 10.71)
    add_sample_s(data, 'RS', -99, -1062, 24, 11.62)
    add_sample_s(data, 'RS', -99, -1062, 34, 12.10)
    add_sample_s(data, 'RS', -99, -1062, 49, 12.64)

    add_sample_upw(
        data, 'ASW50', 651, 1336, 10, 6.8, 4.4, 258.1,
        0.5 * (1.333**2 + 1.319**2 + 0.519**2)
        )
    add_sample_upw(
        data, 'ASW35', 763, 1456, 10, 7.9, 9.0, 259.1,
        0.5 * (1.600**2 + 1.333**2 + 0.601**2)
        )
    add_sample_upw(
        data, 'ASW20', 858, 1562, 10, 10.5, 15.0, 250.1,
        0.5 * (1.433**2 + 1.850**2 + 0.598**2)
        )
    add_sample_upw(
        data, 'ASW10', 920, 1625, 10, 13.1, 13.4, 249.4,
        0.5 * (1.477**2 + 1.960**2 + 0.602**2)
        )

    add_sample_upw(
        data, 'ASW60', 571, 1253, 6, 7.431, 0.804, 260.949,
        0.5 * (1.723**2 + 0.998**2 + 0.357**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 10, 8.688, 2.073, 265.525,
        0.5 * (1.725**2 + 0.573**2 + 0.409**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 20, 9.860, 3.189, 258.859,
        0.5 * (1.880**2 + 1.079**2 + 0.453**2)
        )
    add_sample_upw(
        data, 'ASW60', 571, 1253, 31, 10.665, 1.000, 261.375,
        0.5 * (1.859**2 + 1.107**2 + 0.516**2)
        )

    add_sample_upw(
        data, 'HT', 982, 1703, 10, 15.4, 0.8, 239.2,
        0.5 * (1.607**2 + 1.757**2 + 0.188**2)
        )
    add_sample_s(data, 'HT', 982, 1703, 1, 11.91)
    add_sample_s(data, 'HT', 982, 1703, 3, 13.49)
    add_sample_s(data, 'HT', 982, 1703, 5, 15.43)
    add_sample_s(data, 'HT', 982, 1711, 8, 15.23)
    add_sample_s(data, 'HT', 982, 1711, 15, 15.37)
    add_sample_s(data, 'HT', 982, 1711, 24, 15.63)
    add_sample_s(data, 'HT', 982, 1711, 34, 15.31)
    add_sample_s(data, 'HT', 982, 1711, 49, 15.21)

    add_sample_upw(
        data, 'ANE10', 1055, 1770, 10, 12.0, -9.9, 248.9,
        0.5 * (1.567**2 + 1.568**2 + 0.521**2)
        )
    add_sample_upw(
        data, 'ANE20', 1124, 1842, 10, 7.6, -17.3, 260.7,
        0.5 * (1.917**2 + 1.400**2 + 0.567**2)
        )

    add_sample_w(data, 'CP', 1289, 1451, 1.6, 11.56, np.nan, np.nan, np.nan)
    add_sample_w(data, 'CP', 1289, 1451, 3.0, 13.52, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 5.0, 13.45, 0.56, 250,
        0.5 * (1.700**2 + 1.268**2 + 0.295**2)
        )
    add_sample_w(data, 'CP', 1289, 1451, 7.4, 15.14, np.nan, np.nan, np.nan)
    add_sample_w(
        data, 'CP', 1289, 1451, 10.0, 14.22, 0.77, 249,
        0.5 * (1.608**2 + 1.464**2 + 0.390**2)
        )
    add_sample_w(
        data, 'CP', 1289, 1451, 16.0, 14.55, 0.28, 248,
        0.5 * (1.581**2 + 1.491**2 + 0.432**2)
        )
    add_sample_w(data, 'CP', 1287, 1415, 10, 12.04, np.nan, 246)

    add_sample_w(data, 'AASW10', 1214, 1354, 3, 11.78, np.nan, 248)
    add_sample_w(
        data, 'AASW10', 1224, 1346, 10, 13.38, 1.94, 250,
        0.5 * (1.425**2 + 1.513**2 + 0.441**2)
        )
    add_sample_w(data, 'AASW20', 1154, 1278, 3, 9.66, np.nan, 262)
    add_sample_w(data, 'AASW30', 1084, 1211, 3, 8.95, np.nan, 258)
    add_sample_w(
        data, 'AASW30', 1094, 1202, 10, 10.05, 1.95, 259,
        0.5 * (1.603**2 + 1.401**2 + 0.571**2)
        )
    add_sample_w(data, 'AASW40', 1018, 1132, 3, 7.02, np.nan, 272)
    add_sample_w(
        data, 'AASW50', 953, 1058, 10, 7.64, 0.49, 262,
        0.5 * (1.569**2 + 1.242**2 + 0.489**2)
        )

    return data
