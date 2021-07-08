"""
Vesrion 1.0
Latex, temperature, efield
"""

from scipy.optimize import curve_fit
import matplotlib.collections as mcoll
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import multiprocessing as mp
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import scipy as scp
import h5py
from scipy import ndimage
from scipy.interpolate import splev, splprep
# import matplotlib.animation as animation
import time
# from numba import jit
import matplotlib as mpl
mpl.use('Agg')
# import matplotlib.gridspec as gridspec
# from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use('./spectra.mplstyle')

# ------------------------------------
# SIMULATION CONSTANTS AND PARAMETERS
# ------------------------------------
C = 0.5
ME = 1.0
QE = -0.01120846
DENS0 = 20.0

# GAMMA = 8166.4652
# GAMMA = 11555.127
# GAMMA = 16329.956
GAMMA = 20015.131
LSI = 100.0
V_JET = -0.1*C
GM_JET = C/np.sqrt(C**2-V_JET**2)
THETA = 75.0
PHI = 0.0
# B0 = 1.098
# B0 = 0.776
# B0 = 0.5491
B0 = 0.448
B0X = B0*np.cos(THETA*np.pi/180.0)
B0Y = B0*np.sin(THETA*np.pi/180.0)*np.cos(PHI*np.pi/180.0)
B0Z = B0*np.sin(THETA*np.pi/180.0)*np.sin(PHI*np.pi/180.0)
V_SH = 0.16*C
VT = V_SH/np.cos(THETA*np.pi/180.0)
GM_T = C/np.sqrt(C**2-VT**2)

# Box parameters
NPX = np.int(12)
NPY = np.int(120)
NFX = np.int(5000)
NFY = np.int(10)
MX = np.int(NPX*NFX)
MY = np.int(NPY*NFY)

# Spectra parameters
GPS = 500
PMIN = 0.00001
PMAX = 1000.0
DP = np.log10(PMAX/PMIN)/GPS
DELTA = np.int(100)
XMING = 0.0

# Log plots
ILOGD = True
AFLOORD = -0.3
ILOGB = True
AFLOORB = -2.0
ILOGP = True
ILOGE = True
AFLOORE = -2.0

# RESIZING
IRES = 2
IRESP = 2
IRESF = 2
IRESPS = 5
IRESS = 1

# Limit
STEP_LIMIT = 1000000

# Temp
DELTA_TEMP = np.int(10)

# ------------------
# DATA FOLDER PATHS
# ------------------
trej_path = Path('../result/trace/')
movHR_path = Path('./')
phase_path = Path('../result_b30/phase')
distr_path = Path('../result/distr')
temp_path = Path('../result/temp/')
dens_plot_path = './'
bfield_plot_path = './'
phase_plot_path = '../plots/'
spectra_plot_path = '../spectra/'
spectra_name = '_t75_b20W_'

# ---------------------
# PARTICLE NAMES FILES
# ---------------------
# trej_names = np.loadtxt('./tracing/gamma_trej.txt', dtype=str, usecols=1)

# ----------
# FUNCTIONS
# ----------
# Setup


def chose_Npx(nstep):
    if nstep <= 100000:
        Npx = np.int32(2)
    elif nstep <= 180000:
        Npx = np.int32(3)
    elif nstep <= 260000:
        Npx = np.int32(4)
    elif nstep <= 300000:
        Npx = np.int32(5)
    elif nstep <= 360000:
        Npx = np.int32(6)
    elif nstep <= 430000:
        Npx = np.int32(7)
    elif nstep <= 510000:
        Npx = np.int32(8)
    elif nstep <= 580000:
        Npx = np.int32(9)
    elif nstep <= 650000:
        Npx = np.int32(10)
    elif nstep <= 710000:
        Npx = np.int32(11)
    elif nstep <= 770000:
        Npx = np.int32(12)
    elif nstep <= 850000:
        Npx = np.int32(13)
    elif nstep <= 930000:
        Npx = np.int32(14)
    elif nstep <= 990000:
        Npx = np.int32(15)
    return Npx

# Array creation


def create_density_arrays():
    """
    Density arrays creation: electron and ion density,
    x and y grid pints.
    """
    mxxp = np.int32(MX/IRESP+1)
    myyp = np.int32(MY/IRESP+1)
    mxxRp = np.int32(mxxp/IRES+2)
    myyRp = np.int32(myyp/IRES+2)
    dense_array = np.zeros((mxxRp, myyRp))
    densi_array = np.zeros((mxxRp, myyRp))
    xp = (np.array([i for i in range(np.int32(mxxp/IRES+2))])
          * IRESP*IRES+3.0)/LSI
    yp = (np.array([i for i in range(np.int32(myyp/IRES+2))])
          * IRESP*IRES+3.0)/LSI
    return dense_array, densi_array, xp, yp


def create_bfield_arrays():
    """
    Bfield arrays creation: bx, by, bz components,
    x and y grid pints.
    """
    mxxf = np.int32(MX/IRESF+1)
    myyf = np.int32(MY/IRESF+1)
    mxxRf = np.int32(mxxf/IRES+2)
    myyRf = np.int32(myyf/IRES+2)
    bx_array = np.zeros((mxxRf, myyRf))
    by_array = np.zeros((mxxRf, myyRf))
    bz_array = np.zeros((mxxRf, myyRf))
    xp = (np.array([i for i in range(np.int32(mxxf/IRES+2))])
          * IRESF*IRES+3.0)/LSI
    yp = (np.array([i for i in range(np.int32(myyf/IRES+2))])
          * IRESF*IRES+3.0)/LSI
    return bx_array, by_array, bz_array, xp, yp


def create_efield_arrays():
    """
    Efield arrays creation: ex, ey, ez components,
    x and y grid pints.
    """
    mxxf = np.int32(MX/IRESF+1)
    myyf = np.int32(MY/IRESF+1)
    mxxRf = np.int32(mxxf/IRES+2)
    myyRf = np.int32(myyf/IRES+2)
    bx_array = np.zeros((mxxRf, myyRf))
    by_array = np.zeros((mxxRf, myyRf))
    bz_array = np.zeros((mxxRf, myyRf))
    xp = (np.array([i for i in range(np.int32(mxxf/IRES+2))])
          * IRESF*IRES+3.0)/LSI
    yp = (np.array([i for i in range(np.int32(myyf/IRES+2))])
          * IRESF*IRES+3.0)/LSI
    return ex_array, ey_array, ez_array, xp, yp


def create_phase_arrays():
    mxPS = np.int(MX/IRESPS+2)
    myPS = np.int(400)
    dpPS = 0.025
    px_array = np.zeros((mxPS, myPS))
    py_array = np.zeros((mxPS, myPS))
    pz_array = np.zeros((mxPS, myPS))
    xp = (np.arange(mxPS/IRESS)*IRESPS*IRESS+3)/LSI
    yp = (np.arange(myPS)-myPS/2+1)*dpPS
    return px_array, py_array, pz_array, xp, yp

# Calculations


def dens_log(floor, norm, tab):
    """
    Make logscale density array.
    """
    tab = tab/norm
    index_sign = np.sign(tab)
    tab_mask = ma.masked_equal(tab, 0.0)
    tabb = np.ma.log10(np.abs(tab_mask))
    tabb_mask = ma.masked_less_equal(tabb, floor)
    tabb = tabb_mask.filled(0)
    tabb = tabb*index_sign
    return tabb


def field_log(floor, tab):
    """
    Make logscale field array.
    """
    index_sign = np.sign(tab)
    tab_mask = ma.masked_equal(tab, 0.0)
    tabb_mask = np.abs(tab_mask)
    tabb = np.ma.log10(tabb_mask)
    tabb[tabb < floor] = floor
    index_sign[tabb < floor] = 0.0
    tabb = tabb-floor
    tabb = tabb.filled(0)
    tabb = tabb*index_sign
    return tabb


def shock_position_from_array(dens, xp):
    """
    Find the shock position of density jump
    in the density array (from right).
    """
    tab = np.flip(dens[150])
    # print(tab)
    for i, val in enumerate(tab):
        if val > 0.4:
            index = i
            break
    x_shock = (1-index/dens.shape[1])*np.max(xp)
    return x_shock


def shock_position_linear(step):
    """
    Shock possition assuming that velocity is constant,
    coefficients from regression equation.
    """
    a = 0.00025
    b = 6.64
    return a*step+b


def shock_position_from_file(file, step):
    a = np.where(file[:, 0] == step)
    row = file[a]
    return row[0][1]


def spectrum_from_distr_array(distr, x_range):
    x_left_index = np.int(np.round((x_range[0]-XMING)/DELTA))
    x_right_index = np.int(np.round((x_range[1]-XMING)/DELTA))
    distr_range = distr[(x_left_index):(x_right_index), :, :]
    spctr = np.sum(distr_range, axis=(0, 1))
    at = np.sum(spctr)
    spctr = spctr/(at*DP)

    return spctr


def maxwell_fit(dp0fit, dp1fit, xp, Y):
    global pmin, dp
    B = [1000.0, 100.0]

    is1 = np.int32(np.round(np.log10(dp0fit/PMIN)/DP))
    is2 = np.int32(np.round(np.log10(dp1fit/PMIN)/DP))

    Y = Y[is1:is2]
    xpfit = xp[is1:is2]
    weights = 1.0/Y

    B, yfit = curve_fit(funct_ekin_log, xpfit, Y)
    print("B ekin fit electrons ", B)
    xpfit1 = xp
    yf = B[0]*np.sqrt(xpfit1)*np.exp(-B[1]*1.0*xpfit1)
    print("fit")
    return yf


def power_fit(dp0fit, dp1fit, xp, Y):
    global pmin, dp
    B = [0.00036, -2.4]

    is1 = np.int32(np.round(np.log10(dp0fit/PMIN)/DP))
    is2 = np.int32(np.round(np.log10(dp1fit/PMIN)/DP))

    Y = Y[is1:is2]
    xpfit = xp[is1:is2]
    weights = Y*0+1
    print(np.log(Y))

    B, yfit = curve_fit(funct_pow, np.log(xpfit), np.log(Y), B)
    print("B power fit electrons ", B)
    xpfit1 = xp
    yf = B[0]*xpfit1**B[1]
    print("fit")
    return yf, B[1]


def funct_pow(X, A0, A1):
    return np.log(A0)+A1*X


def funct_ekin_log(X, A0, A1, F, *argv):
    bx = np.exp(-A1*X)
    return A0*X*np.sqrt(X)*bx
    if argv != []:
        argv[0] = [[X*np.sqrt(X)*bx], [-A[0]*X*X*np.sqrt(X)*bx]]


def find_gammax(spectrum, xp, cutoff):
    index_tab = np.nonzero(spectrum > cutoff)[0]
    return xp[index_tab[-1]]+1.0

# Loading data


def load_temperature_data(species, step):
    step_str = "{:06d}".format(step)

    xPS = np.int(NPX*NFX/DELTA_TEMP)
    yPS = NPY

    temp = np.loadtxt(temp_path/(step_str+'_T_'+species+'_R'))
    temp_x = np.loadtxt(temp_path/(step_str+'_Tx_'+species+'_R'))
    temp_y = np.loadtxt(temp_path/(step_str+'_Ty_'+species+'_R'))
    temp_z = np.loadtxt(temp_path/(step_str+'_Tz_'+species+'_R'))

    temp = np.reshape(temp, (xPS, yPS))
    temp_x = np.reshape(temp_x, (xPS, yPS))
    temp_y = np.reshape(temp_y, (xPS, yPS))
    temp_z = np.reshape(temp_z, (xPS, yPS))

    return temp, temp_x, temp_y, temp_z


def load_temperature_profiles(species, step):
    temp, temp_x, temp_y, temp_z = load_temperature_data(
        species, step)
    temp_profile = np.mean(temp, axis=1)
    temp_x_profile = np.mean(temp_x, axis=1)
    temp_y_profile = np.mean(temp_y, axis=1)
    temp_z_profile = np.mean(temp_z, axis=1)
    temp_profile = scp.ndimage.filters.gaussian_filter1d(temp_profile, 25.0)
    temp_x_profile = scp.ndimage.filters.gaussian_filter1d(temp_x_profile, 25.0)
    temp_y_profile = scp.ndimage.filters.gaussian_filter1d(temp_y_profile, 25.0)
    temp_z_profile = scp.ndimage.filters.gaussian_filter1d(temp_z_profile, 25.0)

    return temp_profile, temp_x_profile, temp_y_profile, temp_z_profile


def load_trej_data(filename, step_min, step_max):
    """
    Load trej (partcile tracing) file data.
    """
    data = np.loadtxt(trej_path/filename, skiprows=1)
    a = np.where(data[:, 0] > step_min)
    tab = data[a]
    a = np.where(tab[:, 0] <= step_max)
    data = tab[a]
    step = data.T[0]
    gm = data.T[1]
    x = data.T[2]
    y = data.T[3]
    px = data.T[4]
    py = data.T[5]
    pz = data.T[6]
    ex = data.T[7]
    ey = data.T[8]
    ez = data.T[9]
    bx = data.T[10]
    by = data.T[11]
    bz = data.T[12]
    return (step, gm, x, y, px, py, pz, ex, ey, ez, bx, by, bz)


def load_density_data(dense, densi, step):
    """
    Load electron and ion density data from movHR file.
    """
    dense[:, :] = 0.0
    densi[:, :] = 0.0
    print("step_ =", step)
    step_str = "{:d}".format(step)
    step_file_str = "{:06d}".format(step)

    # File opening
    file_name = 'movHR_'+step_file_str+'XY.h5'
    file_id = h5py.File(movHR_path/file_name, "r")

    # Group opening
    group_id = file_id["Step#"+step_str]

    # Atributes reading
    Npx = np.int32(group_id.attrs["Npx"])
    iminreseR = np.int32(group_id.attrs["iminreseR"])
    imaxreseR = np.int32(group_id.attrs["imaxreseR"])
    iminresiR = np.int32(group_id.attrs["iminresiR"])
    imaxresiR = np.int32(group_id.attrs["imaxresiR"])

    # Data reading
    dense[(iminreseR-1):imaxreseR, :] = group_id["denseresR"]
    densi[(iminresiR-1):imaxresiR, :] = group_id["densiresR"]

    file_id.close()

    # Data filtering
    dense = scp.ndimage.filters.gaussian_filter(dense, 0.5)
    densi = scp.ndimage.filters.gaussian_filter(densi, 0.5)

    # Set minimum field strength for log plots
    if ILOGD == True:
        dense = dens_log(AFLOORD, DENS0, dense)
        densi = dens_log(AFLOORD, DENS0, densi)

    return dense, densi


def load_bfield_data_temp(bx, by, bz, step):
    """
    Load bfield data from movHR file.
    """
    bx[:, :] = 0.0
    by[:, :] = 0.0
    bz[:, :] = 0.0
    print("step_ =", step)
    step_str = "{:05d}".format(step)
    if step >= 980000:
        step_str = "{:07d}".format(step)

    # File opening
    file_name = 'movHR_'+step_str+'XY.h5'
    file_id = h5py.File(movHR_path/file_name, 'r')

    # Group opening
    group_id = file_id["Step#"+step_str]

    # Atributes reading
    Npx = np.int32(group_id.attrs["Npx"])
    iminresf = np.int32(group_id.attrs["iminresf"])
    imaxresf = np.int32(group_id.attrs["imaxresf"])

    # Data reading
    bx[(iminresf-1):imaxresf, :] = group_id["bxres"]
    by[(iminresf-1):imaxresf, :] = group_id["byres"]
    bz[(iminresf-1):imaxresf, :] = group_id["bzres"]

    file_id.close()

    # Data filtering
    bx = scp.ndimage.filters.gaussian_filter(bx, 0.5)
    by = scp.ndimage.filters.gaussian_filter(by, 0.5)
    bz = scp.ndimage.filters.gaussian_filter(bz, 0.5)

    return bx, by, bz


def load_bfield_data(bx, by, bz, step):
    """
    Load bfield data from movHR file.
    """
    bx[:, :] = 0.0
    by[:, :] = 0.0
    bz[:, :] = 0.0
    print("step_ =", step)
    step_str = "{:d}".format(step)
    step_file_str = "{:06d}".format(step)

    # File opening
    file_name = 'movHR_'+step_file_str+'XY.h5'
    file_id = h5py.File(movHR_path/file_name, 'r')

    # Group opening
    group_id = file_id["Step#"+step_str]

    # Atributes reading
    Npx = np.int32(group_id.attrs["Npx"])
    iminresf = np.int32(group_id.attrs["iminresf"])
    imaxresf = np.int32(group_id.attrs["imaxresf"])

    # Data reading
    bx[(iminresf-1):imaxresf, :] = (group_id["bxres"]-B0X)/B0
    by[(iminresf-1):imaxresf, :] = (group_id["byres"]-B0Y)/B0
    bz[(iminresf-1):imaxresf, :] = (group_id["bzres"]-B0Z)/B0

    file_id.close()

    # Data filtering
    bx = scp.ndimage.filters.gaussian_filter(bx, 1.0)
    by = scp.ndimage.filters.gaussian_filter(by, 1.0)
    bz = scp.ndimage.filters.gaussian_filter(bz, 1.0)

    # Set minimum field strength for log plots
    if ILOGB == True:
        bx = field_log(AFLOORB, bx)
        by = field_log(AFLOORB, by)
        bz = field_log(AFLOORB, bz)

    return bx, by, bz


def load_efield_data(ex, ey, ez, step):
    """
    Load efield data from movHR file.
    """
    ex[:, :] = 0.0
    ey[:, :] = 0.0
    ez[:, :] = 0.0
    print("step_ =", step)
    step_str = "{:d}".format(step)
    step_file_str = "{:06d}".format(step)

    # File opening
    file_name = 'movHR_'+step_file_str+'XY.h5'
    file_id = h5py.File(movHR_path/file_name, 'r')

    # Group opening
    group_id = file_id["Step#"+step_str]

    # Atributes reading
    Npx = np.int32(group_id.attrs["Npx"])
    iminresf = np.int32(group_id.attrs["iminresf"])
    imaxresf = np.int32(group_id.attrs["imaxresf"])

    # Data reading
    ax = np.array(group_id["exres"])
    ay = np.array(group_id["eyres"])
    az = np.array(group_id["ezres"])
    ex[(iminresf-1):imaxresf, :] = ax/(B0*C)
    ey[(iminresf-1):imaxresf, :] = (ay-V_JET*B0Z)/(B0*C)
    ez[(iminresf-1):imaxresf, :] = (az+V_JET*B0Y)/(B0*C)

    file_id.close()

    # Data filtering
    ex = scp.ndimage.filters.gaussian_filter(ex, 1.0)
    ey = scp.ndimage.filters.gaussian_filter(ey, 1.0)
    ez = scp.ndimage.filters.gaussian_filter(ez, 1.0)

    # Set minimum field strength for log plots
    if ILOGE == True:
        ex = field_log(AFLOORB, ex)
        ey = field_log(AFLOORB, ey)
        ez = field_log(AFLOORB, ez)

    return ex, ey, ez


def load_phase_ion_data(pxi, pyi, pzi, step):
    """
    Load ion phase data from movHR file.
    """
    pxi[:, :] = 0.0
    pyi[:, :] = 0.0
    pzi[:, :] = 0.0
    print("step =", step)
    step_str = "{:06d}".format(step)

    # File opening
    file_name = 'phase_'+step_str+'.h5'
    file_id = h5py.File(phase_path/file_name, "r")

    # Group opening
    group_id = file_id["Step#"+step_str]

    # Atributes reading
    Npx = np.int(group_id.attrs["Npx"])
    igminiR = np.int(group_id.attrs["igminiR"])
    igmaxiR = np.int(group_id.attrs["igmaxiR"])

    # Data reading
    pxi[(igminiR-1):igmaxiR, :] = group_id["pxiR"]
    pyi[(igminiR-1):igmaxiR, :] = group_id["pyiR"]
    pzi[(igminiR-1):igmaxiR, :] = group_id["pziR"]

    # file_id.close()

    if ILOGP == True:
        pxi = np.ma.log10(pxi)
        pyi = np.ma.log10(pyi)
        pzi = np.ma.log10(pzi)

    return pxi, pyi, pzi


def load_phase_data(pxe, pye, pze, step):
    """
    Load electron phase data from movHR file.
    """
    pxe[:, :] = 0.0
    pye[:, :] = 0.0
    pze[:, :] = 0.0
    print("step =", step)
    step_str = "{:d}".format(step)
    step_file_str = "{:06d}".format(step)

    # File opening
    file_name = 'phase_'+step_file_str+'.h5'
    file_id = h5py.File(phase_path/file_name, "r")

    # Group opening
    group_id = file_id["Step#"+step_str]

    # Atributes reading
    Npx = np.int(group_id.attrs["Npx"])
    igmineR = np.int(group_id.attrs["igmineR"])
    igmaxeR = np.int(group_id.attrs["igmaxeR"])

    # Data reading
    pxe[(igmineR-1):igmaxeR, :] = group_id["pxeR"]
    pye[(igmineR-1):igmaxeR, :] = group_id["pyeR"]
    pze[(igmineR-1):igmaxeR, :] = group_id["pzeR"]

    # file_id.close()

    if ILOGP == True:
        pxe = np.ma.log10(pxe)
        pye = np.ma.log10(pye)
        pze = np.ma.log10(pze)

    return pxe, pye, pze


def load_distr_data(xPS, yPS, step):
    print("step =", step)
    step_str = "{:07d}".format(step)

    # File opening
    file_name = step_str+'_spctre_R'
    file = open(distr_path/file_name, 'r')

    # Read spectra data
    row = np.array([])
    row_new = []
    for i, line in enumerate(file):
        if (i+1) % 46 != 0:
            row = np.append(row, np.fromstring(line, dtype=int, sep=" "))
        else:
            row = np.append(row, np.fromstring(line, dtype=int, sep=" "))
            row_new.append(row)
            row = np.array([])

    file.close()
    row_new = np.array(row_new)
    distr = np.reshape(row_new, (xPS, yPS, GPS))

    return distr


def load_spectra(layers_number, layers_thikness, layers_distance,
                 x_left_downstream, x_left_upstream, step_min, step_max, step_spectra):
    spectra_downstream = [[] for i in range(layers_number)]
    spectra_upstream = [[] for i in range(layers_number)]
    gamma_max = [[] for i in range(layers_number)]
    shock_position_file = np.loadtxt("shock_position.txt")
    xp = pow(10, np.log10(PMIN)+np.arange(GPS)*DP)/C**2
    for step in range(step_min, step_max+1, step_spectra):
        step_str = "{:07d}".format(step)
        Npx = chose_Npx(step)

        # Global x-range of particles
        xming = 0
        xmaxg = Npx*NFX

        xPS = np.int(np.ceil((xmaxg-xming)/DELTA))
        yPS = np.int(np.ceil(MY/DELTA))

        distr = load_distr_data(xPS, yPS, step)
        for k in range(1, Npx):
            distr[np.int(np.round((NFX*k-xming)/DELTA)), :, :] = 0.0
        # Shock position
        x_shock = shock_position_from_file(shock_position_file, step)
        x_left = LSI*(x_shock-x_left_downstream)

        for layer in range(layers_number):
            x_left_index = x_left+layers_distance*layer
            x_right_index = x_left+layers_thikness
            spectrum = spectrum_from_distr_array(distr,
                                                 (x_left_index, x_right_index))
            spectra_downstream[layer].append(spectrum)

        x_left = LSI*(x_shock+x_left_upstream)
        for layer in range(layers_number):
            x_left_index = x_left+layers_distance*layer
            x_right_index = x_left+layers_thikness
            spectrum = spectrum_from_distr_array(distr,
                                                 (x_left_index, x_right_index))
            gmax1 = find_gammax(spectrum, xp, 1e-4)
            gmax2 = find_gammax(spectrum, xp, 1e-5)
            gamma_max[layer].append([gmax1, gmax2])

            spectra_upstream[layer].append(spectrum)
    gamma_max = np.array(gamma_max)

    return spectra_downstream, spectra_upstream, gamma_max, xp


def load_electron_density_data(dense, step):
    """
    Load electron density data from movHR file.
    """
    dense[:, :] = 0.0
    print("step_dens =", step)
    step_str = "{:06d}".format(step)
    if step >= 980000:
        step_str = "{:07d}".format(step)

    # File opening
    file_name = 'movHR_'+step_str+'XY.h5'
    file_id = h5py.File(movHR_path/file_name)

    # Group opening
    group_id = file_id["Step#"+step_str]

    # Atributes reading
    Npx = np.int32(group_id.attrs["Npx"])
    iminreseR = np.int32(group_id.attrs["iminreseR"])
    imaxreseR = np.int32(group_id.attrs["imaxreseR"])

    # Data reading
    dense[(iminreseR-1):imaxreseR, :] = group_id["denseresR"]

    # Data filtering
    dense = scp.ndimage.filters.gaussian_filter(dense, 0.5)

    # Set minimum field strength for log plots
    if ILOGD == True:
        dense = dens_log(AFLOORD, DENS0, dense)
    return dense


def load_electron_density_profile(step_min, step_max, step_dens):
    """Calculate y-averaged density profile from movHR file"""
    # Box parameters
    Npx = 10
    Npy = 120
    nFx = np.int(5000)
    nFy = np.int(10)
    mx = np.int(Npx*nFx)
    my = np.int(Npy*nFy)
    # Log plots
    ilog = 1
    afloorde = -0.3
    # Resizing
    iresp = 2
    ires = 2
    mxxp = np.int32(mx/iresp+1)
    myyp = np.int32(my/iresp+1)
    mxxRp = np.int32(mxxp/ires+2)
    myyRp = np.int32(myyp/ires+2)
    dense = np.zeros((mxxRp, myyRp))

    xp = np.array([i for i in range(np.int32(mxxp/ires+2))])*iresp*ires+3.0
    yp = np.array([i for i in range(np.int32(myyp/ires+2))])*iresp*ires+3.0

    dense_profile = []

    for nstep in range(step_min, step_max+1, step_dens):
        print("step_dens =", nstep)
        nstr = str(nstep)
        dense[:, :] = 0.0
        # File opening
        if nstep < 10000:
            file_id = h5py.File("../result/movHR/movHR_00"+nstr+"XY.h5", "r")
        elif nstep < 100000:
            file_id = h5py.File("../result/movHR/movHR_0"+nstr+"XY.h5", "r")
        elif nstep <= 980000:
            file_id = h5py.File("../result/movHR/movHR_"+nstr+"XY.h5", "r")
        elif nstep < 1000000:
            file_id = h5py.File("../result/movHR/movHR_0"+nstr+"XY.h5", "r")
        else:
            file_id = h5py.File("../result/movHR/movHR_"+nstr+"XY.h5", "r")
        # Group opening
        group_id = file_id["Step#"+nstr]
        # Atributes reading
        Npx = np.int32(group_id.attrs["Npx"])
        iminreseR = np.int32(group_id.attrs["iminreseR"])
        imaxreseR = np.int32(group_id.attrs["imaxreseR"])
        # Data reading
        a = group_id["denseresR"]
        dense[(iminreseR-1):imaxreseR, :] = a
        # Data filtering
        dense = scp.ndimage.filters.gaussian_filter(dense, 2.5)
        # Set minimum field strength for log plots
        if ilog == 1:
            dense = dens_log(afloorde, DENS0, dense)
        dense_y_mean = np.mean(dense, axis=1)
        dense_profile.append(dense_y_mean)
    dense_profile = np.array(dense_profile)
    return dense_profile, xp, yp

# Plotting


def plot_dens(dense, densi, xp, yp, x_shock_linear, x_shock, step):
    time_str = "{:04.1f}".format(step/GAMMA)
    step_str = "{:06d}".format(step)
    floor_str = "{:02.1f}".format(AFLOORD)

    fig = plt.figure(figsize=(22.3, 9.68), dpi=100)

    # Axes limits
    if step > STEP_LIMIT:
        delta = x_shock_linear - shock_position_linear(STEP_LIMIT)
        # xlim = (0+delta, 100+delta)
        xlim = (50, 110)
    else:
        xlim = (10, 70)
    ylim = (0, MY/LSI)

    # Adding axes
    # ax2 = fig.add_axes([0.05, 0.05, 0.88, 0.45], xlim=xlim, ylim=ylim)
    # ax1 = fig.add_axes([0.05, 0.46, 0.88, 0.45], sharex=ax2, ylim=ylim)
    # cbaxes = fig.add_axes([0.95, 0.497, 0.015, 0.376])
    ax2 = fig.add_subplot(212, xlim=xlim, ylim=ylim)
    ax1 = fig.add_subplot(211, sharex=ax2, ylim=ylim)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # cbaxes = fig.add_axes([0.95, 0.497, 0.015, 0.376])
    # cbaxes = inset_axes(ax1,
    #                     width="5%",  # width = 50% of parent_bbox width
    #                     height="50%",  # height : 5%
    #                     loc='upper right')
    divider = make_axes_locatable(ax1)
    cbaxes = divider.append_axes('right', size='1.5%', pad=0.1)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='1.5%', pad=0.1)
    # cax.set_frame_on(False)
    cax.set_axis_off()

    # Contour plots
    levels = 21
    clvls = np.linspace(-0.3, 0.7, levels)
    cset1 = ax1.contourf(xp, yp, dense.T, clvls, cmap='jet')
    cset2 = ax2.contourf(xp, yp, densi.T, clvls, cmap='jet')

    # Lines shock position
    # ax1.axvline(x_shock, color='white', linestyle='dashed', linewidth=1)
    # ax2.axvline(x_shock, color='white', linestyle='dashed', linewidth=1)

    # Fontsize
    fs_ticks = 18
    # fs_labels = 18
    fs_clabel = 16
    fs_text = 22

    # Axes tick params
    ax1.set_aspect('equal')
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.yaxis.set_major_locator(MultipleLocator(4))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))

    ax2.set_aspect('equal')
    ax2.xaxis.set_major_locator(MultipleLocator(10))
    ax2.xaxis.set_minor_locator(MultipleLocator(1))
    ax2.yaxis.set_major_locator(MultipleLocator(4))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))

    # Axes labels
    ax1.set_ylabel(r'$y/\lambda_{si}$')
    ax2.set_xlabel(r'$x/\lambda_{si}$')
    ax2.set_ylabel(r'$y/\lambda_{si}$')

    # Text
    ax1.text(xlim[1]-12, ylim[1]-4.8, r'$a) N_e/N_0$', fontsize=fs_text)
    ax2.text(xlim[1]-12, ylim[1]-4.8, r'$b) N_i/N_0$', fontsize=fs_text)
    # ax1.text(xlim[0]+83.2,ylim[1]+2,'nstep='+step_str, fontsize=fs[1])
    # ax1.text(xlim[0]+34,ylim[1]+2,r'$t=$'+time_str+' '+r'$\Omega_{i}^{-1}$',
    #     fontsize=fs[1])
    # ax1.text(xlim[0]+50,ylim[1]+2,'(log, floor='+floor_str+')',
    #     fontsize=fs[1])
    # ax1.text(xlim[0],ylim[1]+1,'electron density', fontsize=fs[1])
    # ax2.text(xlim[0],ylim[1]+1,'ion density', fontsize=fs[1])

    # Colorbar
    cticks = np.linspace(-0.3, 0.7, 11)
    cbar = fig.colorbar(cset1, cax=cbaxes, ax=ax1, ticks=cticks)
    cbar.ax.tick_params(labelsize=fs_clabel, length=5)
    cbar.ax.set_yticklabels(["{:3.1f}".format(i) for i in cticks])
    # cbar.set_label(r'$N/N_0$', fontsize=fs[1])

    return fig


def plot_bfield(bx, by, bz, xp, yp, x_shock_linear, x_shock, step):
    time_str = "{:04.1f}".format(step/GAMMA)
    step_str = "{:07d}".format(step)
    floor_str = "{:02.1f}".format(AFLOORB)

    fig = plt.figure(figsize=(12.31*0.6, 12.3), dpi=300)

    # x axes limits
    if step > STEP_LIMIT:
        delta = x_shock_linear - shock_position_linear(STEP_LIMIT)
        # xlim = (0+delta, 100+delta)
        xlim = (50, 110)
    else:
        xlim = (10, 70)
    ylim = (0, MY/LSI)

    # Adding axes
    ax3 = fig.add_axes([0.05, 0.05, 0.88, 0.3], xlim=xlim, ylim=ylim)
    ax2 = fig.add_axes([0.05, 0.36, 0.88, 0.3], sharex=ax3, ylim=ylim)
    ax1 = fig.add_axes([0.05, 0.67, 0.88, 0.3], sharex=ax3, ylim=ylim)
    cbaxes = fig.add_axes([0.95, 0.679, 0.015, 0.282])

    # Contour plots
    levels = 21
    clvls = np.linspace(-2.6, 2.6, levels)
    cset1 = ax1.contourf(xp, yp, bx.T, clvls, cmap='jet')
    cset2 = ax2.contourf(xp, yp, by.T, clvls, cmap='jet')
    cset3 = ax3.contourf(xp, yp, bz.T, clvls, cmap='jet')

    # Lines shock position
    ax1.axvline(x_shock, color='white', linestyle='dashed', linewidth=1)
    ax2.axvline(x_shock, color='white', linestyle='dashed', linewidth=1)
    ax3.axvline(x_shock, color='white', linestyle='dashed', linewidth=1)

    # Fontsize
    fs_ticks = 18
    fs_labels = 18
    fs_clabel = 14
    fs_text = 22

    # Axes tick params
    ax1.set_aspect('equal')
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.yaxis.set_major_locator(MultipleLocator(4))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax1.tick_params(axis='both', pad=8, labelsize=fs_ticks, length=10)
    ax1.tick_params(axis='both', which='minor', length=5)

    ax2.set_aspect('equal')
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.yaxis.set_major_locator(MultipleLocator(4))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.tick_params(axis='both', pad=8, labelsize=fs_ticks, length=10)
    ax2.tick_params(axis='both', which='minor', length=5)

    ax3.set_aspect('equal')
    ax3.xaxis.set_major_locator(MultipleLocator(10))
    ax3.xaxis.set_minor_locator(MultipleLocator(1))
    ax3.yaxis.set_major_locator(MultipleLocator(4))
    ax3.yaxis.set_minor_locator(MultipleLocator(1))
    ax3.tick_params(axis='both', pad=8, labelsize=fs_ticks, length=10)
    ax3.tick_params(axis='both', which='minor', length=5)

    # Axes labels
    ax1.set_ylabel(r'$y/\lambda_{si}$', fontsize=fs_labels)
    ax2.set_ylabel(r'$y/\lambda_{si}$', fontsize=fs_labels)
    ax3.set_xlabel(r'$x/\lambda_{si}$', fontsize=fs_labels)
    ax3.set_ylabel(r'$y/\lambda_{si}$', fontsize=fs_labels)

    # Text
    ax1.text(xlim[1]-12, ylim[1]-4.8, r'$a) B_x/B_0$', fontsize=fs_text)
    ax2.text(xlim[1]-12, ylim[1]-4.8, r'$b) B_y/B_0$', fontsize=fs_text)
    ax3.text(xlim[1]-12, ylim[1]-4.8, r'$c) B_z/B_0$', fontsize=fs_text)
    # ax1.text(xlim[0]+83.2,14,'nstep='+step_str, fontsize=fs[1])
    # ax1.text(xlim[0]+34,14,r'$t=$'+time_str+' '+r'$\Omega_{i}^{-1}$',
    # fontsize=fs[1])
    # ax1.text(xlim[0]+50,14,'(log, floor='+floor_str+')',
    # fontsize=fs[1])

    # Colorbar
    cticks = np.linspace(-2.6, 2.6, 11)
    cbar = fig.colorbar(cset1, cax=cbaxes, ax=ax1, ticks=cticks)
    cbar.ax.tick_params(labelsize=fs_clabel, length=5)
    cbar.ax.set_yticklabels(["{:3.1f}".format(i) for i in cticks])
    # cbar.set_label(r'$B/B_0$', fontsize=fs[1])

    return fig


def plot_efield(ex, ey, ez, xp, yp, x_shock_linear, x_shock, step):
    time_str = "{:04.1f}".format(step/GAMMA)
    step_str = "{:07d}".format(step)
    floor_str = "{:02.1f}".format(AFLOORB)

    fig = plt.figure(figsize=(12.31*0.6, 12.3), dpi=300)

    # x axes limits
    if step > STEP_LIMIT:
        delta = x_shock_linear - shock_position_linear(STEP_LIMIT)
        # xlim = (0+delta, 100+delta)
        xlim = (50, 110)
    else:
        xlim = (10, 70)
    ylim = (0, MY/LSI)

    # Adding axes
    ax3 = fig.add_axes([0.05, 0.05, 0.88, 0.3], xlim=xlim, ylim=ylim)
    ax2 = fig.add_axes([0.05, 0.36, 0.88, 0.3], sharex=ax3, ylim=ylim)
    ax1 = fig.add_axes([0.05, 0.67, 0.88, 0.3], sharex=ax3, ylim=ylim)
    cbaxes = fig.add_axes([0.95, 0.679, 0.015, 0.282])

    # Contour plots
    levels = 21
    clvls = np.linspace(-1.4, 1.4, levels)
    cset1 = ax1.contourf(xp, yp, ex.T, clvls, cmap='jet')
    cset2 = ax2.contourf(xp, yp, ey.T, clvls, cmap='jet')
    cset3 = ax3.contourf(xp, yp, ez.T, clvls, cmap='jet')

    # Lines shock position
    ax1.axvline(x_shock, color='white', linestyle='dashed', linewidth=1)
    ax2.axvline(x_shock, color='white', linestyle='dashed', linewidth=1)
    ax3.axvline(x_shock, color='white', linestyle='dashed', linewidth=1)

    # Fontsize
    fs_ticks = 18
    fs_labels = 18
    fs_clabel = 14
    fs_text = 22

    # Axes tick params
    ax1.set_aspect('equal')
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.yaxis.set_major_locator(MultipleLocator(4))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax1.tick_params(axis='both', pad=8, labelsize=fs_ticks, length=10)
    ax1.tick_params(axis='both', which='minor', length=5)

    ax2.set_aspect('equal')
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.yaxis.set_major_locator(MultipleLocator(4))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.tick_params(axis='both', pad=8, labelsize=fs_ticks, length=10)
    ax2.tick_params(axis='both', which='minor', length=5)

    ax3.set_aspect('equal')
    ax3.xaxis.set_major_locator(MultipleLocator(10))
    ax3.xaxis.set_minor_locator(MultipleLocator(1))
    ax3.yaxis.set_major_locator(MultipleLocator(4))
    ax3.yaxis.set_minor_locator(MultipleLocator(1))
    ax3.tick_params(axis='both', pad=8, labelsize=fs_ticks, length=10)
    ax3.tick_params(axis='both', which='minor', length=5)

    # Axes labels
    ax1.set_ylabel(r'$y/\lambda_{si}$', fontsize=fs_labels)
    ax2.set_ylabel(r'$y/\lambda_{si}$', fontsize=fs_labels)
    ax3.set_xlabel(r'$x/\lambda_{si}$', fontsize=fs_labels)
    ax3.set_ylabel(r'$y/\lambda_{si}$', fontsize=fs_labels)

    # Text
    ax1.text(xlim[1]-15, ylim[1]-4.8, r'$a) E_x/(B_0c)$', fontsize=fs_text)
    ax2.text(xlim[1]-15, ylim[1]-4.8, r'$b) E_y/(B_0c)$', fontsize=fs_text)
    ax3.text(xlim[1]-15, ylim[1]-4.8, r'$c) E_z/(B_0c)$', fontsize=fs_text)
    # ax1.text(xlim[0]+83.2,14,'nstep='+step_str, fontsize=fs[1])
    # ax1.text(xlim[0]+34,14,r'$t=$'+time_str+' '+r'$\Omega_{i}^{-1}$',
    # fontsize=fs[1])
    # ax1.text(xlim[0]+50,14,'(log, floor='+floor_str+')',
    # fontsize=fs[1])

    # Colorbar
    cticks = np.linspace(-1.4, 1.4, 11)
    cbar = fig.colorbar(cset1, cax=cbaxes, ax=ax1, ticks=cticks)
    cbar.ax.tick_params(labelsize=fs_clabel, length=5)
    cbar.ax.set_yticklabels(["{:3.1f}".format(i) for i in cticks])
    # cbar.set_label(r'$B/B_0$', fontsize=fs[1])

    return fig


def plot_phase(px, py, pz, xp, yp, x_shock_linear, x_shock, step):
    time_str = "{:04.1f}".format(step/GAMMA)
    step_str = "{:07d}".format(step)

    fig = plt.figure(figsize=(12.31*0.8, 12.3*0.6), dpi=300)

    # x axes limits
    if step > STEP_LIMIT:
        delta = x_shock_linear - shock_position_linear(STEP_LIMIT)
        # xlim = (0+delta, 250+delta)
        xlim = (115, 205)
    else:
        xlim = (0, 80)
    ymax = np.max(yp)
    ylim = (-ymax, ymax)

    # Adding axes
    ax3 = fig.add_axes([0.05, 0.05, 0.88, 0.3], xlim=xlim, ylim=ylim)
    ax2 = fig.add_axes([0.05, 0.36, 0.88, 0.3], sharex=ax3, ylim=ylim)
    ax1 = fig.add_axes([0.05, 0.67, 0.88, 0.3], sharex=ax3, ylim=ylim)
    cbaxes = fig.add_axes([0.95, 0.67, 0.015, 0.3])

    # Contour plots
    levels = 37
    clvls = np.linspace(0, 3.6, levels)
    cset1 = ax1.contourf(xp, yp, px.T, clvls, cmap='jet')
    cset2 = ax2.contourf(xp, yp, py.T, clvls, cmap='jet')
    cset3 = ax3.contourf(xp, yp, pz.T, clvls, cmap='jet')

    # Lines shock position
    ax1.axvline(x_shock, color='black', linestyle='dashed', linewidth=1)
    ax2.axvline(x_shock, color='black', linestyle='dashed', linewidth=1)
    ax3.axvline(x_shock, color='black', linestyle='dashed', linewidth=1)

    # Fontsize
    fs_ticks = 18
    fs_labels = 18
    fs_clabel = 14
    fs_text = 22

    # Axes tick params
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.yaxis.set_major_locator(MultipleLocator(2))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.4))
    ax1.tick_params(axis='both', pad=8, labelsize=fs_ticks, length=10)
    ax1.tick_params(axis='both', which='minor', length=5)

    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.yaxis.set_major_locator(MultipleLocator(2))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.4))
    ax2.tick_params(axis='both', pad=8, labelsize=fs_ticks, length=10)
    ax2.tick_params(axis='both', which='minor', length=5)

    ax3.xaxis.set_major_locator(MultipleLocator(10))
    ax3.xaxis.set_minor_locator(MultipleLocator(1))
    ax3.yaxis.set_major_locator(MultipleLocator(2))
    ax3.yaxis.set_minor_locator(MultipleLocator(0.4))
    ax3.tick_params(axis='both', pad=8, labelsize=fs_ticks, length=10)
    ax3.tick_params(axis='both', which='minor', length=5)

    # Axes labels
    ax1.set_ylabel(r'$p_{x}/(mc)$', fontsize=fs_labels)
    ax2.set_ylabel(r'$p_{y}/(mc)$', fontsize=fs_labels)
    ax3.set_xlabel(r'$x/\lambda_{si}$', fontsize=fs_labels)
    ax3.set_ylabel(r'$p_{z}/(mc)$', fontsize=fs_labels)

    # Text
    # ax1.text(xlim[0]+208.5, 5.5, 'nstep='+step_str, fontsize=fs[1])
    # ax1.text(xlim[0]+85, 5.5, r'$t=$'+time_str+' '+r'$\Omega_{i}^{-1}$',
    #          fontsize=fs[1])
    # ax1.text(xlim[0]+122, 5.5, '(log)',
    #          fontsize=fs[1])

    # Colorbar
    cticks = np.linspace(0, 3.6, 7)
    cbar = fig.colorbar(cset1, cax=cbaxes, ax=ax1, ticks=cticks)
    cbar.ax.tick_params(labelsize=fs_clabel, length=5)
    cbar.ax.set_yticklabels(["{:3.1f}".format(i) for i in cticks])
    # cbar.set_label("electrons", fontsize=fs[1])

    return fig


def plot_spectra_downstream(
        layers_number, x_left_downstream, layers_thikness, layers_distance,
        spectra, xp, step_min, step_max, step_spectra):
    norm = mpl.colors.Normalize(vmin=(step_min/GAMMA), vmax=(step_max/GAMMA))
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap="jet")
    ticks = np.arange(0, 72, 2)
    pltnumber = np.int((step_max-step_min)/step_spectra+1)
    inorm = -1
    my_map = plt.cm.get_cmap("jet")
    colors = my_map(np.linspace(0, 1, pltnumber, endpoint=True))

    xlim = (1e-2, 100)
    ylim = (1e-6, 3)
    for layer in range(layers_number):
        fig = plt.figure(figsize=(8.5, 6), dpi=300)
        ax = fig.add_axes([0.1, 0.1, 0.81, 0.85], xlim=xlim, ylim=ylim)
        cbaxes = fig.add_axes([0.92, 0.1, 0.03, 0.81])
        for i, spectrum in enumerate(spectra[layer]):
            ax.loglog(xp, spectrum, linestyle="-", color=colors[i],
                      linewidth=1.0)

        ax.set_ylabel(r'$(\gamma-1)(dN/d\gamma)/N$', fontsize=14)
        ax.set_xlabel(r'$\gamma-1$', fontsize=14)
        ax.text(xlim[0]+3e-3, ylim[0]+5e-6, 'downstream', fontsize=16)

        x_range_string_r = x_left_downstream+layer*layers_distance/LSI
        x_range_string_l = (x_left_downstream-layer*layers_distance/LSI
                            - layers_thikness/LSI)
        ax.text(xlim[0]+3e-3, ylim[0]+1e-6, r'$x_{sh}-x=(%d-%d)\lambda_{si}$'
                % (x_range_string_l, x_range_string_r), fontsize=16)

        cbar = fig.colorbar(cmap, cax=cbaxes, ticks=ticks)
        cbar.ax.set_title(r'$t \/ \Omega_i$', fontsize=12)

        # Maxwellian fit
        maxwellian_curve = xp*maxwell_fit(0.03, 1.0, xp, spectra[layer][-1])
        ax.plot(xp, maxwellian_curve, linestyle='--', color='grey')

        # Power fit
        power_index = np.int(pltnumber-10)
        power_curve, p = power_fit(0.2, 2.0, xp, spectra[layer][power_index])
        ax.plot(xp[290:], power_curve[290:], linestyle='--', color='black')
        ax.text(10, 1e-3, r'$p=$'+"{:.2f}".format(p), fontsize=16)

        fig_name = 'Downstream'+spectra_name+str("%d" % layer)+'.png'
        fig.savefig(spectra_plot_path+fig_name)
        plt.close(fig)


def plot_spectra_upstream(
        layers_number, x_left_upstream, layers_thikness, layers_distance,
        spectra, gamma_max, xp, step_min, step_max, step_spectra):
    norm = mpl.colors.Normalize(vmin=(step_min/GAMMA), vmax=(step_max/GAMMA))
    time = np.arange(step_min, step_max+1, step_spectra)/GAMMA
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap="jet")
    ticks = np.arange(0, 72, 2)
    pltnumber = np.int((step_max-step_min)/step_spectra+1)
    inorm = -1
    my_map = plt.cm.get_cmap("jet")
    colors = my_map(np.linspace(0, 1, pltnumber, endpoint=True))

    xlim = (1e-2, 100)
    ylim = (1e-6, 3)
    for layer in range(layers_number):
        fig = plt.figure(figsize=(8.5, 6), dpi=300)
        ax = fig.add_axes([0.1, 0.1, 0.81, 0.85], xlim=xlim, ylim=ylim)
        ax_gamma = fig.add_axes([0.17, 0.33, 0.3, 0.3])
        cbaxes = fig.add_axes([0.92, 0.1, 0.03, 0.81])
        for i, spectrum in enumerate(spectra[layer]):
            ax.loglog(xp, spectrum, linestyle="-", color=colors[i],
                      linewidth=1.0)

        ax.set_ylabel(r'$(\gamma-1)(dN/d\gamma)/N$', fontsize=14)
        ax.set_xlabel(r'$\gamma-1$', fontsize=14)
        ax.text(xlim[0]+3e-3, ylim[0]+5e-6, 'upstream', fontsize=16)

        x_range_string_l = x_left_upstream+layer*layers_distance/LSI
        x_range_string_r = x_left_upstream+layer*layers_distance/LSI+layers_thikness/LSI
        ax.text(xlim[0]+3e-3, ylim[0]+1e-6, r'$x_{sh}-x=(%d-%d)\lambda_{si}$' %
                (x_range_string_l, x_range_string_r), fontsize=16)
        cbar = fig.colorbar(cmap, cax=cbaxes, ticks=ticks)
        cbar.ax.set_title(r'$t \/ \Omega_i$', fontsize=12)

        # # Maxwellian fit
        maxwellian_curve = xp*maxwell_fit(0.03, 1.0, xp, spectra[layer][-1])
        ax.plot(xp, maxwellian_curve, linestyle='--', color='grey')

        # Power fit
        ax_gamma.plot(time, gamma_max[layer, :, 0], label="cutoff 1e-4")
        ax_gamma.plot(time, gamma_max[layer, :, 1], label="cutoff 1e-5")
        ax_gamma.legend(loc='upper left', fontsize=8)
        ax_gamma.set_xlabel(r'$t \/ \Omega_i$', fontsize=10)
        ax_gamma.set_ylabel(r'$\gamma_{max}$', fontsize=10)
        plt.setp(ax_gamma.get_xticklabels(), fontsize=8)
        plt.setp(ax_gamma.get_yticklabels(), fontsize=8)

        fig_name = 'Upstream'+spectra_name+str("%d" % layer)+'.png'
        fig.savefig(spectra_plot_path+fig_name)
        plt.close(fig)
# ----------
# CLASSES
# ----------


class AsyncPlotter():
    """
    Asynchronus plotting class adapted from GitHub astrofrog/async_plotting.py.
    """

    def __init__(self, processes=mp.cpu_count()):

        self.manager = mp.Manager()
        self.nc = self.manager.Value('i', 0)
        self.pids = []
        self.processes = processes

    def async_plotter(self, nc, fig, filename, processes):
        while nc.value >= processes:
            time.sleep(0.1)
        nc.value += 1
        print("Plotting " + filename)
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        nc.value -= 1

    def save(self, fig, filename):
        p = mp.Process(target=self.async_plotter,
                       args=(self.nc, fig, filename, self.processes))
        p.start()
        self.pids.append(p)

    def join(self):
        for p in self.pids:
            p.join()
#
# def colorline(x, y, z=None, cmap='copper', norm=plt.Normalize(0.0, 1.0),
#         linewidth=3, alpha=1.0):
#     """Adapted from https://stackoverflow.com/questions/36074455/python-matplotlib-with-a-line-color-gradient-and-colorbar
#
#     Plot a colored line with coordinates x and y
#     Optionally specify colors in the array z
#     Optionally specify a colormap, a norm function and a line width
#     """
#     # Default colors equally spaced on [0,1]:
#     if z is None:
#         z = np.linspace(0.0, 1.0, len(x))
#     # Special case if a single number:
#     # to check for numerical input -- this is a hack
#     if not hasattr(z, "__iter__"):
#         z = np.array([z])
#     z = np.asarray(z)
#     segments = make_segments(x, y)
#     lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
#                               linewidth=linewidth, alpha=alpha)
#     return lc
#
# def make_segments(x, y):
#     """Create list of line segments from x and y coordinates, in the
#     correct format for LineCollection: an array of the form numlines
#     x (points per line) x 2 (x and y) array.
#     """
#     points = np.array([x, y]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     return segments
#
# # Simulation steps
# step_min = np.int(70000)
# step_max = np.int(480000)
# step_dens = np.int(1000)
# step_trace = np.int(20)
#
# # Frame specification
# HT = True
# UP = False
#
# # dense_profile, x_dens, y_dens = load_electron_density_profile(step_min,
# #     step_max,step_dens)
# # time_dens = [i/GAMMA for i in range(step_min,step_max+1,step_dens)]
#
# # Main loop
# for i, particle in enumerate(trej_names):
#     print(particle)
#     (step, gm, x, y, px, py, pz,
#         ex, ey, ez, bx, by, bz) = load_trej_data(particle, step_min, step_max)
#     time = step/GAMMA
#
#     # Quantities calculation
#     vx = px/gm
#     vy = py/gm
#     vz = pz/gm
#
#     vx_up = (vx-V_JET)/(1.0-vx*V_JET/C**2)
#     vy_up = vy*np.sqrt(1.0-(V_JET/C)**2)/(1.0-vx*V_JET/C**2)
#     vz_up = vz*np.sqrt(1.0-(V_JET/C)**2)/(1.0-vx*V_JET/C**2)
#     v_up = np.sqrt(vx_up**2+vy_up**2+vz_up**2)
#     gm_up = C/np.sqrt(C**2-v_up**2)
#
#     if HT == True:
#         cosa = (vx_up*B0X+vy_up*B0Y+vz_up*B0Z)/(v_up*B0)
#         v_up_parl = v_up*cosa
#         v_up_perp = v_up*np.sqrt(1.0-cosa**2)
#         v_ht_parl = (v_up_parl-VT)/(1.0-v_up_parl*VT/C**2)
#         v_ht_perp = v_up_perp/(1.0-v_up_parl*VT/C**2)/GM_T
#
#         pparl = v_ht_parl/C*gm_up*GM_T*(1.0-v_up_parl*VT/C**2)
#         pperp = v_ht_perp/C*gm_up*GM_T*(1.0-v_up_parl*VT/C**2)
#     else:
#         cosa = ((px*bx+py*by+pz*bz)
#             /np.sqrt((px**2+py**2+pz**2)*(bx**2+by**2+bz**2)))
#
#         pparl = np.sqrt(gm**2-1.0)*cosa
#         pperp = np.sqrt(gm**2-1.0)*np.sqrt(1.0-cosa**2)
#
#     if UP == True:
#         gam = gam_up-1.0
#         ex_up = ex*GAM_JET
#         ey_up = (ey-V_JET*bz)*GAM_JET
#         ez_up = (ez+V_JET*by)*GAM_JET
#
#         qve = QE*(ex_up*vx_up+ey_up*vy_up+ez_up*vz_up)*gamma/(ME*C**2)
#         qvex = QE*ex_up*vx_up*gamma/(ME*C**2)
#         qvey = QE*ey_up*vy_up*gamma/(ME*C**2)
#         qvez = QE*ez_up*vz_up*gamma/(ME*C**2)
#     else:
#         gam = gm-1.0
#         qve = QE*(ex*vx + ey*vy + ez*vz)*GAMMA/(ME*C**2)
#         qvex = QE*ex*vx*GAMMA/(ME*C**2)
#         qvey = QE*ey*vy*GAMMA/(ME*C**2)
#         qvez = QE*ez*vz*GAMMA/(ME*C**2)
#
#     gamparl = gam*cosa**2
#     gamperp = gam*(1.0 - cosa**2)
#
#     acc = [0.0]
#     # sda = [0.0]
#     sda = [gam[0]]
#     for i, val in enumerate(gam[:-1]):
#         acc.append((gam[i+1]-gam[i])/step_trace*GAMMA)
#         sda.append(sda[i]+qvez[i+1]/GAMMA*step_trace)
#
#     # Filtering
#     gam = scp.ndimage.filters.gaussian_filter1d(gam, 50.0)
#     gamparl = scp.ndimage.filters.gaussian_filter1d(gamparl, 50.0)
#     gamperp = scp.ndimage.filters.gaussian_filter1d(gamperp, 50.0)
#     sda = scp.ndimage.filters.gaussian_filter1d(sda, 50.0)
#     qvez = scp.ndimage.filters.gaussian_filter1d(qvez, 50.0)
#     acc = scp.ndimage.filters.gaussian_filter1d(acc, 50.0)
#
#     # Momentum line  interpolation
#     tck,u = splprep([pparl,pperp],s=0.0)
#     x_i,y_i = splev(u,tck)
#
#     # PLOT SECTION
#     fig = plt.figure(figsize=(10,16), dpi=200)
#     fs = [18,22,16,12]
#
#     # Limits and colormap levels
#     xmin_trace = time[0]
#     xmax_trace = time[-1]
#     xlim_trace = (xmin_trace,xmax_trace)
#     ymin_trace = np.round(np.min(x/LSI)-10,-1)
#     ymax_trace = np.round(np.max(x/LSI)+10,-1)
#     ylim_trace = (ymin_trace, ymax_trace)
#
#     ymax_qvez = np.round(np.max(qvez)+5,-1)
#     ylim_qvez = (-ymax_qvez,ymax_qvez)
#
#     xmax_mom = np.round(np.max(pparl)+2,0)
#     ymax_mom = np.round(np.max(pperp)+2,0)
#     xlim_mom = (-xmax_mom,xmax_mom)
#     ylim_mom = (0.0,ymax_mom)
#
#     levels = 21
#     clvls = np.linspace(-0.3,0.7,levels)
#
#     # Axes specification
#     ax4 = fig.add_axes([0.1, 0.055, 0.85, 0.19],xlim=xlim_mom, ylim=ylim_mom)
#     ax3 = fig.add_axes([0.1, 0.30, 0.85, 0.22],xlim=xlim_trace, ylim=ylim_trace)
#     ax2 = fig.add_axes([0.1, 0.53, 0.85, 0.22],ylim=ylim_qvez, sharex=ax3)
#     ax1 = fig.add_axes([0.1, 0.76, 0.85, 0.22],sharex=ax3)
#
#     # Kinetic energy plot
#     ax1.plot(time, gam, color='black', label=r'$(\gamma-1)_{simulation}$')
#     ax1.plot(time, gamparl, color='blue', label=r'$(\gamma-1)_{\parallel}$')
#     ax1.plot(time, gamperp, color='green', label=r'$(\gamma-1)_{\perp}$')
#     ax1.plot(time, sda, color='red', label=r'$(\gamma-1)_{drift}$')
#     plt.setp(ax1.get_xticklabels(), visible=False)
#     plt.setp(ax1.get_yticklabels(), fontsize=fs[0])
#     ax1.set_ylabel(r'$\gamma-1$', fontsize=fs[1])
#     ax1.yaxis.set_major_locator(MultipleLocator(5))
#     ax1.yaxis.set_minor_locator(MultipleLocator(1))
#     ax1.legend(loc='best', fontsize=fs[3])
#
#     # Qvez and acc plot
#     ax2.plot(time, qvez, color='red', label=r'$q(v_z \cdot E_z)/(mc^2)$')
#     ax2.plot(time, acc, color='black', label=r'$\Delta(\gamma-1)$')
#     ax2.axhline(0.0, color='black', linestyle='dashed')
#     plt.setp(ax2.get_xticklabels(), visible=False)
#     plt.setp(ax2.get_yticklabels(), fontsize=fs[0])
#     ax2.set_ylabel(r'$d\gamma/d(t\Omega_i)$', fontsize=fs[1])
#     ax2.yaxis.set_major_locator(MultipleLocator(4))
#     ax2.yaxis.set_minor_locator(MultipleLocator(0.5))
#     ax2.legend(loc='best', fontsize=fs[3])
#
#     # Trace on y-averaged density plot
#     ax3.plot(time, x/LSI, color='black', label=r'$$')
#     # ax3.contourf(time_dens,x_dens/LSI, dense_profile.T, clvls, cmap='jet')
#     plt.setp(ax3.get_xticklabels(), fontsize=fs[0])
#     plt.setp(ax3.get_yticklabels(), fontsize=fs[0])
#     ax3.set_xlabel(r'$t\Omega_{i}$', fontsize=fs[1])
#     ax3.set_ylabel(r'$x/\lambda_{si}$', fontsize=fs[1])
#     ax3.xaxis.set_major_locator(MultipleLocator(1))
#     ax3.xaxis.set_minor_locator(MultipleLocator(0.1))
#     ax3.yaxis.set_major_locator(MultipleLocator(20))
#     ax3.yaxis.set_minor_locator(MultipleLocator(2))
#
#     # Momentum plot
#     lc = colorline(x_i,y_i, cmap='jet', linewidth=0.5)
#     ax4.add_collection(lc)
#     # ax4.colorbar(lc)
#     sc = ax4.scatter(pparl, pperp, s=0.2, c=time, cmap='jet')
#     # ax4.plot(x_i, y_i, linewidth=0.1)
#     ax4.axvline(0.0, color='black', linestyle='dashed')
#     ax4.set_aspect('equal')
#     plt.setp(ax4.get_xticklabels(), fontsize=fs[0])
#     plt.setp(ax4.get_yticklabels(), fontsize=fs[0])
#     if HT == True:
#         ax4.set_xlabel(r'$p_{\parallel}/(mc) (HT)$', fontsize=fs[1])
#         ax4.set_ylabel(r'$p_{\perp}/(mc) (HT)$', fontsize=fs[1])
#     elif UP == True:
#         ax4.set_xlabel(r'$p_{\parallel}/(mc) (UP)$', fontsize=fs[1])
#         ax4.set_ylabel(r'$p_{\perp}/(mc) (UP)$', fontsize=fs[1])
#     else:
#         ax4.set_xlabel(r'$p_{\parallel}/(mc)$', fontsize=fs[1])
#         ax4.set_ylabel(r'$p_{\perp}/(mc)$', fontsize=fs[1])
#     ax4.xaxis.set_major_locator(MultipleLocator(5))
#     ax4.xaxis.set_minor_locator(MultipleLocator(1))
#     ax4.yaxis.set_major_locator(MultipleLocator(5))
#     ax4.yaxis.set_minor_locator(MultipleLocator(1))
#
#     # Colormap
#     cbaxes = inset_axes(ax4, width="40%", height="4%", loc=2)
#     cbar = plt.colorbar(sc, cax=cbaxes, orientation='horizontal')
#     cbar.set_label(r'$t\Omega_{i}$', fontsize=fs[2])
#
#     plt.savefig("./tracing/plot_momentum/"+particle+".png")
#     plt.close(fig)
