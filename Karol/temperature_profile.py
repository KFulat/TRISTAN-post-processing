"""
Python script producing density plots from movHR files.
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import scipy as scp
import h5py
from scipy import ndimage
#import matplotlib.animation as animation
import time
#from numba import jit
import matplotlib
matplotlib.use('Agg')
import multiprocessing as mp
#import matplotlib.gridspec as gridspec
#from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

def dens_log(floor, norm, tab):
    """
    Make logscale density array.
    """
    tab = tab/norm
    index_sign = np.sign(tab)
    tab_mask = ma.masked_equal(tab, 0.0)
    tabb = np.log10(np.abs(tab_mask))
    tabb_mask = ma.masked_less_equal(tabb, floor)
    tabb = tabb_mask.filled(0)
    tabb = tabb*index_sign
    return tabb

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

def shock_position_index(dens):
    """
    Find index of density jump in dnsity array (from right).
    """
    tab = np.flip(dens[150])
    for i, val in enumerate(tab):
        if val > 0.4:
            index = i
            break
    return index

def shock_position_linear(nstep):
    """
    Shock possition assuming that velocity is constant,
    coefficients from regression equation.
    """
    a = 0.00025
    b = 6.64
    return a*nstep+b

# Box parameters
NPX = np.int(17)
NPY = np.int(120)
NFX = np.int(5000)
NFY = np.int(10)
MX = np.int(NPX*NFX)
MY = np.int(NFY*NFY)

DELTA = np.int(10)
LSI = 150.0

def load_temperature_data(species, step):
    temp_folder_path = './result/temp/'
    step_str = "{:06d}".format(step)

    xPS = np.int(NPX*NFX/DELTA)
    yPS = NPY

    temp = np.loadtxt(temp_folder_path+step_str+'_T_'+species+'_R')
    temp_x = np.loadtxt(temp_folder_path+step_str+'_Tx_'+species+'_R')
    temp_y = np.loadtxt(temp_folder_path+step_str+'_Ty_'+species+'_R')
    temp_z = np.loadtxt(temp_folder_path+step_str+'_Tz_'+species+'_R')

    temp = np.reshape(temp, (xPS,yPS))
    temp_x = np.reshape(temp_x, (xPS,yPS))
    temp_y = np.reshape(temp_y, (xPS,yPS))
    temp_z = np.reshape(temp_z, (xPS,yPS))

    return temp, temp_x, temp_y, temp_z

print("Loading and averaging temperature data.")
temp, temp_x, temp_y, temp_z = load_temperature_data('ele', 640000)
temp_profile = np.mean(temp, axis=1)
temp_x_profile = np.mean(temp_x, axis=1)
temp_y_profile = np.mean(temp_y, axis=1)
temp_z_profile = np.mean(temp_z, axis=1)
temp_profile = scp.ndimage.filters.gaussian_filter1d(temp_profile, 25.0)
temp_x_profile = scp.ndimage.filters.gaussian_filter1d(temp_x_profile, 25.0)
temp_y_profile = scp.ndimage.filters.gaussian_filter1d(temp_y_profile, 25.0)
temp_z_profile = scp.ndimage.filters.gaussian_filter1d(temp_z_profile, 25.0)

x_temp = np.linspace(0,MX,temp_profile.shape[0])/LSI

print("Plotting.")
fig = plt.figure(figsize=(10,8), dpi=200)
xlim = (0,300)
ax1 = fig.add_subplot(111, xlim=xlim)
ax1.plot(x_temp, temp_profile)
ax1.plot(x_temp, temp_x_profile)
ax1.plot(x_temp, temp_y_profile)
ax1.plot(x_temp, temp_z_profile)

plt.savefig("./temp_test.png")
plt.close(fig)
