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

#--------------------------------
# SIMULATION PARAMETERS AND SETUP
#--------------------------------

# Box parameters
Npx = 18
Npy = 120
nFx = np.int(5000)
nFy = np.int(10)
mx = np.int(Npx*nFx)
my = np.int(Npy*nFy)

# Plasma parameters
dens0 = 20.0
gamma = 16329.956
skin = 100.0

# Time steps range
nstepmin = np.int32(70000)
nstepmax = np.int32(75000)
nstepstep = np.int32(1000)
nsteptr = np.int32(20)

# Log plots
ilog = 1
afloorde = -0.3
afloordi = -0.3

# Resizing and B-field arrays
iresp=2
ires = 2

mxxp = np.int32(mx/iresp+1)
myyp = np.int32(my/iresp+1)
mxxRp = np.int32(mxxp/ires+2)
myyRp = np.int32(myyp/ires+2)

dense=np.zeros((mxxRp,myyRp))
densi=np.zeros((mxxRp,myyRp))

# Assync plot initialization
allplots = AsyncPlotter()

f = open("shock_position.txt", "w")
    ####### TRACING PARAT
file_tr = np.loadtxt("./result/trace/trej_00001", skiprows=1, usecols=(0,2,3))
# print(file_tr[0:5,1:3])
# a = np.where(file_tr[:,0] <= 70100)
# print(file_tr[a,1:3])
# print(file_tr[np.where(file_tr[:][0] < 90000)])


#--------------------------
# MAIN LOOP OVER TIME STEPS
#--------------------------

for nstep in range(nstepmin, nstepmax+1, nstepstep):
    print("nstep =", nstep)
    nstr = str(nstep)
    dense[:,:] = 0.0
    densi[:,:] = 0.0

    # File opening
    if nstep < 10000:
        file_id=h5py.File("./result/movHR/movHR_00"+nstr+"XY.h5", "r")
    elif nstep < 100000:
        file_id=h5py.File("./result/movHR/movHR_0"+nstr+"XY.h5", "r")
    elif nstep <= 980000:
        file_id=h5py.File("./result/movHR/movHR_"+nstr+"XY.h5", "r")
    elif nstep < 1000000:
        file_id=h5py.File("./result/movHR/movHR_0"+nstr+"XY.h5", "r")
    else:
        file_id=h5py.File("./result/movHR/movHR_"+nstr+"XY.h5", "r")

    # Group opening
    group_id = file_id["Step#"+nstr]

    # Atributes reading
    Npx = np.int32(group_id.attrs["Npx"])
    iminreseR = np.int32(group_id.attrs["iminreseR"])
    imaxreseR = np.int32(group_id.attrs["imaxreseR"])
    iminresiR = np.int32(group_id.attrs["iminresiR"])
    imaxresiR = np.int32(group_id.attrs["imaxresiR"])

    # Data reading
    a = group_id["denseresR"]
    dense[(iminreseR-1):imaxreseR,:] = a

    a = group_id["densiresR"]
    densi[(iminresiR-1):imaxresiR,:] = a

    # Data filtering
    dense = scp.ndimage.filters.gaussian_filter(dense, 0.5)
    densi = scp.ndimage.filters.gaussian_filter(densi, 0.5)

    # Set minimum field strength for log plots
    if ilog == 1:
        dense = dens_log(afloorde, dens0, dense)
        densi = dens_log(afloordi, dens0, densi)

    # Plot
    tgam = nstep*1.0/gamma
    tgam = str("{:04.1f}".format(tgam))
    ntitle = str(nstep)
    floorstr = str("{:03.1f}".format(afloorde))

    xp = np.array([i for i in range(np.int32(mxxp/ires+2))])*iresp*ires+3.0
    yp = np.array([i for i in range(np.int32(myyp/ires+2))])*iresp*ires+3.0

    DENSE = np.transpose(dense)
    DENSI = np.transpose(densi)

    xshock1 = shock_position_linear(nstep)
    xshock2 = (1-shock_position_index(DENSE)/DENSE.shape[1])*np.max(xp/skin)
    f.write("%d %.2f\n" % (nstep, xshock2))

    # Use trace
    a = np.where(file_tr[:,0] <= (nstep+nstepstep))
    tab = file_tr[a]
    a = np.where(tab[:,0] > nstep-2*nstepstep)
    trace = tab[a,1:3]
    # trace = file_tr[a,1:3]
    trace = np.transpose(trace)
    trace = np.reshape(trace,(2, trace.shape[1]))
    tracex = trace[0]/skin
    tracey = trace[1]/skin
    # print(tracex.shape)
    # s = np.argsort(tracey)
    # tracex = tracex[s]
    # tracey = tracey[s]
    # print(tracex.shape)
    # a = np.array([1,3,2])
    # b = np.array([4,5,6])
    # s = np.argsort(a)
    # print(a[s].shape)
    # print(b[s].shape)
    # exit()
    # print(tracex)
    # print(tracey)

    # PLOT
    fig = plt.figure(figsize=(15,2), dpi=200)
    fs = [12,18,20]
    levels = 21
    clvls = np.linspace(-0.3,0.7,levels)
    # yticks = [0,2,4,6,8]

    # x axes limits
    steplim = 216000
    if nstep > steplim:
        delta = xshock1 - shock_position_linear(steplim)
        xlim = (0+delta, 80+delta)
    else:
        xlim = (0,80)

    ax1 = fig.add_subplot(1,1,1, xlim=xlim)
    # gs1 = gridspec.GridSpec(2, 1)
    # ax2 = fig.add_subplot(gs1[1],xlim=(0,140))
    # ax1 = fig.add_subplot(gs1[0],sharex=ax2)
    # gs1.tight_layout(fig,rect=[0, 0.1, 1, 0.9])

    # cset1 = ax1.pcolormesh(xp/skin,yp/skin,DENSE, cmap='jet', vmin=afloorde, vmax=0.7)
    cset1 = ax1.contourf(xp/skin,yp/skin,DENSE, clvls, cmap='jet')
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), fontsize=fs[0])
    # ax1.set_yticks(yticks)
    ax1.set_ylabel(r'$y/\lambda_{si}$', fontsize=fs[0])
    ax1.set_aspect('equal')
    if nstep < 1000000:
        ax1.text(xlim[0]+80,14,'nstep='+ntitle, fontsize=fs[1])
    else:
        ax1.text(xlim[0]+78.3,14,'nstep='+ntitle, fontsize=fs[1])
    ax1.text(xlim[0]+33,14,r'$t=$'+tgam, fontsize=fs[1])
    ax1.text(xlim[0]+44,14,r'$\Omega_{i}^{-1}$'+' (log, floor='+floorstr+')', fontsize=fs[1])
    ax1.text(xlim[0],13.0,'electron density', fontsize=fs[1])
    #ax1.axvline(xshock1, color='black', linestyle='dashed', linewidth=1)
    ax1.axvline(xshock2, color='white', linestyle='dashed', linewidth=1)
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax1.yaxis.set_major_locator(MultipleLocator(4))
    # ax1.plot(tracex,tracey, color='black', linewidth=0.5)
    ax1.scatter(tracex,tracey, color='black', s=0.05)
    ax1.scatter(tracex[-1],tracey[-1], color='black', s=10.0)

    # cset2 = ax2.pcolormesh(xp/skin,yp/skin,DENSI, cmap='jet', vmin=afloorde, vmax=0.7)
    # cset2 = ax2.contourf(xp/skin,yp/skin,DENSI, clvls, cmap='jet')
    plt.setp(ax1.get_xticklabels(), fontsize=fs[0])
    # plt.setp(ax2.get_yticklabels(), fontsize=fs[0])
    # ax2.set_ylabel(r'$y/\lambda_{si}$', fontsize=fs[0])
    ax1.set_aspect('equal')
    # # ax2.set_yticks(yticks)
    ax1.set_xlabel(r'$x/\lambda_{si}$', fontsize=fs[0])
    # ax2.text(xlim[0],13.0,'ion density', fontsize=fs[1])
    # #ax2.axvline(xshock1, color='black', linestyle='dashed', linewidth=1)
    # ax2.axvline(xshock2, color='white', linestyle='dashed', linewidth=1)
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.xaxis.set_major_locator(MultipleLocator(10))
    # ax2.yaxis.set_minor_locator(MultipleLocator(1))
    # ax2.yaxis.set_major_locator(MultipleLocator(4))


    cticks = clvls = np.linspace(-0.3,0.7,11)
    fig.subplots_adjust(right=0.8)
    cbar1 = fig.colorbar(cset1, ax=ax1, ticks=cticks)
    cbar1.set_label(r'$N/N_0$', fontsize=fs[1])
    cbar1.ax.set_yticklabels(["{:4.2f}".format(i) for i in cticks])

    # gs1.tight_layout(fig)
    # plt.tight_layout()
    ntitle = '{:07d}'.format(nstep)


    allplots.save(fig, "./dens_tr__"+ntitle+".png")
    plt.close(fig)

allplots.join()
f.close()
