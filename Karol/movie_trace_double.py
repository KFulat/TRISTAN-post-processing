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
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

# Assync plot initialization
allplots = AsyncPlotter()

# Plasma parameters
DENS0 = 20.0
GAMMA = 16329.956
SKIN = 100.0
ME = 1.0
C = 0.5
QE = -0.01120846

trej_path = Path('./result/trace/')
trej_names = np.loadtxt("trej_chosen_2.txt", dtype=str)

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

def load_trej_data(filename, step_min, step_max):
    data = np.loadtxt(trej_path/filename, skiprows=1)
    a = np.where(data[:,0] > step_min)
    tab = data[a]
    a = np.where(tab[:,0] <= step_max)
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

def load_electron_density_data(step_dens):
    nstep = step_dens
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
    iresp=2
    ires = 2
    mxxp = np.int32(mx/iresp+1)
    myyp = np.int32(my/iresp+1)
    mxxRp = np.int32(mxxp/ires+2)
    myyRp = np.int32(myyp/ires+2)
    dense=np.zeros((mxxRp,myyRp))

    xp = np.array([i for i in range(np.int32(mxxp/ires+2))])*iresp*ires+3.0
    yp = np.array([i for i in range(np.int32(myyp/ires+2))])*iresp*ires+3.0

    print("step_dens =", nstep)
    nstr = str(nstep)
    dense[:,:] = 0.0
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
    # Data reading
    a = group_id["denseresR"]
    dense[(iminreseR-1):imaxreseR,:] = a
    # Data filtering
    dense = scp.ndimage.filters.gaussian_filter(dense, 0.5)
    # Set minimum field strength for log plots
    if ilog == 1:
        dense = dens_log(afloorde, DENS0, dense)
    return dense.T, xp, yp

step_min = np.int(260000)
step_max = np.int(480000)
step_dens = np.int(1000)
step_trace = np.int(20)

# Loading tracing data
trej_data_particles = []
for i, particle in enumerate(trej_names):
    print(particle)
    trej_data = np.loadtxt("./result/trace/"+particle, skiprows=1, usecols=[0,2,3])
    trej_data_particles.append(trej_data)

# Loop over simulation steps
for step in range(step_min, step_max+step_dens, step_dens):
    dense, x_dens, y_dens = load_electron_density_data(step)
    tgam = "{:04.1f}".format(step/GAMMA)
    fig = plt.figure(figsize=(12,3.01), dpi=300)
    levels = 21
    clvls = np.linspace(-0.3,0.7,levels)
    cticks = np.linspace(-0.3,0.7,11)
    xlim = (0,100)
    xlim2 = (xlim[1], 2*xlim[1])
    ylim = (0,12)
    # ax1 = fig.add_subplot(1,1,1, xlim=(0,100))
    ax1 = fig.add_axes([0.05, 0.55, 0.88, 0.4], xlim=xlim, ylim=ylim)
    ax2 = fig.add_axes([0.05, 0.05, 0.88, 0.4], xlim=xlim2, ylim=ylim)

    cset1 = ax1.contourf(x_dens/SKIN,y_dens/SKIN,dense, clvls, cmap='jet')
    ax1.set_aspect('equal')
    ax1.set_ylabel(r'$y/\lambda_{si}$', fontsize=14)
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.xaxis.set_major_locator(MultipleLocator(10))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax1.yaxis.set_major_locator(MultipleLocator(4))
    ax1.text(xlim[0],13,'electron density', fontsize=16)
    ax1.text(xlim[0]+36,13,r'$t=$'+tgam, fontsize=16)
    ax1.text(xlim[0]+44.5,13,r'$\Omega_{i}^{-1}$'+' (log, floor=-0.3)', fontsize=16)
    ax1.text(xlim[0]+83.8,13,'nstep='+"{:7.0f}".format(step), fontsize=16)

    cset2 = ax2.contourf(x_dens/SKIN,y_dens/SKIN,dense, clvls, cmap='jet')
    ax2.set_aspect('equal')
    ax2.set_xlabel(r'$x/\lambda_{si}$', fontsize=14)
    ax2.set_ylabel(r'$y/\lambda_{si}$', fontsize=14)
    ax2.xaxis.set_minor_locator(MultipleLocator(1))
    ax2.xaxis.set_major_locator(MultipleLocator(10))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.yaxis.set_major_locator(MultipleLocator(4))

    cbaxes = fig.add_axes([0.95, 0.05, 0.01, 0.9])
    cbar = plt.colorbar(cset1, cax=cbaxes, orientation='vertical', ticks=cticks)
    cbar.ax.set_yticklabels(["{:4.1f}".format(i) for i in cticks])
    cbar.set_label(r'$N/N_0$', fontsize=12)

    for i,val in enumerate(trej_data_particles):
        index = np.where(val[:,0] <= step)
        tab = val[index]
        index = np.where(tab[:,0] > step-2*step_dens)
        trace_xy = tab[index,1:3].T
        trace_xy = np.reshape(trace_xy,(2, trace_xy.shape[1]))
        x = trace_xy[0]
        y = trace_xy[1]
        if len(x) == 0:
            continue
        ax1.scatter(x/SKIN,y/SKIN, color='black', s=0.08, marker=',')
        ax1.scatter(x[-1]/SKIN,y[-1]/SKIN, color='black', s=8.0, marker='o')
        ax2.scatter(x/SKIN,y/SKIN, color='black', s=0.08, marker=',')
        ax2.scatter(x[-1]/SKIN,y[-1]/SKIN, color='black', s=8.0, marker='o')
    allplots.save(fig, "./tracing/movie_trace/step_tr_"+str(step)+".png")
    plt.close(fig)
allplots.join()
# exit()
#
# levels = 21
# clvls = np.linspace(-0.3,0.7,levels)
#
# # ax4 = fig.add_subplot(4,1,4, xlim=xlim_mom, ylim=ylim_mom)
# # ax3 = fig.add_subplot(4,1,3, xlim=xlim_trace, ylim=ylim_trace)
# # ax2 = fig.add_subplot(4,1,2, ylim=ylim_qvez, sharex=ax3)
# # ax1 = fig.add_subplot(4,1,1, sharex=ax3)
#
# ax4 = fig.add_axes([0.1, 0.055, 0.85, 0.19],xlim=xlim_mom, ylim=ylim_mom)
# ax3 = fig.add_axes([0.1, 0.30, 0.85, 0.22],xlim=xlim_trace, ylim=ylim_trace)
# ax2 = fig.add_axes([0.1, 0.53, 0.85, 0.22],ylim=ylim_qvez, sharex=ax3)
# ax1 = fig.add_axes([0.1, 0.76, 0.85, 0.22],sharex=ax3)
#
# # gamma-1
# ax1.plot(time, gam, color='black', label=r'$(\gamma-1)_{simulation}$')
# ax1.plot(time, gamparl, color='blue', label=r'$(\gamma-1)_{\parallel}$')
# ax1.plot(time, gamperp, color='green', label=r'$(\gamma-1)_{\perp}$')
# ax1.plot(time, sda, color='red', label=r'$(\gamma-1)_{drift}$')
# plt.setp(ax1.get_xticklabels(), visible=False)
# plt.setp(ax1.get_yticklabels(), fontsize=fs[0])
# ax1.set_ylabel(r'$\gamma-1$', fontsize=fs[1])
# ax1.yaxis.set_major_locator(MultipleLocator(5))
# ax1.yaxis.set_minor_locator(MultipleLocator(1))
# ax1.legend(loc='best', fontsize=fs[3])
#
# # qvez and acc
# ax2.plot(time, qvez, color='red', label=r'$q(v_z \cdot E_z)/(mc^2)$')
# ax2.plot(time, acc, color='black', label=r'$\Delta(\gamma-1)$')
# ax2.axhline(0.0, color='black', linestyle='dashed')
# plt.setp(ax2.get_xticklabels(), visible=False)
# plt.setp(ax2.get_yticklabels(), fontsize=fs[0])
# ax2.set_ylabel(r'$d\gamma/d(t\Omega_i)$', fontsize=fs[1])
# ax2.yaxis.set_major_locator(MultipleLocator(4))
# ax2.yaxis.set_minor_locator(MultipleLocator(0.5))
# ax2.legend(loc='best', fontsize=fs[3])
#
# # trace
# ax3.plot(time, x/SKIN, color='black', label=r'$$')
# ax3.contourf(time_dens,x_dens/SKIN, dense_profile.T, clvls, cmap='jet')
# plt.setp(ax3.get_xticklabels(), fontsize=fs[0])
# plt.setp(ax3.get_yticklabels(), fontsize=fs[0])
# ax3.set_xlabel(r'$t\Omega_{i}$', fontsize=fs[1])
# ax3.set_ylabel(r'$x/\lambda_{si}$', fontsize=fs[1])
# ax3.xaxis.set_major_locator(MultipleLocator(1))
# ax3.xaxis.set_minor_locator(MultipleLocator(0.1))
# ax3.yaxis.set_major_locator(MultipleLocator(20))
# ax3.yaxis.set_minor_locator(MultipleLocator(2))
#
# # momentum
# sc = ax4.scatter(pparl, pperp, s=0.2, c=time, cmap='jet')
# ax4.axvline(0.0, color='black', linestyle='dashed')
# ax4.set_aspect('equal')
# plt.setp(ax4.get_xticklabels(), fontsize=fs[0])
# plt.setp(ax4.get_yticklabels(), fontsize=fs[0])
# ax4.set_xlabel(r'$p_{\parallel}/(mc)$', fontsize=fs[1])
# ax4.set_ylabel(r'$p_{\perp}/(mc)$', fontsize=fs[1])
# ax4.xaxis.set_major_locator(MultipleLocator(4))
# ax4.xaxis.set_minor_locator(MultipleLocator(1))
# ax4.yaxis.set_major_locator(MultipleLocator(4))
# ax4.yaxis.set_minor_locator(MultipleLocator(1))
#
# # fig.tight_layout()
#
# cbaxes = inset_axes(ax4, width="40%", height="4%", loc=2)
# cbar = plt.colorbar(sc, cax=cbaxes, orientation='horizontal')
# cbar.set_label(r'$t\Omega_{i}$', fontsize=fs[2])
