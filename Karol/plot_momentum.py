import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import scipy as scp
import h5py
from scipy import ndimage
from scipy.interpolate import splev, splprep
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
import matplotlib.collections as mcoll

# Plasma parameters
DENS0 = 20.0
GAMMA = 16329.956
LSI = 100.0
ME = 1.0
C = 0.5
QE = -0.01120846
V_JET = -0.1*C
GM_JET = C/np.sqrt(C**2-V_JET**2)
THETA = 75.0
PHI = 0.0
B0 = 0.5491
B0X = B0*np.cos(THETA*np.pi/180.0)
B0Y = B0*np.sin(THETA*np.pi/180.0)*np.cos(PHI*np.pi/180.0)
B0Z = B0*np.sin(THETA*np.pi/180.0)*np.sin(PHI*np.pi/180.0)
V_SH = 0.16*C
VT = V_SH/np.cos(THETA*np.pi/180.0)
GM_T = C/np.sqrt(C**2-VT**2)

trej_path = Path('./result/trace/')
trej_names = np.loadtxt("gamma_trej.txt", dtype=str, usecols=1)

def dens_log(floor, norm, tab):
    """Make logscale density array."""
    tab = tab/norm
    index_sign = np.sign(tab)
    tab_mask = ma.masked_equal(tab, 0.0)
    tabb = np.log10(np.abs(tab_mask))
    tabb_mask = ma.masked_less_equal(tabb, floor)
    tabb = tabb_mask.filled(0)
    tabb = tabb*index_sign
    return tabb

def load_trej_data(filename, step_min, step_max):
    """Load trej file data."""
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

def load_electron_density_profile(step_min, step_max, step_dens):
    """Calculate y-averaged density profile from movHR file"""
    # Box parameters
    Npx = 18
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

    dense_profile=[]

    for nstep in range(step_min, step_max+1, step_dens):
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
        dense = scp.ndimage.filters.gaussian_filter(dense, 2.5)
        # Set minimum field strength for log plots
        if ilog == 1:
            dense = dens_log(afloorde, DENS0, dense)
        dense_y_mean = np.mean(dense, axis=1)
        dense_profile.append(dense_y_mean)
    dense_profile = np.array(dense_profile)
    return dense_profile, xp, yp

def colorline(x, y, z=None, cmap='copper', norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """Adapted from https://stackoverflow.com/questions/36074455/python-matplotlib-with-a-line-color-gradient-and-colorbar

    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, "__iter__"):
        z = np.array([z])
    z = np.asarray(z)
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)
    return lc

def make_segments(x, y):
    """Create list of line segments from x and y coordinates, in the
    correct format for LineCollection: an array of the form numlines
    x (points per line) x 2 (x and y) array.
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

# Simulation steps
step_min = np.int(70000)
step_max = np.int(480000)
step_dens = np.int(1000)
step_trace = np.int(20)

# Frame specification
HT = True
UP = False

# dense_profile, x_dens, y_dens = load_electron_density_profile(step_min,
#     step_max,step_dens)
# time_dens = [i/GAMMA for i in range(step_min,step_max+1,step_dens)]

# Main loop
for i, particle in enumerate(trej_names):
    print(particle)
    (step, gm, x, y, px, py, pz,
        ex, ey, ez, bx, by, bz) = load_trej_data(particle, step_min, step_max)
    time = step/GAMMA

    # Quantities calculation
    vx = px/gm
    vy = py/gm
    vz = pz/gm

    vx_up = (vx-V_JET)/(1.0-vx*V_JET/C**2)
    vy_up = vy*np.sqrt(1.0-(V_JET/C)**2)/(1.0-vx*V_JET/C**2)
    vz_up = vz*np.sqrt(1.0-(V_JET/C)**2)/(1.0-vx*V_JET/C**2)
    v_up = np.sqrt(vx_up**2+vy_up**2+vz_up**2)
    gm_up = C/np.sqrt(C**2-v_up**2)

    if HT == True:
        cosa = (vx_up*B0X+vy_up*B0Y+vz_up*B0Z)/(v_up*B0)
        v_up_parl = v_up*cosa
        v_up_perp = v_up*np.sqrt(1.0-cosa**2)
        v_ht_parl = (v_up_parl-VT)/(1.0-v_up_parl*VT/C**2)
        v_ht_perp = v_up_perp/(1.0-v_up_parl*VT/C**2)/GM_T

        pparl = v_ht_parl/C*gm_up*GM_T*(1.0-v_up_parl*VT/C**2)
        pperp = v_ht_perp/C*gm_up*GM_T*(1.0-v_up_parl*VT/C**2)
    else:
        cosa = ((px*bx+py*by+pz*bz)
            /np.sqrt((px**2+py**2+pz**2)*(bx**2+by**2+bz**2)))

        pparl = np.sqrt(gm**2-1.0)*cosa
        pperp = np.sqrt(gm**2-1.0)*np.sqrt(1.0-cosa**2)

    if UP == True:
        gam = gam_up-1.0
        ex_up = ex*GAM_JET
        ey_up = (ey-V_JET*bz)*GAM_JET
        ez_up = (ez+V_JET*by)*GAM_JET

        qve = QE*(ex_up*vx_up+ey_up*vy_up+ez_up*vz_up)*gamma/(ME*C**2)
        qvex = QE*ex_up*vx_up*gamma/(ME*C**2)
        qvey = QE*ey_up*vy_up*gamma/(ME*C**2)
        qvez = QE*ez_up*vz_up*gamma/(ME*C**2)
    else:
        gam = gm-1.0
        qve = QE*(ex*vx + ey*vy + ez*vz)*GAMMA/(ME*C**2)
        qvex = QE*ex*vx*GAMMA/(ME*C**2)
        qvey = QE*ey*vy*GAMMA/(ME*C**2)
        qvez = QE*ez*vz*GAMMA/(ME*C**2)

    gamparl = gam*cosa**2
    gamperp = gam*(1.0 - cosa**2)

    acc = [0.0]
    # sda = [0.0]
    sda = [gam[0]]
    for i, val in enumerate(gam[:-1]):
        acc.append((gam[i+1]-gam[i])/step_trace*GAMMA)
        sda.append(sda[i]+qvez[i+1]/GAMMA*step_trace)

    # Filtering
    gam = scp.ndimage.filters.gaussian_filter1d(gam, 50.0)
    gamparl = scp.ndimage.filters.gaussian_filter1d(gamparl, 50.0)
    gamperp = scp.ndimage.filters.gaussian_filter1d(gamperp, 50.0)
    sda = scp.ndimage.filters.gaussian_filter1d(sda, 50.0)
    qvez = scp.ndimage.filters.gaussian_filter1d(qvez, 50.0)
    acc = scp.ndimage.filters.gaussian_filter1d(acc, 50.0)

    # Momentum line  interpolation
    tck,u = splprep([pparl,pperp],s=0.0)
    x_i,y_i = splev(u,tck)

    # PLOT SECTION
    fig = plt.figure(figsize=(10,16), dpi=200)
    fs = [18,22,16,12]

    # Limits and colormap levels
    xmin_trace = time[0]
    xmax_trace = time[-1]
    xlim_trace = (xmin_trace,xmax_trace)
    ymin_trace = np.round(np.min(x/LSI)-10,-1)
    ymax_trace = np.round(np.max(x/LSI)+10,-1)
    ylim_trace = (ymin_trace, ymax_trace)

    ymax_qvez = np.round(np.max(qvez)+5,-1)
    ylim_qvez = (-ymax_qvez,ymax_qvez)

    xmax_mom = np.round(np.max(pparl)+2,0)
    ymax_mom = np.round(np.max(pperp)+2,0)
    xlim_mom = (-xmax_mom,xmax_mom)
    ylim_mom = (0.0,ymax_mom)

    levels = 21
    clvls = np.linspace(-0.3,0.7,levels)

    # Axes specification
    ax4 = fig.add_axes([0.1, 0.055, 0.85, 0.19],xlim=xlim_mom, ylim=ylim_mom)
    ax3 = fig.add_axes([0.1, 0.30, 0.85, 0.22],xlim=xlim_trace, ylim=ylim_trace)
    ax2 = fig.add_axes([0.1, 0.53, 0.85, 0.22],ylim=ylim_qvez, sharex=ax3)
    ax1 = fig.add_axes([0.1, 0.76, 0.85, 0.22],sharex=ax3)

    # Kinetic energy plot
    ax1.plot(time, gam, color='black', label=r'$(\gamma-1)_{simulation}$')
    ax1.plot(time, gamparl, color='blue', label=r'$(\gamma-1)_{\parallel}$')
    ax1.plot(time, gamperp, color='green', label=r'$(\gamma-1)_{\perp}$')
    ax1.plot(time, sda, color='red', label=r'$(\gamma-1)_{drift}$')
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), fontsize=fs[0])
    ax1.set_ylabel(r'$\gamma-1$', fontsize=fs[1])
    ax1.yaxis.set_major_locator(MultipleLocator(5))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax1.legend(loc='best', fontsize=fs[3])

    # Qvez and acc plot
    ax2.plot(time, qvez, color='red', label=r'$q(v_z \cdot E_z)/(mc^2)$')
    ax2.plot(time, acc, color='black', label=r'$\Delta(\gamma-1)$')
    ax2.axhline(0.0, color='black', linestyle='dashed')
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), fontsize=fs[0])
    ax2.set_ylabel(r'$d\gamma/d(t\Omega_i)$', fontsize=fs[1])
    ax2.yaxis.set_major_locator(MultipleLocator(4))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax2.legend(loc='best', fontsize=fs[3])

    # Trace on y-averaged density plot
    ax3.plot(time, x/LSI, color='black', label=r'$$')
    # ax3.contourf(time_dens,x_dens/LSI, dense_profile.T, clvls, cmap='jet')
    plt.setp(ax3.get_xticklabels(), fontsize=fs[0])
    plt.setp(ax3.get_yticklabels(), fontsize=fs[0])
    ax3.set_xlabel(r'$t\Omega_{i}$', fontsize=fs[1])
    ax3.set_ylabel(r'$x/\lambda_{si}$', fontsize=fs[1])
    ax3.xaxis.set_major_locator(MultipleLocator(1))
    ax3.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax3.yaxis.set_major_locator(MultipleLocator(20))
    ax3.yaxis.set_minor_locator(MultipleLocator(2))

    # Momentum plot
    lc = colorline(x_i,y_i, cmap='jet', linewidth=0.5)
    ax4.add_collection(lc)
    # ax4.colorbar(lc)
    sc = ax4.scatter(pparl, pperp, s=0.2, c=time, cmap='jet')
    # ax4.plot(x_i, y_i, linewidth=0.1)
    ax4.axvline(0.0, color='black', linestyle='dashed')
    ax4.set_aspect('equal')
    plt.setp(ax4.get_xticklabels(), fontsize=fs[0])
    plt.setp(ax4.get_yticklabels(), fontsize=fs[0])
    if HT == True:
        ax4.set_xlabel(r'$p_{\parallel}/(mc) (HT)$', fontsize=fs[1])
        ax4.set_ylabel(r'$p_{\perp}/(mc) (HT)$', fontsize=fs[1])
    elif UP == True:
        ax4.set_xlabel(r'$p_{\parallel}/(mc) (UP)$', fontsize=fs[1])
        ax4.set_ylabel(r'$p_{\perp}/(mc) (UP)$', fontsize=fs[1])
    else:
        ax4.set_xlabel(r'$p_{\parallel}/(mc)$', fontsize=fs[1])
        ax4.set_ylabel(r'$p_{\perp}/(mc)$', fontsize=fs[1])
    ax4.xaxis.set_major_locator(MultipleLocator(5))
    ax4.xaxis.set_minor_locator(MultipleLocator(1))
    ax4.yaxis.set_major_locator(MultipleLocator(5))
    ax4.yaxis.set_minor_locator(MultipleLocator(1))

    # Colormap
    cbaxes = inset_axes(ax4, width="40%", height="4%", loc=2)
    cbar = plt.colorbar(sc, cax=cbaxes, orientation='horizontal')
    cbar.set_label(r'$t\Omega_{i}$', fontsize=fs[2])

    plt.savefig("./tracing/plot_momentum/"+particle+".png")
    plt.close(fig)
