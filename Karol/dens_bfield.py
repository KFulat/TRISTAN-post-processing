"""
Python script producing density plots from movHR files.
"""
import shocks_postprocessing_module as spm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
plt.style.use('./spectra.mplstyle')

# Time steps range
step_min = np.int32(590000)
step_max = np.int32(590000)
step_dens = np.int32(1000)

# Arrays creation
dense_array, densi_array, xp, yp = spm.create_density_arrays()
bx_array, by_array, bz_array, xp, yp = spm.create_bfield_arrays()
ex_array, ey_array, ez_array, xp, yp = spm.create_bfield_arrays()

# Asynchronous plotting initialization
allplots = spm.AsyncPlotter()

for step in range(step_min, step_max+1, step_dens):
    dense, densi = spm.load_density_data(dense_array,
                                         densi_array, step)
    bx, by, bz = spm.load_bfield_data(bx_array,
                                      by_array, bz_array, step)
    # ex, ey, ez = spm.load_efield_data(ex_array,
    #                                   ey_array, ez_array, step)

    x_shock_linear = spm.shock_position_linear(step)
    x_shock = spm.shock_position_from_array(dense.T, xp)

    # PLOT
    fig = plt.figure(figsize=(12.31*0.8, 9.2*1.118), dpi=300)

    # x axes limits
    if step > spm.STEP_LIMIT:
        delta = x_shock_linear - spm.shock_position_linear(spm.STEP_LIMIT)
        # xlim = (0+delta, 100+delta)
        xlim = (115, 195)
    else:
        xlim = (20, 80)
    ylim = (0, spm.MY/spm.LSI)

    # Adding axes
    ax3 = fig.add_axes([0.05, 0.05, 0.88, 0.45], xlim=xlim, ylim=ylim)
    ax1 = fig.add_axes([0.05, 0.46, 0.88, 0.45], sharex=ax3, ylim=ylim)
    cbax3 = fig.add_axes([0.95, 0.086, 0.03, 0.379])
    # cbax2 = fig.add_axes([0.95, 0.368, 0.015, 0.284])
    cbax1 = fig.add_axes([0.95, 0.4955, 0.03, 0.379])

    # Contour plots
    lvls_dens = 21
    lvls_bfield = 21
    lvls_efield = 21
    clvls_dens = np.linspace(-0.2, 0.7, lvls_dens)
    clvls_bfield = np.linspace(-2.6, 2.6, lvls_bfield)
    clvls_efield = np.linspace(-1.4, 1.4, lvls_efield)
    cset1 = ax1.contourf(xp, yp, dense.T, clvls_dens, cmap='jet')
    # cset2 = ax2.contourf(xp, yp, bz.T, clvls_bfield, cmap='jet')
    cset3 = ax3.contourf(xp, yp, bz.T, clvls_bfield, cmap='jet')
    # cset3 = ax3.contourf(xp, yp, ex.T, clvls_efield, cmap='jet')

    # # Lines shock position
    # ax1.axvline(x_shock, color='white', linestyle='dashed', linewidth=1)
    # ax2.axvline(x_shock, color='white', linestyle='dashed', linewidth=1)
    # ax3.axvline(x_shock, color='white', linestyle='dashed', linewidth=1)

    # Fontsize
    fs_ticks = 18
    fs_labels = 18
    fs_clabel = 16
    fs_text = 22

    # Axes tick params
    ax1.set_aspect('equal')
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.yaxis.set_major_locator(MultipleLocator(4))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax1.tick_params(axis='both', pad=8, labelsize=fs_ticks, length=10)
    ax1.tick_params(axis='both', which='minor', length=5)

    # ax2.set_aspect('equal')
    # plt.setp(ax2.get_xticklabels(), visible=False)
    # ax2.yaxis.set_major_locator(MultipleLocator(4))
    # ax2.yaxis.set_minor_locator(MultipleLocator(1))
    # ax2.tick_params(axis='both', pad=8, labelsize=fs_ticks, length=10)
    # ax2.tick_params(axis='both', which='minor', length=5)

    ax3.set_aspect('equal')
    ax3.xaxis.set_major_locator(MultipleLocator(10))
    ax3.xaxis.set_minor_locator(MultipleLocator(1))
    ax3.yaxis.set_major_locator(MultipleLocator(4))
    ax3.yaxis.set_minor_locator(MultipleLocator(1))
    ax3.tick_params(axis='both', pad=8, labelsize=fs_ticks, length=10)
    ax3.tick_params(axis='both', which='minor', length=5)

    # Axes labels
    ax1.set_ylabel(r'$\mathrm{y/\lambda_{si}}$', fontsize=fs_labels)
    # ax2.set_ylabel(r'$y/\lambda_{si}$', fontsize=fs_labels)
    ax3.set_xlabel(r'$\mathrm{x/\lambda_{si}}$', fontsize=fs_labels)
    ax3.set_ylabel(r'$\mathrm{y/\lambda_{si}}$', fontsize=fs_labels)

    # Text
    t1 = ax1.text(xlim[1]-10.6, ylim[1]-4.8, r'$\mathrm{N_e/N_0}$', fontsize=fs_text)
    t1.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='white'))
    # ax2.text(xlim[1]-12, ylim[1]-4.8, r'$e) B_z/B_0$', fontsize=fs_text)
    t2 = ax3.text(xlim[1]-10.6, ylim[1]-4.8, r'$\mathrm{B_z/B_0}$', fontsize=fs_text)
    t2.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='white'))
    # ax1.text(xlim[0]+83.2,14,'nstep='+step_str, fontsize=fs[1])
    # ax1.text(xlim[0]+34,14,r'$t=$'+time_str+' '+r'$\Omega_{i}^{-1}$',
    # fontsize=fs[1])
    # ax1.text(xlim[0]+50,14,'(log, floor='+floor_str+')',
    # fontsize=fs[1])

    # Colorbars
    cticks1 = np.linspace(-0.2, 0.7, 5)
    cbar = fig.colorbar(cset1, cax=cbax1, ax=ax1, ticks=cticks1)
    cbar.ax.tick_params(labelsize=fs_clabel, length=0)
    cbar.ax.set_yticklabels(["{:3.1f}".format(i) for i in cticks1])
    cbar.outline.set_visible(False)

    cticks3 = np.linspace(-2.6, 2.6, 5)
    cbar = fig.colorbar(cset3, cax=cbax3, ax=ax3, ticks=cticks3)
    cbar.ax.tick_params(labelsize=fs_clabel, length=0)
    cbar.ax.set_yticklabels(["{:3.1f}".format(i) for i in cticks3])
    cbar.outline.set_visible(False)

    # cticks3 = np.linspace(-1.4, 1.4, 11)
    # cbar = fig.colorbar(cset3, cax=cbax3, ax=ax3, ticks=cticks3)
    # cbar.ax.tick_params(labelsize=fs_clabel, length=5)
    # cbar.ax.set_yticklabels(["{:3.1f}".format(i) for i in cticks3])

    step_str = "{:07d}".format(step)
    fig_name = 'step_dens_bfield_'+step_str+'.png'
    allplots.save(fig, "./"+fig_name)
    plt.close(fig)

allplots.join()
