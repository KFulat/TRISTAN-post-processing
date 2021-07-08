"""
Python script producing magnetic field plots from movHR files.
"""
import shocks_postprocessing_module as spm

import numpy as np
import matplotlib.pyplot as plt

# Time steps range
step_min = np.int32(140000)
step_max = np.int32(1140000)
step_bfield = np.int32(1000)

# Arrays creation
bx_array, by_array, bz_array, xp, yp = spm.create_bfield_arrays()

# Asynchronous plotting initialization
allplots = spm.AsyncPlotter()

shock_position = np.loadtxt("shock_position.txt")

for step in range(step_min, step_max+1, step_bfield):
    bx, by, bz = spm.load_bfield_data(bx_array,
        by_array, bz_array, step)

    x_shock_linear = spm.shock_position_linear(step)
    x_shock = spm.shock_position_from_file(shock_position, step)

    fig = spm.plot_bfield(bx, by, bz, xp, yp,
            x_shock_linear, x_shock, step)
    step_str = "{:07d}".format(step)
    fig_name = 'step_bfield_'+step_str+'.png'
    allplots.save(fig, spm.bfield_plot_path+fig_name)
    plt.close(fig)

allplots.join()
