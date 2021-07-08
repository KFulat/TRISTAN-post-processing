"""
Python script producing phase space plots from phase files.
"""
import shocks_postprocessing_module as spm

import numpy as np
import matplotlib.pyplot as plt

# Time steps range
step_min = np.int32(880000)
step_max = np.int32(880000)
step_phase = np.int32(1000)

# Arrays creation
pxe_array, pye_array, pze_array, xp, yp = spm.create_phase_arrays()

# Asynchronous plotting initialization
allplots = spm.AsyncPlotter()

shock_position = np.loadtxt("shock_position.txt")

for step in range(step_min, step_max+1, step_phase):
    pxe, pye, pze = spm.load_phase_data(pxe_array,
        pye_array, pze_array, step)

    x_shock_linear = spm.shock_position_linear(step)
    x_shock = spm.shock_position_from_file(shock_position, step)

    fig = spm.plot_phase(pxe, pye, pze, xp, yp,
            x_shock_linear, x_shock, step)
    step_str = "{:07d}".format(step)
    fig_name = 'step_phase_'+step_str+'.png'
    allplots.save(fig, spm.phase_plot_path+fig_name)
    plt.close(fig)

allplots.join()
