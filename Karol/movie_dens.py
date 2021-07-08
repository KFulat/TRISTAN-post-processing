"""
Python script producing density plots from movHR files.
"""
import shocks_postprocessing_module as spm

import numpy as np
import matplotlib.pyplot as plt

import time

# Time steps range
step_min = np.int32(400000)
step_max = np.int32(405000)
step_dens = np.int32(1000)

# Arrays creation
dense_array, densi_array, xp, yp = spm.create_density_arrays()

# Asynchronous plotting initialization
allplots = spm.AsyncPlotter()

shock_position_file = open("shock_position.txt", "w")

t1=time.perf_counter()

for step in range(step_min, step_max+1, step_dens):
    dense, densi = spm.load_density_data(dense_array,
        densi_array, step)

    x_shock_linear = spm.shock_position_linear(step)
    x_shock = spm.shock_position_from_array(dense.T, xp)

    shock_position_file.write("%d %.2f\n" % (step, x_shock))

    fig = spm.plot_dens(dense, densi, xp, yp,
            x_shock_linear, x_shock, step)
    step_str = "{:07d}".format(step)
    fig_name = 'step_dens_'+step_str+'.png'
    allplots.save(fig, spm.dens_plot_path+fig_name)
    plt.close(fig)

allplots.join()
shock_position_file.close()
t2=time.perf_counter()
print(f'Finished in {t2-t1} seconds.')
