"""
Python script producing density plots from movHR files.
"""
import shocks_postprocessing_module as spm

import numpy as np
import matplotlib.pyplot as plt

import time
import concurrent.futures
from itertools import repeat

# Time steps range
step_min = np.int32(901000)
step_max = np.int32(1140000)
step_dens = np.int32(1000)

# Arrays creation
dense_array, densi_array, xp, yp = spm.create_density_arrays()

# Asynchronous plotting initialization
allplots = spm.AsyncPlotter()

t1=time.perf_counter()

steps = range(step_min, step_max+1, step_dens)

def process_step(step):
    dense, densi = spm.load_density_data(dense_array,
        densi_array, step)

    x_shock_linear = spm.shock_position_linear(step)
    x_shock = spm.shock_position_from_array(dense.T, xp)
    print(f'Step {step} was processed.')
    fig = spm.plot_dens(dense, densi, xp, yp,
           x_shock_linear, x_shock, step)
    step_str = "{:07d}".format(step)
    fig_name = 'step_dens_'+step_str+'.png'
    allplots.save(fig, spm.dens_plot_path+fig_name)
    plt.close(fig)
    return x_shock

with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
    result = executor.map(process_step, steps)

with open("shock.txt", "w") as f:
    for step,i in zip(steps,result):
        f.write('{:d} {:.2f}\n'.format(step,i))


# for step in steps:
#     process_step(step)

# for step in range(step_min, step_max+1, step_dens):
#     dense, densi = spm.load_density_data(dense_array,
#         densi_array, step)
#
#     x_shock_linear = spm.shock_position_linear(step)
#     x_shock = spm.shock_position_from_array(dense.T, xp)
#
#     shock_position_file.write("%d %.2f\n" % (step, x_shock))

#    fig = spm.plot_dens(dense, densi, xp, yp,
#            x_shock_linear, x_shock, step)
#    step_str = "{:07d}".format(step)
#    fig_name = 'step_dens_'+step_str+'.png'
#    allplots.save(fig, spm.dens_plot_path+fig_name)
#    plt.close(fig)

allplots.join()
t2=time.perf_counter()
print(f'Finished in {t2-t1} seconds.')
