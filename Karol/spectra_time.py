import shocks_postprocessing_module as spm

import numpy as np
import matplotlib.pyplot as plt

# Time steps range
step_min = np.int(160000.0)
step_max = np.int(820000.0)
step_spectra = np.int(10000.0)

# Layers
x_left_downstream = np.int(28)
x_left_upstream = np.int(4)
layers_thikness = 8*spm.LSI
layers_number = np.int(1)  # number of layers
layers_distance = spm.LSI*4  # distance

(spectra_downstream,
 spectra_upstream,
 gamma_max,
 xp) = spm.load_spectra(
    layers_number, layers_thikness, layers_distance,
    x_left_downstream, x_left_upstream, step_min, step_max,
    step_spectra)

spm.plot_spectra_downstream(
    layers_number, x_left_downstream, layers_thikness, layers_distance,
    spectra_downstream, xp, step_min, step_max, step_spectra)
spm.plot_spectra_upstream(
    layers_number, x_left_upstream, layers_thikness, layers_distance,
    spectra_upstream, gamma_max, xp, step_min, step_max, step_spectra)
