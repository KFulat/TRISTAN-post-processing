"""Python script producing density plots from movHR files."""

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


data_path = Path('./result/trace')
files = np.array([f for f in data_path.glob("trej_*")])
gamma_trej =[]

for i, file in enumerate(files):
    gamma = np.loadtxt(file, skiprows=1, usecols=1)
    # tab = np.transpose(tab)
    gamma_max = np.max(gamma)
    if gamma_max >= 10.0:
        gamma_trej.append((gamma_max, file.name))
    if i%10 == 0:
        print(i)

# print(gamma_trej[0])
sorted_gamma_trej = sorted(gamma_trej, key=lambda x: x[0])
gamma_trej = np.array(sorted_gamma_trej)

np.savetxt("gamma_trej.txt", gamma_trej, fmt=['%.5s','%s'])
