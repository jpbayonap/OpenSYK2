# Import libraries
from random import random
from re import L
from tkinter.ttk import Progressbar
import qutip as qt
import numpy as np
from numpy import linalg
import itertools
from qutip.qip.operations import swap
import pickle
import json
import os
from joblib import Parallel, delayed
import time
import matplotlib.pyplot as plt

from Custom_Bar import TextProgressBar

############################################################
############################################################


# TEST BETTER COMMENTS
# * IMPORTANT INFO IS HIGHTLIGHTED
# ! DEPRECATED METHOD, DO NOT USE
# ? SHOUD THIS METHOD BE EXPOSED
# TODO: REFACTOR THIS FUNCTION
# @param myParam


import logging 

############################################################
############################################################

from OpenSYK_functions import *
from OpenSYK_images import *

############################################################
############################################################

N_JOBS = 14
# random realizations
samples = 250
I, J = 0, 0
# Coupling strength
K = 1
# Temperature
L = 0
# Particle number
# M = 1
M = -1
DELTA_T = 10
# DELTA = 0
# OPTIONS= qt.Options
font_size = 24

home = "/home/j-bayona/Git/"
    #         home ='/home/pablo/Git/OpenSYK2/'
def path(Model,Delta, s):
    path_m = home + "ImagesBatchParallel/OccupationNumber/N_%s" % (Model.N) + "/loss_gain/"
    
    name_diag = (
        "diag TEST%.1f OccupationN samples_%s " % (Delta, s)
        + "w%.1f_g%.3f_" % (Model.w, Model.g)
        + "N%s_" % (Model.N)
        + "gamma%.3f_beta%.3f.png" % (Model.gamma, Model.beta)
    )

    isExist = os.path.exists(path_m)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path_m)
        print("The new directory is created!")
    return os.path.join(path_m, name_diag)



############ Process to parallelize ###################


if __name__ == "__main__":
    logger.debug("start")

    random_seed = np.random.randint(np.iinfo(np.int32).max, size=samples)
    # random_seed = ImportData_seed(I, J, K, L, M, samples, DELTA_T)
    # random_seed = ImportData_seed(I, J, K, L, M, samples, DELTA)

    random_evo = Parallel(
        n_jobs=N_JOBS, verbose=5, backend="loky", pre_dispatch="1.5*n_jobs"
    )(delayed(EVO)(r, I, J, K, L, M) for r in random_seed)

    random_evo = np.array(random_evo, dtype=object)

    purity = np.take(random_evo, 0, axis=1)
    entropy = np.take(random_evo, 1, axis=1)
    counts = np.take(random_evo, 2, axis=1)
    traced = np.take(random_evo, 3, axis=1)
    eigen_L = np.take(random_evo, 4, axis=1)
    eigen_H = np.take(random_evo, 5, axis=1)
    relative_ent = np.take(random_evo, 6, axis=1)

    fig = plt.figure(figsize=(10, 10))  # Figureを作成
    ax = fig.add_subplot(1, 1, 1)  # Axesを作成
    occ_N(ax, I, J, K, L, M, counts, delta, end, samples, font_size)
    Model = create_Model(I,J,K,L,M)
    Path= path(Model,DELTA_T, samples)
    plt.savefig(Path)
    plt.close(fig)

    # saveDATA(
    #     I,
    #     J,
    #     K,
    #     L,
    #     M,
    #     random_seed,
    #     samples,
    #     purity,
    #     entropy,
    #     traced,
    #     eigen_H,
    #     counts,
    #     eigen_L,
    #     relative_ent,
    # )

# #mprof run  --multiprocess -o delta_500 /home/j-bayona/Git/python/OpenSYK_mainOP.py
