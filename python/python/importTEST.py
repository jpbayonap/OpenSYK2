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
import sys

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

N_JOBS = 1
# random realizations
samples = 1
I, J = 0, 0
# Coupling strength
K = int(sys.argv[1])
# K= 0
# Temperature
L = 1
# Particle number
# M = 1
M = -1
# DELTA = 100
# DELTA = 0
# OPTIONS= qt.Options
font_size = 24
type = 0


if __name__ == "__main__":

    # logger.debug("start")
    random_seed = np.random.randint(np.iinfo(np.int32).max, size=samples)
    Model = create_Model(I, J, K, L, M)
    print(
        "d=%.1f S=%s, N=%s, w=%.3f, gamma=%.3f, g=%.3f, beta=%.3f"
        % (delta, samples, Model.N, Model.w, Model.gamma, Model.g, Model.beta),
        " process started",
    )
    # RND_data = [batch_data(seed, Model, step)[4] for seed in random_seed]
    # K = state.randn(N, N)
    #     K = 0.5 * (K - K.T)
    # print(min(RND_data), max(RND_data))

    COUNTS = Parallel(
        n_jobs=N_JOBS, verbose=5, backend="loky", pre_dispatch="1.5*n_jobs"
    )(delayed(EVO_diag)(r, I, J, K, L, M, type) for r in random_seed)
    print(COUNTS)
    # COUNTS_d = np.take(COUNTS, 0, axis=1)
    # COUNTS_a = np.take(COUNTS, 1, axis=1)
    # COUNTS_i = np.take(COUNTS, 2, axis=1)
    # # print(COUNTS_a, COUNTS_d)
    # # lbd = np.arange(-3*np.sqrt(Model.N),
    # #                 3*np.sqrt(Model.N), 0.001*np.sqrt(Model.N))
    # # atai_infty = [occ_INFTYxpm(Model, l) for l in lbd]
    # fig = plt.figure(figsize=(10, 10))  # Figureを作成
    # ax = fig.add_subplot(1, 1, 1)  # Axesを作成
    # # # ax.scatter(lbd, atai_infty, marker="1",
    # # #            label=r" av(<d^\dagger d>)(\infty)", c='orange', s=40, alpha=1)

    # occ_N(
    #     ax,
    #     I,
    #     J,
    #     K,
    #     L,
    #     M,
    #     COUNTS_a,
    #     COUNTS_d,
    #     samples,
    #     font_size,
    #     "analytic",
    #     1,
    #     "r",
    # )
    # occ_N(ax, I, J, K, L, M, COUNTS_d, 0, samples, font_size, "diag", 0, "g")
    # plt.savefig("/home/j-bayona/Git/prev_central.png")
    # plt.close(fig)

    # a = Model.Liouvilian(step, random_seed)
    # atai = occ_N2(Model, times_m3, a[-1][0])

    # atai = occ_INFTY(Model, lbd)
    # print(atai_infty)
