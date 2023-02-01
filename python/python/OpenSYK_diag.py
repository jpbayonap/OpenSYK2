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
I, J = 1, 0
# Coupling strength
K = int(sys.argv[1])
# K= 0
# Temperature
L = 0


# Particle number
# M = 1
M = -1
# DELTA = 100
# DELTA = 0
# OPTIONS= qt.Options
font_size = 20
type = 1


home = "/home/j-bayona/Git/"
#         home ='/home/pablo/Git/OpenSYK2/'


def path(Model, Delta, s, COND, var, mu, ini):
    if COND:
        path_m = (
            home
            + "ImagesBatchParallel/OccupationNumber/N_%s" % (Model.N)
            + "/M_%s/loss_gain/S_%s/J_%.3f/beta_%.3f/%s/"
            % (type, s, Model.gamma, Model.beta, ini)
        )
        path_mL = (
            home
            + "ImagesBatchParallel/Eigenvalues_L/N_%s" % (Model.N)
            + "/M_%s/loss_gain/S_%s/J_%.3f/beta_%.3f"
            % (type, s, Model.gamma, Model.beta)
        )
        path_mP = (
            home
            + "ImagesBatchParallel/Purity/N_%s" % (Model.N)
            + "/M_%s/loss_gain/S_%s/J_%.3f/beta_%.3f/%s/"
            % (type, s, Model.gamma, Model.beta, ini)
        )
        name_diag = (
            "ANALYTIC%.1f OccupationN  LAMBDA mu%.2f var%.2f" % (Delta, mu, var)
            + "w%.1f_g%.3f_" % (Model.w, Model.g)
            + "N%s_" % (Model.N)
            + "gamma%.3f_beta%.3f.png" % (Model.gamma, Model.beta)
        )
        name_diagP = (
            "ANALYTIC%.1f Purity LAMBDA mu%.2f var%.2f " % (Delta, mu, var)
            + "w%.1f_g%.3f_" % (Model.w, Model.g)
            + "N%s_" % (Model.N)
            + "gamma%.3f_beta%.3f.png" % (Model.gamma, Model.beta)
        )
        name_diagL = (
            "ANALYTIC%.1f EigL  LAMBDA mu%.2f var%.2f" % (Delta, mu, var)
            + "w%.1f_g%.3f_" % (Model.w, Model.g)
            + "N%s_" % (Model.N)
            + "gamma%.3f_beta%.3f.png" % (Model.gamma, Model.beta)
        )
    else:
        path_m = (
            home
            + "ImagesBatchParallel/OccupationNumber/N_%s" % (Model.N)
            + "/M_%s/loss_gain/FIXED/%s/J_%.3f/beta_%.3f"
            % (type, ini, Model.gamma, Model.beta)
        )
        path_mL = (
            home
            + "ImagesBatchParallel/Eigenvalues_L/N_%s" % (Model.N)
            + "/M_%s/loss_gain/FIXED/%s/J_%.3f/beta_%.3f"
            % (type, ini, Model.gamma, Model.beta)
        )
        path_mP = (
            home
            + "ImagesBatchParallel/Purity/N_%s" % (Model.N)
            + "/M_%s/loss_gain/FIXED/%s/J_%.3f/beta_%.3f"
            % (type, ini, Model.gamma, Model.beta)
        )

        name_diag = (
            "ANALYTIC%.1f OccupationN  lambda" % (Delta)
            + "w%.1f_g%.3f_" % (Model.w, Model.g)
            + "N%s_" % (Model.N)
            + "gamma%.3f_beta%.3f.png" % (Model.gamma, Model.beta)
        )
        name_diagP = (
            "ANALYTIC%.1f Purity lambda " % (Delta)
            + "w%.1f_g%.3f_" % (Model.w, Model.g)
            + "N%s_" % (Model.N)
            + "gamma%.3f_beta%.3f.png" % (Model.gamma, Model.beta)
        )
        name_diagL = (
            "ANALYTIC%.1f EigL  lambda" % (Delta)
            + "w%.1f_g%.3f_" % (Model.w, Model.g)
            + "N%s_" % (Model.N)
            + "gamma%.3f_beta%.3f.png" % (Model.gamma, Model.beta)
        )
    isExist = os.path.exists(path_m)
    isExistP = os.path.exists(path_mP)
    isExistL = os.path.exists(path_mL)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path_m)
        print("The new directory is created!")
    if not isExistP:
        # Create a new directory because it does not exist
        os.makedirs(path_mP)
        print("The new directory is created!")
    if not isExistL:
        # Create a new directory because it does not exist
        os.makedirs(path_mL)
        print("The new directory is created!")

    return (
        os.path.join(path_m, name_diag),
        os.path.join(path_mL, name_diagL),
        os.path.join(path_mP, name_diagP),
    )


ISRN = False
# VARIANCE = 0.03
VARIANCE = 1
INI = "ALL"
if __name__ == "__main__":

    logger.debug("start")

    Model = create_Model(I, J, K, L, M)
    MU = 0
    print(
        "d=%.1f S=%s, N=%s, w=%.3f, gamma=%.3f, g=%.3f, beta=%.3f"
        % (delta, samples, Model.N, Model.w, Model.gamma, Model.g, Model.beta),
        " process started",
    )
    if ISRN:
        random_seed = np.random.randint(np.iinfo(np.int32).max, size=samples)
    else:
        random_seed = [ 0.95]
        # random_seed = [-0.1,-0.95, -1]
    # random_seed = ImportData_seed(I, J, K, L, M, samples, DELTA)
    # random_counts = ImportData(I, J, K, L, M, samples, DELTA)

    COUNTS = Parallel(
        n_jobs=N_JOBS, verbose=5, backend="loky", pre_dispatch="1.5*n_jobs"
    )(
        delayed(EVO_diag)(r, I, J, K, L, M, type, ISRN, VARIANCE, MU)
        for r in random_seed
    )
    COUNTS = np.array(COUNTS, dtype=object)
    # COUNTS_d = np.take(COUNTS, 0, axis=1)
    COUNTS_cond = np.take(COUNTS, 0, axis=1)
    COUNTS_a = np.take(COUNTS, 1, axis=1)
    COUNTS_infty = np.take(COUNTS, 2, axis=1)

    COUNTS_p = np.take(COUNTS, 3, axis=1)
    COUNTS_l = np.take(COUNTS, 4, axis=1)
    COUNTS_G1 = np.take(COUNTS, 5, axis=1)

    # Plots

    fig = plt.figure(figsize=(6, 6))  # Figureを作成
    ax = fig.add_subplot(1, 1, 1)  # Axesを作成

    occ_N(
        ax,
        I,
        J,
        K,
        L,
        M,
        COUNTS_a,
        COUNTS_infty,
        COUNTS_cond,
        VARIANCE,
        MU,
        random_seed,
        font_size,
        "analytic",
        1,
        "r",
    )

    del COUNTS
    Path, PathL, PathP = path(Model, delta, samples, ISRN, VARIANCE, MU, INI)

    plt.savefig(Path)
    plt.close(fig)

    fig2 = plt.figure(figsize=(6, 6))  # Figureを作成
    ax2 = fig2.add_subplot(1, 1, 1)  # Axesを作成
    PURITY_im(
        ax2,
        I,
        J,
        K,
        L,
        M,
        COUNTS_p,
        COUNTS_cond,
        VARIANCE,
        MU,
        random_seed,
        font_size,
        "diag",
        0,
        "r",
        COUNTS_G1,
    )

    plt.savefig(PathP)
    plt.close(fig)
    fig3 = plt.figure(figsize=(6, 6))  # Figureを作成
    ax3 = fig3.add_subplot(1, 1, 1)  # Axesを作成
    Lind_spec(
        ax3,
        I,
        J,
        K,
        L,
        M,
        COUNTS_l,
        COUNTS_cond,
        VARIANCE,
        MU,
        random_seed,
        font_size,
    )
    plt.savefig(PathL)
    plt.close(fig)

    print(
        "d=%.1f S=%s, N=%s, w=%.3f, gamma=%.3f, g=%.3f, beta=%.3f"
        % (delta, samples, Model.N, Model.w, Model.gamma, Model.g, Model.beta),
        " process saved",
    )

# ! poster:cond-mat_510 凝縮系　user name
