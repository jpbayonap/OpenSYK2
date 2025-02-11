# Import libraries
from random import random
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

# N_JOBS = 8
N_JOBS = 10
home = "/home/j-bayona/Git/"
# logging.basicConfig(filename='debug.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s p%(process)s {%(filename)s:%(lineno)d}  - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def start_logger_if_necessary():
    logger = logging.getLogger("mylogger")
    if len(logger.handlers) == 0:
        logger.setLevel(logging.DEBUG)
        sh = logging.StreamHandler()
        sh.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s p%(process)s {%(filename)s:%(lineno)d}  - %(message)s"
            )
        )
        fh = logging.FileHandler("debug.log")
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s p%(process)s {%(filename)s:%(lineno)d}  - %(message)s"
            )
        )
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger


logger = start_logger_if_necessary()

logger.debug("start simulation")
############################################################
############################################################


from OpenSYK_functions import *


# merge data


def mergeDictionary(dict_1, dict_2):
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = np.array([value, dict_1[key]])
            dict_3[key] = dict_3[key].reshape(len(value) * 2)
    return dict_3


############################################################
############################################################


# import data
## import data ##
DATA = []
i, j = 0, 0
# Majorana fermion number
m = -1
# sampling number
samples = 500

# for a in range(len_k):
for a in [1]:
    #   for b in range(len_l):
    for b in [0]:
        Model = create_Model(0, 0, a, b, m)

        N = Model.N

        base_folder = f"../DataBatchParallel/N_{Model.N}/loss_gain/S_{samples}/"

        name_1 = os.path.join(
            base_folder,
            "p1/seed w%.3f_g%.3f_N%s_gamma%.3f_beta%.6f"
            % (Model.w, Model.g, N, Model.gamma, Model.beta)
            + ".pickle",
        )

        name_2 = os.path.join(
            base_folder,
            "p2/seed w%.3f_g%.3f_N%s_gamma%.3f_beta%.6f"
            % (Model.w, Model.g, N, Model.gamma, Model.beta)
            + ".pickle",
        )

        with open(name_1, "rb") as handle:
            DATA1 = pickle.load(handle)
        with open(name_2, "rb") as handle:
            DATA2 = pickle.load(handle)
        DATA.append(mergeDictionary(DATA1, DATA2))


## obtain seeds for previous simulations


random_seed = DATA[0]["seed"]
random_seed = random_seed.reshape(2, 250)

###  reset dictionary to save memory

DATA = []
DATA1 = []
DATA2 = []

############################################################
############################################################
# end= 15000
# end = 2e6
# times = np.append(np.linspace(0, 10, 5000), np.arange(10, end, 1))

for k in [1]:
    part = 0
    for l in [0, 0]:

        # initial conditions

        Model = create_Model(0, 0, k, l, m)
        N = Model.N
        Op = qt.Options()
        Op.nsteps = 50000
        step = 2e4
        samples = 250
        seed = random_seed[part]

        print(
            "S=%s, N=%s, w=%.3f, gamma=%.3f, g=%.3f, beta=%.3f"
            % (samples, Model.N, Model.w, Model.gamma, Model.g, Model.beta),
            " process started",
        )

        psi_0 = qt.tensor(*[qt.basis(2, 1) for j in range(N // 2 + 1)])
        data = {
            "counts_d": [],
            "purity": [],
            "entropy": [],
            "eigenvalues": [],
            "eigenvalues_L": [],
            "norm": [],
        }

        # functions
        ############################################################
        ############################################################
        def evo(args):
            logger = start_logger_if_necessary()
            logger.debug("🏁 Starting evo function")
            b = Model.beta
            psi = psi_0
            T = times
            l, h, d, eig = args[0], args[1], args[2], args[3]
            # min_eps = min(eig)
            # if min_eps < -0.1:
            #     h_g = h + abs(min_eps)
            # else:
            #     h_g = h
            # try:

            #     gibbs = (-Model.beta * h_g).expm()
            #     gibbs = gibbs / gibbs.tr()
            #     time.sleep(1)

            # except Exception as e:
            #     print(gibbs, e)
            logger.debug("🚀 Starting time evolution")
            rho = qt.mesolve(l, psi, T, [], [], options=Op)
            rhot = rho.states
            logger.debug("😎 time evolution finished")
            time.sleep(1)
            logger.debug("🚀 Starting time evolution 2")
            result_d = qt.mesolve(l, psi, T, [], [d], options=Op)
            counts_d = result_d.expect
            logger.debug("😎 time evolution 2 finished")

            purity = []
            entropy = []
            for t in range(len(rhot)):
                # purity
                purity.append((rhot[t] ** 2).tr())
                # entropy
                entropy.append(qt.entropy_vn(rhot[t]))

            purity = np.array(purity)
            entropy = np.array(entropy)

            # Trace distance data

            # A= [r- gibbs for r in rhot]
            # dt=[a.norm() for a in A]
            dt = []
            # rel_ent= [qt.entropy_relative(r,gibbs) for r in rhot]
            rel_ent = []

            # Liouvillian eigenvalues (takes longer than density matrix evolution)
            time.sleep(1)
            logger.debug("🚀  Starting Liouvillian diagonalization")
            egL = eig_L(l)
            logger.debug("😎  Liouvillian diagonalization finished")

            return [purity, entropy, counts_d, dt, egL, rel_ent]

        ############################################################
        ############################################################

        # random Hamiltonian and Liouvillian sampling
        # joblib
        # random_states= Parallel(n_jobs=10, verbose=15)(delayed(batch_data)\
        #                        (Model, step, Op, times) for _ in range (samples) )
        # random_states= np.array(random_states, dtype = object)

        # Qutip parallelization

        # random parameter generator
        random_states = qt.parallel_map(
            batch_data, seed, task_args=(Model, step, Op, times), progress_bar=True
        )
        random_states = np.array(random_states, dtype=object)

        eigen_H = np.take(random_states, 3, axis=1)

        time.sleep(1.2)

        # joblib
        random_evo = Parallel(n_jobs=N_JOBS, verbose=5, pre_dispatch="1.5*n_jobs")(
            delayed(evo)(r) for r in random_states
        )
        random_evo = np.array(random_evo, dtype=object)

        purity = np.take(random_evo, 0, axis=1)
        entropy = np.take(random_evo, 1, axis=1)
        counts = np.take(random_evo, 2, axis=1)
        traced = np.take(random_evo, 3, axis=1)
        eigen_L = np.take(random_evo, 4, axis=1)
        relative_ent = np.take(random_evo, 5, axis=1)

        # save data

        data.update(
            {
                "purity": purity,
                "entropy": entropy,
                "eigenvalues": eigen_H,
                "counts_d": counts,
                "eigenvalues_L": eigen_L,
            }
        )

        base_folder = f"../DataBatchParallel/N_{Model.N}/long_T/loss_gain/S_{samples*2}/p{part+1}/"
        isExist = os.path.exists(base_folder)
        if not isExist:
            os.makedirs(base_folder)
            print("The new directory is created!")

        name_lg = os.path.join(
            base_folder,
            "Long T w%.3f_g%.3f_" % (Model.w, Model.g)
            + "N%i_gamma%.3f_beta%.3f" % (N, Model.gamma, Model.beta)
            + ".pickle",
        )

        with open(name_lg, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logging.debug(
            "S=%s, N=%s, w=%.3f, gamma=%.3f, g=%.3f, beta=%.3f"
            % (samples, Model.N, Model.w, Model.gamma, Model.g, Model.beta),
            " process saved",
        )

        # update seed
        part += 1
