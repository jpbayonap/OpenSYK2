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

N_JOBS = 10
# N_JOBS = 1
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


st = time.time()


# for k in range(len_k):
for k in [1]:

    part = 0

    # for l in range(len_l):
    # done 0 ,1, 2, 3,4
    for l in [0, 0]:

        # initial conditions

        # Model = create_Model(0, 0, k, l, 0)
        Model = create_Model(0, 0, k, l, 1)
        N = Model.N
        Op = qt.Options()
        Op.nsteps = 50000

        step = 2e4
        # samples = 250
        samples = 1

        print(
            "S=%s, N=%s, w=%.3f, gamma=%.3f, g=%.3f, beta=%.3f"
            % (samples, Model.N, Model.w, Model.gamma, Model.g, Model.beta),
            " process started",
        )

        random_seed = np.random.randint(np.iinfo(np.int32).max, size=samples)
        psi = qt.tensor(*[qt.basis(2, 1) for j in range(N // 2 + 1)])
        data = {
            "counts_d": [],
            "purity": [],
            "entropy": [],
            "eigenvalues": [],
            "L": [],
            "norm": [],
        }
        data_seed = {"seed": []}

        # functions
        ############################################################
        ############################################################
        def evo(args):
            ############################################
    
            logger = start_logger_if_necessary()
            logger.debug("🏁 Starting evo function")
        
            b = Model.beta
            psi = psi_0
            T = times

            l, h, d, eig = args[0], args[1], args[2], args[3]
            min_eps = min(eig)

            if min_eps < -0.1:
                h_g = h + abs(min_eps)
            else:
                h_g = h
            try:
                gibbs = (-Model.beta * h_g).expm()
                gibbs = gibbs / gibbs.tr()
                time.sleep(1)

            except Exception as e:
                print(gibbs, e)
                logger.error(e)

            logger.debug("🚀 Starting time evolution")
            rho = qt.mesolve(
                l, psi, T, [], [], options=Op, progress_bar=TextProgressBar()
            )
            logger.debug("😎 End time evolution")
            rhot = rho.states
            time.sleep(1)
            logger.debug("🚀 Starting time evolution 2")
            result_d = qt.mesolve(
                l, psi, T, [], [d], options=Op, progress_bar=TextProgressBar()
            )
            logger.debug("😎 End time evolution 2")
            counts_d = result_d.expect
            purity = []
            entropy = []
            logger.debug("finishing time evolutions")
            for t in range(len(rhot)):
                # purity
                purity.append((rhot[t] ** 2).tr())
                # entropy
                entropy.append(qt.entropy_vn(rhot[t]))

            purity = np.array(purity)
            entropy = np.array(entropy)

            # Trace distance data
            logger.debug("Starting trace-distance ")
            A = [r - gibbs for r in rhot]
            dt = [a.norm() for a in A]
            rel_ent = [qt.entropy_relative(r, gibbs) for r in rhot]
            logger.debug("Finishing trace-distance ")
            # Liouvillian eigenvalues (takes longer than density matrix evolution)
            time.sleep(1)
            logging.debug("Lindbladian diagonalization")
            egL = eig_L(l)
            logging.debug("diagonalization is finished")

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
            batch_data,
            random_seed,
            task_args=(Model, step, Op, times),
            progress_bar=TextProgressBar(),
        )
        random_states = np.array(random_states, dtype=object)

        eigen_H = np.take(random_states, 3, axis=1)

        time.sleep(1.2)

        # * joblib
       
        random_evo = np.array(random_evo, dtype=object)

        purity = np.take(random_evo, 0, axis=1)
        entropy = np.take(random_evo, 1, axis=1)
        counts = np.take(random_evo, 2, axis=1)
        traced = np.take(random_evo, 3, axis=1)
        eigen_L = np.take(random_evo, 4, axis=1)
        relative_ent = np.take(random_evo, 5, axis=1)

        # ! save data

        data.update(
            {
                "purity": purity,
                "entropy": entropy,
                "norm": traced,
                "eigenvalues": eigen_H,
                "counts_d": counts,
                "eigenvalues_L": eigen_L,
                "KLdiv": relative_ent,
            }
        )

        data_seed.update({"seed": random_seed})

        #############

        base_folder = (
            f"../DataBatchParallel/N_{Model.N}/loss_gain/S_{samples*2}/p{part+1}/"
        )
        isExist = os.path.exists(base_folder)
        if not isExist:
            os.makedirs(base_folder)
            print("The new directory is created!")

        name_seed = os.path.join(
            base_folder,
            "seed w%.3f_g%.3f_" % (Model.w, Model.g)
            + "N%i_gamma%.3f_beta%.3f" % (N, Model.gamma, Model.beta)
            + ".pickle",
        )

        name_lg = os.path.join(
            base_folder,
            "w%.3f_g%.3f_" % (Model.w, Model.g)
            + "N%i_gamma%.3f_beta%.3f" % (N, Model.gamma, Model.beta)
            + ".pickle",
        )

        with open(name_lg, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(name_seed, "wb") as handle:
            pickle.dump(data_seed, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logging.debug(
            "S=%s, N=%s, w=%.3f, gamma=%.3f, g=%.3f, beta=%.3f"
            % (samples, Model.N, Model.w, Model.gamma, Model.g, Model.beta),
            " process saved",
        )

        part += 1

# # * get the end time

# et = time.time()

# # * get the execution time

# elapsed_time = et - st

# logging.debug("Execution time:", elapsed_time, "seconds")


# ! poster:cond-mat_510 凝縮系　user name
