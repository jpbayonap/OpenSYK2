from random import random
from re import L
from numpy import linalg
import itertools
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import sys
import numpy as np
from mpmath import *
import pickle

N_JOBS = 0
w = 1
g_w = np.arange(0, 4.02, 0.02)
# lbd = np.linspace(0, 16, 300)

beta = 0.1
samples = 500
sigma = np.append(np.arange(0, 1, 0.01), np.arange(1, 16, 0.1))
P2 = 0
P3 = 1


# class VALUES:
#     def __init__(self, w, g_w, BETA):
#         # Central fermion strength
#         self.w = w
#         self.g_w = g_w
#         # Interaction strength
#         self.g = self.g_w * self.w
#         self.beta = BETA


def INFTY(*variables):
    w, beta, g, SIGMA, seed = variables
    atai = []
    for G in g:
        for value in seed:
            state = np.random.RandomState(value)
            L = SIGMA * abs(state.randn())
            e0 = (w + L) / 2 - np.sqrt(((w - L) / 2) ** 2 + G**2)
            e1 = (w + L) / 2 + np.sqrt(((w - L) / 2) ** 2 + G**2)
            e = e0 + e1

            C2 = (w - L) / (e1 - e0)
            c2 = 0.5 * (1 + C2)
            s2 = 0.5 * (1 - C2)

            if e0 < -7:

                ATAI = (s2 + np.exp(-beta * (e1 - e0))) / (
                    1 + np.exp(-beta * (e1 - e0))
                )
            if e0 >= -7:
                ATAI = (
                    np.exp(-beta * e)
                    + s2 * (np.exp(-beta * e0))
                    + c2 * (np.exp(-beta * e1))
                ) / (1 + np.exp(-beta * e) + np.exp(-beta * e0) + np.exp(-beta * e1))

            atai.append(ATAI)

    atai = np.array(atai)
    atai = atai.reshape(len(g), len(seed))
    atai_av, atai_var = np.mean(atai, axis=1), np.var(atai, axis=1)
    return [atai_av, atai_var]


def INFTY1(*variables):
    w, beta, g, SIGMA, seed, p2, p3 = variables
    atai = []
    for G in g:
        for value in seed:
            state = np.random.RandomState(value)
            L = SIGMA * abs(state.randn())
            e0 = (w + L) / 2 - np.sqrt(((w - L) / 2) ** 2 + G**2)
            e1 = (w + L) / 2 + np.sqrt(((w - L) / 2) ** 2 + G**2)
            e = e0 + e1
            C2 = (w - L) / (e1 - e0)
            c2 = 0.5 * (1 + C2)
            s2 = 0.5 * (1 - C2)

            if e0 < -7:
                if w - L != 0:
                    ATAI = (s2 + np.exp(-beta * (e1 - e0))) / (
                        1 + np.exp(-beta * (e1 - e0))
                    )

            else:
                if w - L != 0:
                    ATAI = (
                        np.exp(-beta * e)
                        + s2 * (np.exp(-beta * e0))
                        + c2 * (np.exp(-beta * e1))
                    ) / (
                        1 + np.exp(-beta * e) + np.exp(-beta * e0) + np.exp(-beta * e1)
                    )
                if beta >= 100:
                    if w - L == 0 and e0 > 0:
                        ATAI = 0.5 * (p2 + p3)
                    if w - L == 0 and e0 < 0:
                        ATAI = 0.5 * (p2 + p3 + 1)
                if beta < 100:
                    if w - L == 0:
                        ATAI = 0.5 * (p2 + p3 + 1 / (1 + np.exp(beta * e0)))

            atai.append(ATAI)

    atai = np.array(atai)
    atai = atai.reshape(len(g), len(seed))
    atai_av, atai_var = np.mean(atai, axis=1), np.var(atai, axis=1)
    return [atai_av, atai_var]


if __name__ == "__main__":

    random_seed = np.random.randint(np.iinfo(np.int32).max, size=samples)

    DATA = {"mean": [], "variance": []}

    # COUNTS = INFTY(w, beta, w * g_w, sigma[int(sys.argv[1])], random_seed)
    COUNTS = INFTY1(w, beta, w * g_w, sigma[int(sys.argv[1])], random_seed, P2, P3)
    COUNTS_m = np.take(COUNTS, 0, axis=0)

    COUNTS_v = np.take(COUNTS, 1, axis=0)

    DATA.update({"mean": COUNTS_m, "variance": COUNTS_v})
    print(COUNTS_m.size)
    base_folder = f"../DataBatchParallel/N_2/INFTY/OCC/M1/"
    isExist = os.path.exists(base_folder)
    if not isExist:
        os.makedirs(base_folder)
        print("The new directory is created!")

    name_data = os.path.join(
        base_folder,
        "M1 beta%.1f sigma%.2f" % (beta, sigma[int(sys.argv[1])]) + ".pickle",
    )

    with open(name_data, "wb") as handle:
        pickle.dump(DATA, handle, protocol=pickle.HIGHEST_PROTOCOL)
