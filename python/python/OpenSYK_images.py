## required Libraries##
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.lib.function_base import kaiser
from PIL import Image
import pickle
from itertools import combinations


from OpenSYK_functions import *

# plt.rcParams['text.usetex'] = True
####################################################################
####################################################################

# Occupation number sampling


def occ_N(
    ax, i, j, k, l, m, DATA, infty, cond, VAR, MU, samples, size, LABEL, type, color
):
    markers = ["solid", "dashed", "dashdot"]
    Model = create_Model(i, j, k, l, m)
    N = Model.N
    # t = times_m
    # t = times_m2
    t = times_m3
    SAMPLES = len(samples)
    occ = DATA
    o = np.array([occ[i] for i in range(SAMPLES)])
    # shape = o.shape
    # o = o.reshape(shape[0], shape[-1])
    av_occ = np.mean(o, axis=0), np.var(o, axis=0)
    av_infty = [np.mean(infty)] * len(t)
    # inft = [infty] * len(t)
    if type == 0:
        # for n in range(samples):
        #     ax.plot(t, o[n], linewidth=0.1, c="grey")

        ax.plot(
            t,
            av_occ[0],
            linewidth=4,
            label=r"$ av(<d^\dagger d>)_{diag}$",
            c=color,
            linestyle="dashed",
        )
    else:
        for n in range(SAMPLES):
            if cond[n]:
                ax.plot(
                    t,
                    abs(o[n]),
                    c="green",
                    linewidth=4,
                    linestyle=markers[n],
                    label=r"$\Delta=%.2f$" % (1 - samples[n]),
                )
                # ax.plot(
                #     t, [infty[n]] * len(t), c="green", linewidth=0.3, linestyle="dashed"
                # )
            else:
                ax.plot(
                    t,
                    abs(o[n] - inft),
                    c="green",
                    linewidth=4,
                    linestyle=markers[n],
                    label=r"$\Delta=%.2f$" % (1 - samples[n]),
                )
                # ax.plot(
                #     t, [infty[n]] * len(t), c="green", linewidth=0.3, linestyle="dashed"
                # )
        # ax.scatter(
        #     t,
        #     av_occ[0],
        #     marker=".",
        #     label=r"$ av(<d^\dagger d>)(t)$",
        #     c=color,
        #     s=20,
        #     alpha=0.3,
        # )
        # ax.plot(
        #     t,
        #     av_infty,
        #     linewidth=1,
        #     label=r"$ steady state$",
        #     c=color,
        #     linestyle="dashed",
        # )
        # ax.set_ylim(0.5*np.mean(infty), 1)
    # title = (
    #     r"$\omega_d=%.1f, g=%.3f, $" % (Model.w, Model.g)
    #     + "N=%s, " % (N)
    #     + r"$J=%.1f, \beta=%.1f$" % (Model.gamma, Model.beta)
    # )

    ax.set_ylabel("Mean Particle Number", fontsize=size)
    ax.set_xlabel(r"$time$ ", fontsize=size)
    ax.set_xscale("log")

    # ax.set_yscale("log")

    # text = (
    #     "samples=%s\n" % (samples)
    #     + r"$\sigma(\lambda_1)=%.2f$" % (VAR)
    #     + "\n"
    #     + r"$\mu=%.2f $" % (MU)
    #     + "\n"
    #     + r"${\rm green} \to |2\theta|> \pi/4$"
    # )
    # text = r"${\rm green} \to |2\theta|> \pi/4$"
    # plt.text(
    #     0.3e-5,
    #     0.6,
    #     text,
    #     fontsize=size,
    # )
    # plt.title(title, fontsize=size)
    # plt.legend(loc="upper right", fontsize=size)
    ax.tick_params(labelsize=size)
    return


def PURITY_im(
    ax, i, j, k, l, m, DATA, cond, VAR, MU, samples, size, LABEL, type, color, g1
):

    Model = create_Model(i, j, k, l, m)
    N = Model.N
    # t = times_m
    # t = times_m2
    t = times_m3

    SAMPLES = len(samples)
    markers = ["solid", "dashed", "dashdot"]
    purity = DATA

    av_p = np.mean(purity, axis=0), np.var(purity, axis=0)
    # av_infty = [np.mean(infty)] * len(t)

    if type == 0:
        for n in range(SAMPLES):
            # if n == 1:
            # ax.vlines(
            #     5 / g1[n],
            #     0,
            #     max(purity[n]),
            #     linestyle=markers[n],
            #     label=r"$\frac{5}{\Gamma_1}$",
            # )

            if cond[n]:
                ax.plot(
                    t,
                    purity[n],
                    linewidth=4,
                    linestyle=markers[n],
                    c="green",
                    label=r"$\Delta=%.2f$" % (1 - samples[n]),
                )
            else:
                ax.plot(
                    t,
                    purity[n],
                    linewidth=4,
                    c="green",
                    linestyle=markers[n],
                    label=r"$\Delta=%.2f$" % (1 - samples[n]),
                )

    else:
        ax.scatter(
            t, av_p[0], marker=".", label="%s" % (LABEL), c=color, s=20, alpha=0.3
        )
        # ax.plot(
        #     t,
        #     av_infty,
        #     linewidth=1,
        #     label=r"$ av(P(t))(\infty)$",
        #     c=color,
        #     linestyle="dashed",
        # )

    # title = (
    #     "w=%.1f, g=%.3f, " % (Model.w, Model.g)
    #     + "N=%s, " % (N)
    #     + r"$J=%.1f, \beta=%.1f$" % (Model.gamma, Model.beta)
    # )

    ax.set_ylabel("  $P(t)$", fontsize=size)
    ax.set_xlabel(r"$time\sqrt{N}$ ", fontsize=size)
    ax.set_xscale("log")

    # text = (
    #     "samples=%s\n" % (samples)
    #     + r"$\sigma(\lambda_1)=%.2f$" % (VAR)
    #     + "\n"
    #     + r"$\mu=%.2f $" % (MU)
    #     + "\n"
    #     + r"${\rm green} \to |2\theta|> \pi/4$"
    # )
    # text = r"${\rm green} \to |2\theta|> \pi/4$"
    # plt.text(
    #     0.5 * 1e-4,
    #     0.6,
    #     text,
    #     fontsize=size,
    # )
    # plt.title(title, fontsize=size)
    plt.legend(loc="upper right", fontsize=size)
    ax.tick_params(labelsize=size)
    return


def Lind_spec(ax, i, j, k, l, m, DATA, cond, VAR, MU, samples, size):
    Model = create_Model(i, j, k, l, m)
    N = Model.N
    gamma = Model.gamma
    eig_L = DATA / gamma
    Re_eig = np.array([np.real(r) for r in eig_L])
    Im_eig = np.array([np.imag(r) for r in eig_L])
    SAMPLES = len(samples)
    markers = ["x", "^", "v"]

    for n in range(SAMPLES):

        if cond[n]:
            plt.scatter(
                Re_eig[n],
                Im_eig[n],
                c="green",
                s=45,
                marker=markers[n],
                label=r"$\Delta=%.2f$" % (1 - samples[n]),
            )
            # plt.scatter(Re_eig, Im_eig, c="green", s=5, alpha=1, marker="x")
        else:
            plt.scatter(
                Re_eig[n],
                Im_eig[n],
                c="blue",
                s=15,
                marker="^",
                label=r"$\Delta=%.2f$" % (1 - samples[n]),
            )

    plt.grid(True)
    plt.ylabel(r"$Im \Omega_n$", fontsize=size)
    plt.xlabel(r"$Re \Omega_n$", fontsize=size)
    plt.tick_params(labelsize=size)

    # title = (
    #     "w=%.1f,g=%.3f, " % (Model.w, Model.g)
    #     + "N=%s, " % (N)
    #     + r"$J=%.1f, \beta=%.1f$" % (Model.gamma, Model.beta)
    # )
    plt.legend(loc="upper right", fontsize=size)
    # plt.title(title, fontsize=size)

    # text = (
    #     "samples=%s\n" % (samples)
    #     + r"$\sigma(\lambda_1)=%.2f$" % (VAR)
    #     + "\n"
    #     + r"$\mu=%.2f $" % (MU)
    #     + "\n"
    #     + r"${\rm green} \to |2\theta|> \pi/4$"
    # )
    # text = r"${\rm green} \to |2\theta|> \pi/4$"
    # plt.text(
    #     -1.5,
    #     5,
    #     text,
    #     fontsize=size,
    # )

    return
