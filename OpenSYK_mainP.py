#Import libraries 
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
############################################################
############################################################

from OpenSYK_functions import *

for k in range(len_k):

    for l in range(len_l):

    #initial conditions
        Model = create_Model(0,0,k,l,0)
        N= Model.N
        Op= qt.Options()
        Op.nsteps= 50000
        step= 2e4
        samples= 350
        psi_0= qt.tensor(*[qt.basis(2,1) for j in range(N//2+1)])
        data= { "counts_d":[], "purity":[], "entropy":[], "eigenvalues":[], "L":[],\
            'norm':[]}

        # functions
        ############################################################
        ############################################################
        def evo(args):
            b= Model.beta
            psi= psi_0
            t= times
            l,h,d, eig = args[0],args[1], args[2], args[3]
            min_eps= min(eig)
            if min_eps < -0.1:
                h_g= h+ abs(min_eps)
            else:
                h_g= h
            try:
                gibbs = (-Model.beta*h_g).expm()
                gibbs= gibbs/gibbs.tr()
            except Exception as e:
                print(gibbs,e)

            rho= qt.mesolve(l, psi, t, [], [], options= Op )
            rhot= rho.states
            result_d= qt.mesolve(l, psi, t, [], [d], options= Op)
            counts_d = result_d.expect
            purity=[]
            entropy=[]
            for t in range(len(rhot)):
            # purity
                purity.append((rhot[t]**2).tr())
            #entropy
                entropy.append(qt.entropy_vn(rhot[t]))

            purity = np.array(purity)
            entropy = np.array(entropy)

            #Trace distance data
            A= [r- gibbs for r in rhot]
            dt=[a.norm() for a in A]  
            

            return [purity,entropy,counts_d, dt]

        ############################################################
        ############################################################


        # random Hamiltonian and Liouvillian sampling
        # joblib
        # random_states= Parallel(n_jobs=10, verbose=15)(delayed(batch_data)\
        #                        (Model, step, Op, times) for _ in range (samples) )
        # random_states= np.array(random_states, dtype = object)



        # qutip

        # random parameter generator 
        random_states = qt.parallel_map(batch_data,np.arange(samples),\
                            task_args= (Model,step,Op,times), progress_bar= True)
        random_states= np.array(random_states,dtype=object)

        eigen_H= np.take(random_states,3,axis=1)
        # D_f= np.take(random_states,2,axis=1)
        # Hamiltonians= np.take(random_states,1,axis=1)
       # Liouvillians= np.take(random_states,0,axis=1)
        #input=np.array([ (L,H,D,E) for L,H,D,E in zip(Liouvillians,Hamiltonians,D_f) ], dtype= object)

        # random physical quantities
        random_evo = qt.parallel_map(evo,random_states,progress_bar= True)
        random_evo= np.array(random_evo, dtype= object)
        rand_eigL= qt.parallel_map(eig_L,Liouvillians)

        purity=np.take(random_evo, 0,axis=1)
        entropy=np.take(random_evo, 1,axis=1)
        counts= np.take(random_evo,2,axis=1)
        traced=np.take(random_evo, 3,axis=1)

        print('S=%s, N=%s, w=%.3f, gamma=%.3f, g=%.3f, beta=%.3f'%(samples,\
                Model.N, Model.w, Model.gamma,\
                Model.g, Model.beta)," process saved")

        #save data 

        data.update({"purity":purity,"entropy":entropy, 'norm':traced,\
                    "eigenvalues":eigen_H, "counts_d":counts,"eigenvalues_L":rand_eigL })

        name_lg = 'DataBatchParallel/N_%s/loss_gain/S_%s/w%.3f_g%.3f_'\
                %(Model.N, samples, Model.w, Model.g)+\
                'N%.1f_'%(N)+'gamma%.3f_beta%.6f'\
                %(Model.gamma, Model.beta)+'.pickle'
        with open(name_lg,'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        




