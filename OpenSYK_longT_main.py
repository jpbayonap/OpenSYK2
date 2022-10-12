#Import libraries 
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

############################################################
############################################################

from OpenSYK_functions import *


#merge data 

def mergeDictionary(dict_1, dict_2):
   dict_3 = {**dict_1, **dict_2}
   for key, value in dict_3.items():
       if key in dict_1 and key in dict_2:
               dict_3[key] = np.array([value , dict_1[key]])
               dict_3[key]= dict_3[key].reshape(len(value)*2)
   return dict_3

############################################################
############################################################


#import data 
## import data ##
DATA=[]
i,j= 0,0
# Majorana fermion number
m= 0
#sampling number
samples = 500
for a in range(len_k):
#   for b in range(len_l):
  for b in [0]:
    Model = OpenSYK(P["w"][0], P["gamma/w"][0], P["w/g"][a],\
    P["gamma/beta"][b], P["N"][m],  1 )

    N= Model.N
#     name = 'DataBatchParallel/N_%s/gain/w%.3f_w_g%.3f_'%(Model.N, Model.w, Model.w_g)+'N%.1f_'%(N)\
#               +'gamma_w%.3f_gamma_beta%.6f'%(Model.gamma_w, Model.gamma_beta)
    name_1 = 'DataBatchParallel/N_%s/loss_gain/S_%s/p1/w%.3f_g%.3f_'\
            %(Model.N, samples, Model.w, Model.g)+\
            'N%.1f_'%(N)+'gamma%.3f_beta%.6f'\
            %(Model.gamma, Model.beta)
    name_2 = 'DataBatchParallel/N_%s/loss_gain/S_%s/p2/w%.3f_g%.3f_'\
            %(Model.N, samples, Model.w, Model.g)+\
            'N%.1f_'%(N)+'gamma%.3f_beta%.6f'\
            %(Model.gamma, Model.beta)
#     name_3 = 'DataBatchParallel/N_%s/loss_gain/S_%s/w%.3f_g%.3f_'\
#             %(Model.N, samples, Model.w, Model.g)+\
#             'N%.1f_'%(N)+'gamma%.3f_beta%.3f'\
#             %(Model.gamma, Model.beta)
#     with open(name_3+'.pickle','rb') as handle:
#             DATA3.append(pickle.load( handle))
    with open(name_1+'.pickle','rb') as handle:
      DATA1=pickle.load( handle)
    with open(name_2+'.pickle','rb') as handle:
      DATA2= pickle.load(handle)
    DATA.append(mergeDictionary(DATA1, DATA2 ))


  
## obtain seeds for previous simulations
DATA= np.array(DATA)
random_seed= DATA[0]['seed'] 
random_seed= random_seed.reshape(2,250)
#reset dictionary to save memory
DATA =[]
DATA1=[]
DATA2=[]
############################################################
############################################################
end= 50000
times = np.append(np.linspace(0,10,4000),np.arange(10,end,1))

for k in range(len_k):
    part=0
    for l in [0,0]: 
        
    #initial conditions
        
        
        Model = create_Model(0,0,k,l,0)
        N= Model.N
        Op= qt.Options()
        Op.nsteps= 50000
        step= 2e4
        samples= 10
        seed= random_seed[part]

        print('S=%s, N=%s, w=%.3f, gamma=%.3f, g=%.3f, beta=%.3f'%(samples,\
                Model.N, Model.w, Model.gamma,\
                Model.g, Model.beta)," process started")

        
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
                time.sleep(1)

            except Exception as e:
                print(gibbs,e)

            rho= qt.mesolve(l, psi, t, [], [], options= Op )
            rhot= rho.states
            time.sleep(1)
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
            rel_ent= [qt.entropy_relative(r,gibbs) for r in rhot]
            # Liouvillian eigenvalues (takes longer than density matrix evolution)
            time.sleep(1)
            egL= eig_L(l)
              
            
            return [purity,entropy,counts_d, dt,egL, rel_ent]

        ############################################################
        ############################################################


        # random Hamiltonian and Liouvillian sampling
        # joblib
        # random_states= Parallel(n_jobs=10, verbose=15)(delayed(batch_data)\
        #                        (Model, step, Op, times) for _ in range (samples) )
        # random_states= np.array(random_states, dtype = object)



        # Qutip parallelization

        # random parameter generator 
        random_states = qt.parallel_map(batch_data, seed,\
                            task_args= (Model,step,Op,times), progress_bar= True)
        random_states= np.array(random_states,dtype=object)

        eigen_H= np.take(random_states,3,axis=1)

        time.sleep(1.2)
            
        
        #joblib 
        random_evo= Parallel(n_jobs=7,verbose=5, pre_dispatch='1.5*n_jobs')(delayed(evo)(r) for r in random_states )
        random_evo= np.array(random_evo, dtype= object)

        purity=np.take(random_evo, 0, axis=1)
        entropy=np.take(random_evo, 1, axis=1)
        counts= np.take(random_evo, 2, axis=1)
        traced=np.take(random_evo, 3, axis=1)
        eigen_L= np.take(random_evo, 4, axis=1)
        relative_ent= np.take(random_evo, 5, axis=1)

    
        #save data 

        data.update( {"purity":purity,"entropy":entropy, 'norm':traced,\
                    "eigenvalues":eigen_H, "counts_d":counts,"eigenvalues_L": eigen_L,\
                        'seed':seed, 'KLdiv': relative_ent } )

        name_lg = 'DataBatchParallel/N_%s/loss_gain/S_%s/long_T/p%s/w%.3f_g%.3f_'\
                %(Model.N, samples*2, part+1, Model.w, Model.g)+\
                'N%.1f_'%(N)+'gamma%.3f_beta%.6f'\
                %(Model.gamma, Model.beta)+'.pickle'

        # name_lg = 'DataBatchParallel/N_%s/loss_gain/S_%s/w%.3f_g%.3f_'\
        #         %(Model.N, samples, Model.w, Model.g)+\
        #         'N%.1f_'%(N)+'gamma%.3f_beta%.3f'\
        #         %(Model.gamma, Model.beta)+'.pickle'
        with open(name_lg,'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        print('S=%s, N=%s, w=%.3f, gamma=%.3f, g=%.3f, beta=%.3f'%(samples,\
                Model.N, Model.w, Model.gamma,\
                Model.g, Model.beta)," process saved")

        #update seed
        part += 1

