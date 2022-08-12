#Import libraries 
import qutip as qt
import numpy as np
from numpy import linalg
import itertools
from qutip.qip.operations import swap
import pickle
import json 
import os
from multiprocessing import Pool
############################################################
############################################################

%run OpenSYK_functions.ipynb


'''
Time evolution for quantum master equation in Linddblad form
'''

#linear sampling
# w, w/Gamma, N value
i,j,m= 0,0,-1
#sampling number
s= 1
samples = P["AverageNumber"][1]
#integration steps
Op= qt.Options()
Op.nsteps= 10000


for k in range(len_k):

  for l in range(len_l):
    
    Model = create_Model(i,j,k,l,m)
    data= { "counts_d":[], "purity":[], "entropy":[], "eigenvalues":[], "eigenvalues_L":[],\
    "norm min":[],"norm gibbs":[]  }
    N = Model.N
    entropy_max= np.log(2**(N//2+1))
    purity_min = 1/2**(N//2+1)
    for s in range (samples):
        
      purity=[]
      entropy=[]
      #random variables   
      H, eig, lamb, coupl_c, coupl_cm, quasi_norms, f =  Model.Hamiltonian1
        
      # time evolution for the system
      psi_0 = qt.tensor(*[qt.basis(2,1) for j in range(N//2+1)]) 
      result_d, rho,  eig_L, L = solve2(times, coupl_c, coupl_cm, f, N, H, quasi_norms,psi_0 ,Op)
      counts_d= result_d.expect
      rhot = rho.states
    
      # Maximally entangled state
      r_mi = qt.Qobj(np.eye(2**(N//2+1))/(2**(N//2+1)), dims= H.dims)
    
     # Gibbs state
      g = (-Model.beta*H).expm()
      g= g/g.tr()
      g= qt.Qobj(np.nan_to_num(np.real(g.full()),nan=1),dims= H.dims)
     
      for n in range(len(rhot)):
          # purity
          purity.append((rhot[n]**2).tr())
          #entropy
          entropy.append(qt.entropy_vn(rhot[n]))
            
      
      purity = np.array(purity)
      entropy = np.array(entropy)
      
    
      A= [r-r_mi for r in rhot]
      B= [r-g for r in rhot]
      dt_A=[a.norm() for a in A] 
      dt_B= [b.norm() for b in B] 

    
      #save data
      data["counts_d"].append(counts_d)
      data["eigenvalues"].append(eig)
      data["eigenvalues_L"].append(eig_L)
     # data["Lindbladian"].append(L)
      data["purity"].append(purity)
      data["entropy"].append(entropy)
      data['norm min'].append(dt_A)
      data['norm gibbs'].append(dt_B)
      


    print('S=%s, N=%s, w=%.3f, gamma=%.3f, g=%.3f, beta=%.3f'%(samples,\
              Model.N, Model.w, Model.gamma,\
              Model.g, Model.beta)," process saved")
#     name = 'DataBatchParallel/N_%s/w%.3f_w_g%.3f_'%(Model.N, Model.w, Model.w_g)+\
#         'N%.1f_'%(N)+'gamma_w%.3f_gamma_beta%.6f'\
#          %(Model.gamma_w, Model.gamma_beta)+'.pickle'
    
    # loss and gain included 
    name_lg = 'DataBatchParallel/N_%s/loss_gain/S_%s/w%.3f_g%.3f_'\
            %(Model.N, samples, Model.w, Model.g)+\
            'N%.1f_'%(N)+'gamma%.3f_beta%.6f'\
            %(Model.gamma, Model.beta)+'.pickle'
    with open(name_lg,'wb') as handle:
      pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 

'''
Time evolution for general Liouvillian operator 

'''

i,j,m= 0,0,-1
step = 4e7
samples = P["AverageNvumber"][1]
#integration steps
Op= qt.Options()
Op.nsteps= 10000

for k in range(len_k):

  for l in range(len_l):
    
    Model = create_Model(i,j,k,l,m)
    data= { "counts_d":[], "purity":[], "entropy":[], "eigenvalues":[], "eigenvalues_L":[],\
     'norm':[]}
    data2= { "counts_d":[], "purity":[], "entropy":[], "eigenvalues_L":[],\
     'norm':[]}
    data3= { "counts_d":[], "purity":[], "entropy":[], "eigenvalues_L":[],\
     'norm':[]}
    N = Model.N
    psi_0 =qt.tensor(*[qt.basis(2,1) for p in range(N//2+1)])
    # Maximally entangled state
    r_max = qt.Qobj(np.eye(2**(N//2+1))/(2**(N//2+1)), dims= H.dims)
    
    for sample in range (samples):
      if not sample%100:
        print('omg, only %s samples'%(sample))

      purity=[]
      entropy=[]
      purity2=[]
      entropy2=[]
      purity3=[]
      entropy3=[]  
      #random variables  Liouvilians
      L ,LS, LG, D,eps, coupl, H = Model.Liouvilian(step)
      eig_L= L.eigenenergies()
      eig_LS= LS.eigenenergies()
      eig_LG= LG.eigenenergies()
      G=(-Model.beta*H).expm()
      g= G/G.tr()
      #g_vec= qt.operator_to_vector(g)
      g= qt.Qobj(np.nan_to_num(np.real(g.full()),nan=1),dims= H.dims)
      

      # time evolution for the system

        # with RWA
      result_d= qt.mesolve(L, psi_0, times, [], [D],options=Op )
      counts_d = result_d.expect
      rho= qt.mesolve(L, psi_0, times, [], [], options=Op)
      rhot= rho.states

        #without RWA
      result_d2= qt.mesolve(LS, psi_0, times, [], [D], options=Op)
      counts_d2 = result_d2.expect
      rho2= qt.mesolve(LG, psi_0, times, [],[], options=Op)
      rhot2= rho2.states

        #General
      result_d3= qt.mesolve(LG, psi_0, times, [], [D],options=Op)
      counts_d3 = result_d3.expect
      rho3= qt.mesolve(LG, psi_0, times, [],[], options=Op)
      rhot3= rho3.states


      for n in range(len(rhot)):
          # purity
          purity.append((rhot[n]**2).tr())
          #entropy
          entropy.append(qt.entropy_vn(rhot[n]))

         # purity without RWA
          purity2.append((rhot2[n]**2).tr())
          #entropy  without RWA
          entropy2.append(qt.entropy_vn(rhot2[n]))  

        # general
          purity3.append((rhot3[n]**2).tr())
          #entropy  without RWA
          entropy3.append(qt.entropy_vn(rhot3[n]))  


      purity = np.array(purity)
      entropy = np.array(entropy)
      purity2 = np.array(purity2)
      entropy2 = np.array(entropy2)
      purity3 = np.array(purity3)
      entropy3 = np.array(entropy3)  

    #Trace distance data
      A= [r- g for r in rhot]
      dt=[a.norm() for a in A]  
      A2= [r2-g for r2 in rhot2]
      dt2=[a2.norm() for a2 in A2]
      A3= [r3-g for r3 in rhot3]
      dt3=[a3.norm() for a3 in A3]

      #save data
      data["counts_d"].append(counts_d)
      data["eigenvalues"].append(eps)
      data["eigenvalues_L"].append(eig_L)
      data["purity"].append(purity)
      data["entropy"].append(entropy)
      data['norm'].append(dt)

      data2["counts_d"].append(counts_d2)
      data2["eigenvalues_L"].append(eig_LS)
      data2["purity"].append(purity2)
      data2["entropy"].append(entropy2)
      data2['norm'].append(dt2)

      data3["counts_d"].append(counts_d3)
      data3["eigenvalues_L"].append(eig_LG)
      data3["purity"].append(purity3)
      data3["entropy"].append(entropy3)
      data3['norm'].append(dt3)



    print('S=%s, N=%s, w=%.3f, gamma=%.3f, g=%.3f, beta=%.3f'%(samples,\
              Model.N, Model.w, Model.gamma,\
              Model.g, Model.beta)," process saved")

    
    # loss and gain included 
    name_L = 'DataBatchParallel/N_%s/loss_gain/S_%s/L/w%.3f_g%.3f_'\
            %(Model.N, samples, Model.w, Model.g)+\
            'N%.1f_'%(N)+'gamma%.3f_beta%.6f'\
            %(Model.gamma, Model.beta)+'.pickle'
    name_LS=  'DataBatchParallel/N_%s/loss_gain/S_%s/LS/w%.3f_g%.3f_'\
            %(Model.N, samples, Model.w, Model.g)+\
            'N%.1f_'%(N)+'gamma%.3f_beta%.6f'\
            %(Model.gamma, Model.beta)+'.pickle'
        
    name_LG ='DataBatchParallel/N_%s/loss_gain/S_%s/LG/w%.3f_g%.3f_'\
            %(Model.N, samples, Model.w, Model.g)+\
            'N%.1f_'%(N)+'gamma%.3f_beta%.6f'\
            %(Model.gamma, Model.beta)+'.pickle'
    with open(name_L,'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    with open(name_LS,'wb') as handle:
        pickle.dump(data2, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    with open(name_LG,'wb') as handle:
        pickle.dump(data3, handle, protocol=pickle.HIGHEST_PROTOCOL) 



 


