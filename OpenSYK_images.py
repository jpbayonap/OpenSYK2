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


%run OpenSYK_functions.ipynb

####################################################################
####################################################################

#import data
## import data ##
DATA=[]
i,j= 0,0
# Majorana fermions number
m= -1
#sampling number
s= 1
samples = P["AverageNumber"][s]

for a in range(len_k):
  for b in range(len_l):

    Model = OpenSYK(P["w"][0], P["gamma/w"][0], P["w/g"][a],\
    P["gamma/beta"][b], P["N"][m],  1 )

    N= Model.N
    name = 'DataBatchParallel/N_%s/gain/w%.3f_w_g%.3f_'%(Model.N, Model.w, Model.w_g)+'N%.1f_'%(N)\
              +'gamma_w%.3f_gamma_beta%.6f'%(Model.gamma_w, Model.gamma_beta)
    name_lg = 'DataBatchParallel/N_%s/loss_gain/S_%s/w%.3f_g%.3f_'\
            %(Model.N, samples, Model.w, Model.g)+\
            'N%.1f_'%(N)+'gamma%.3f_beta%.6f'\
            %(Model.gamma, Model.beta)
    with open(name_lg+'.pickle','rb') as handle:
      DATA.append(pickle.load( handle))

DATA = np.array(DATA)
DATA = DATA.reshape(len_k,len_l)
shape= DATA.shape

####################################################################
####################################################################

####################################################################
####################################################################

#legend size
size =20

#Purity sampling 


for x in range(9):
  for y in range(9):  
    Model = create_Model(0,0,x,y,m)
    N= Model.N
    t= times/Model.beta
    
    fig = plt.figure(figsize=(15,15)) # Figureを作成
    ax1 = fig.add_subplot(1,1,1) # Axesを作成
    
    purity_min = 1/2**(N//2+1)
    pu=np.array(DATA[x][y]["purity"])*purity_min
    av= np.mean(pu,axis=0),np.var(pu, axis=0)
    
    eig =np.array(DATA[x][y]["eigenvalues"])
    cond,cond2 = Internal_TR(eig)
    '''
    N=2 condition
    d_eig= np.abs(eig.T[:][0]-eig.T[:][1])
    cond=np.where(d_eig== min(d_eig))
    cond2= np.where(d_eig== max(d_eig))
    d_eig2 = np.delete(d_eig,cond)
    cond3 = np.where(d_eig == min(d_eig2))
    '''
    av_eig=  np.mean(eig, axis=0)  
   #Pty= [P_infty(Model.beta,av_eig)/purity_min]*len(times)
  
    for n in range(samples):
        ax1.plot(t,pu[n],linewidth=0.12, c='grey')
        
    ax1.plot(t,pu[cond][0],linewidth=2, c='red',label= r"$\Delta E_{min}$")
    ax1.plot(t,pu[cond2][0],linewidth=2, c='green',label= r"$\Delta E_{max}$")
    #ax1.plot(times,pu[cond3][0],linewidth=2, c='yellow')

    ax1.errorbar(t, av[0], av[1], capsize=5, fmt='o',\
    markersize=1, ecolor='blue', markeredgecolor = "blue", color='w',label= r'$<P>$')
    #ax1.plot(times,Pty,c='y', linestyle='-.', label=r'$P_\infty$')
    ax1.set_xscale('log')
    
    plt.ylabel("Purity of $\\rho(t)$",fontsize=size)
    plt.xlabel(r"$t/\beta$", fontsize=size)
    
    name_m = "ImagesBatchParallel/Purity/loss_gain/N_%s/S_%s"%(N,samples)+"/Purity "\
    'w%.3f_g%.5f_'%(Model.w, Model.g)+\
        'N%.1f_'%(N)+'gamma%.3f_beta%.6f.jpg'\
         %(Model.gamma, Model.beta)
    title = 'w%.3f_g%.5f_'%(Model.w, Model.g)+\
        'N%.1f_'%(N)+'gamma%.3f_beta%.6f'\
         %(Model.gamma, Model.beta)
    
    ax1.tick_params(labelsize=20)
    plt.title(title, fontsize= size)
    plt.legend(fontsize=size)
    plt.savefig(name_m)
    plt.close(fig)


####################################################################
####################################################################

#Entropy sampling 


for k in range(len_k):
  for l in range(len_l):
    
    Model = create_Model(i,j,k,l,m)
    N= Model.N
    t= times/Model.beta

    fig = plt.figure(figsize=(10,10)) # Figureを作成
    ax1 = fig.add_subplot(1,1,1) # Axesを作成
    
    entropy_max= np.log(2**(N//2+1))
    ent=np.array(DATA[k][l]["entropy"])
    av= np.mean(ent,axis=0),np.var(ent, axis=0)
    eig =np.array(DATA[k][l]["eigenvalues"])
    cond,cond2= Internal_TR(eig)

    
    for n in range(samples):
        ax1.plot(t,ent[n],linewidth=0.12, c='grey')
        
    ax1.plot(t,ent[cond][0],linewidth=2, c='red', label= r"$\Delta E_{min}$")
    ax1.plot(t,ent[cond2][0],linewidth=2, c='green', label= r"$\Delta E_{max}$")
    #ax1.plot(times,ent[cond3][0],linewidth=2, c='yellow')
    
    ax1.errorbar(t, av[0], av[1], capsize=5, fmt='o',\
    markersize=1, ecolor='blue', markeredgecolor = "blue", color='w',label= r'$<S>$')

    #ax1.plot(times, S, c='orange', linewidth= 1, linestyle= "-."\
     #        ,label=r'$ S_\infty $')
   
    plt.ylabel(r"$S(\rho(t))$",fontsize=size)
    plt.xlabel(r"$t/\beta$", fontsize=size)
    ax1.set_xscale('log')
    #ax1.set_yscale('log')
    
    name_lg = "ImagesBatchParallel/Entropy/N_%s/loss_gain/S_%s"%(N,samples)\
    +"/Entropy w%.3f_g%.5f_"%(Model.w, Model.g)+\
        'N%.s'%(N)+'gamma%.3f_beta%.6f.jpg'\
         %(Model.gamma, Model.beta)
    title = "Entropy "+ 'w%.3f_g%.5f_'%(Model.w, Model.g)+\
        'N%.1f_'%(N)+'gamma%.3f_beta%.6f'\
         %(Model.gamma, Model.beta)
    
    plt.title(title, fontsize= size)
    plt.legend(fontsize = size)
    ax1.tick_params(labelsize=20)
    
    plt.savefig(name_lg)
    plt.close(fig)

####################################################################
####################################################################

#Occupation number sampling 

N = P["N"][m]

for k in range(len_k):
  for l in range(len_l):
    
    t= times/Model.beta
    fig = plt.figure(figsize=(10,10)) # Figureを作成
    ax1 = fig.add_subplot(1,1,1) # Axesを作成
    Model = create_Model(i,j,k,l,m)
    occ= np.array(DATA[k][l]["counts_d"])
    occ= occ.reshape(samples, 2390)
    av_occ= np.mean(occ,axis=0), np.var(occ, axis=0)

    eig =np.array(DATA[k][l]["eigenvalues"])
    cond,cond2= Internal_TR(eig)
        
    for n in range(samples):
      ax1.plot(t,occ[n], c='grey',linewidth=0.2)
   
    ax1.errorbar(t, av_occ[0], av_occ[1], capsize=5, fmt='o',\
    markersize=1, ecolor='blue', markeredgecolor = "blue", \
                 color='w',label= r'$<d^\dagger d>$')

    ax1.plot(t,occ[cond][0],linewidth=2, c='red', label= r"$\Delta E_{min}$")
    ax1.plot(t,occ[cond2][0],linewidth=2, c='green', label= r"$\Delta E_{max}$")
    ax1.set_xscale('log')
    title= "Occupation number prob."+'w%.3f_g%.5f_'%(Model.w, Model.g)+\
            'N%.1f_'%(N)+'gamma%.3f_beta%.6f'\
             %(Model.gamma, Model.beta)

    name_lg= "ImagesBatchParallel/OccupationNumber/N_%s"%(N)\
    +"/loss_gain/S_%s/OccupationN "%(samples)+'w%.3f_g%.5f_'%(Model.w, Model.g)+\
    'N%.1f_'%(N)+'gamma%.3f_beta%.6f.jpg'\
             %(Model.gamma, Model.beta)

    plt.ylabel("$d^\dagger d$",fontsize=18)
    plt.xlabel(r"$t/\beta$", fontsize=18)
    plt.title(title, fontsize= size)
    plt.legend(fontsize = size)
    plt.savefig(name_lg)
    plt.close(fig)

####################################################################
####################################################################

##Liouvillian  eigenvalues

for k in range(len_k):
  for l in range(len_l):
    
    fig = plt.figure(figsize=(10,10))
    Model = create_Model(i,j,k,l,m)
    

    eig =np.array(DATA[k][l]["eigenvalues_L"])
    Re_eig= np.real(eig)
    Im_eig= np.imag(eig)
    
    # unusual behavior 
    eigH =np.array(DATA[k][l]["eigenvalues"])
    cond,cond2= Internal_TR(eigH)
    #d_eigH= np.abs(eigH.T[:][0]-eigH.T[:][1])
    #cond=np.where(d_eigH== min(d_eigH))
    
    crystal=np.logical_and(np.real(eig)==0,np.imag(eig)!=0)
    
    plt.scatter(Re_eig, Im_eig, c ="b",s=0.3)
    plt.scatter(Re_eig[cond][0], Im_eig[cond][0], c="red", s=0.3,\
               label= r"$\Delta E_{min}$")
#     plt.scatter(Re_eig[cond2][0], Im_eig[cond2][0], c="y", s=20,\
#                label= r"$\Delta E_{max}$")

    plt.scatter(np.real(eig[crystal]),np.imag(eig[crystal]), c="green", s=0.3)
    
    plt.ylabel("Im"+"$L/\omega$",fontsize=size)
    plt.xlabel("Re"+"$L/\omega$", fontsize=size)
    
    title = "Eig_L "+\
    'w=%.3f g=%.5f_'%(Model.w, Model.g)\
    +'N=%s_'%(N)+r'$\gamma=%.3f \beta=%.6f$'%(Model.gamma, Model.beta)
   

    name_lg= "ImagesBatchParallel/Eigenvalues_L/N_%s"%(N)\
    +"/loss_gain/S_%s/Eigenvalues_L "%(samples)\
    +'w%.3f_g%.5f_'%(Model.w, Model.g)+\
     'N%.s_'%(N)+'gamma%.3f_beta%.6f.jpg'\
             %(Model.gamma, Model.beta)
    plt.title( title, fontsize = size )
    
    plt.legend(fontsize=size)
    plt.savefig(name_lg)
    plt.close(fig)


####################################################################
####################################################################

#Trace distance



for k in range(len_k):
    print(k)
    for l in range(len_l):

        fig, (ax1,ax2) = plt.subplots(2,1, figsize=(16,18))
        Model = create_Model(i,j,k,l,m)
        N = Model.N
        t= times/Model.beta
        eig =np.array(DATA[k][l]["eigenvalues"])
        cond,cond2 = Internal_TR(eig)
        norm_m= np.array(DATA[k][l]["norm min"])
        norm_G = np.array(DATA[k][l]["norm gibbs"])
        av_m= np.mean(norm_m,axis=0),np.var(norm_m, axis=0)
        av_G= np.mean(norm_G,axis=0),np.var(norm_G, axis=0)

        for n in range(samples):
          ax1.plot(t,norm_m[n],linewidth=0.1,c="grey")
        ax1.plot(t,norm_m[cond[0]],linewidth=2,c="red", label='min')
        ax1.plot(t,norm_m[cond2[0]],linewidth=2,c="green", label='max')

        ax1.plot(t, av_m[0], c='y', linewidth= 2, label="<S>")

        ax1.errorbar(t,av_m[0],av_m[1], linestyle='-.')
        ax1.set_ylabel(r"$|\rho(t)- \rho_{max}| $",fontsize=18)
        ax1.set_xlabel(r"$t/\beta$", fontsize=18)
        title = "Norm "+ 'w=%.3f, g=%.5f_'%(Model.w, Model.g)+\
                'N=%s'%(N)+', beta=%.6f'\
                 %(Model.beta)

        ax1.set_title( title, fontsize = size )
        ax1.set_yscale('log')
        ax1.set_xscale('log')

        ax1.legend(fontsize=15)

        for n in range(samples):
          ax2.plot(t,norm_G[n],linewidth=0.1,c="grey")
        ax2.plot(t,norm_G[cond[0]],linewidth=2,c="red", label='min')
        ax2.plot(t,norm_G[cond2[0]],linewidth=2,c="green", label='max')

        ax2.plot(t, av_G[0], c='y', linewidth= 2, label="<S>")

        ax2.errorbar(t,av_G[0], av_G[1], linestyle='-.')
        ax2.set_ylabel(r"$|\rho(t)- \rho_{G}| $",fontsize=18)
        ax2.set_xlabel(r"$t/\beta$", fontsize=18)
        ax2.set_title( title, fontsize = size )
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        ax2.legend(fontsize=15)


        name_lg= "ImagesBatchParallel/Norm/N_%s"%(N)\
        +"/loss_gain/S_%s/Norm "%(samples)\
        +'w%.3f_g%.5f_'%(Model.w, Model.g)+\
         'N%.s_'%(N)+'gamma%.3f_beta%.6f.jpg'\
                 %(Model.gamma, Model.beta)



        plt.savefig(name_lg)
        plt.close(fig)

####################################################################
####################################################################



# Multiple quantities plot

for k in range(len_k):
  print(k)
  for l in range(len_l):
    t= times/Model.beta
    Model = create_Model(i,j,k,l,m)
    N = Model.N
    eig =np.array(DATA[k][l]["eigenvalues"])
    cond,cond2 = Internal_TR(eig)
    fig, ((ax1,ax2),(ax3,ax4), (ax5,ax6)) = plt.subplots(3,2, figsize=(16,22))

    # Purity
    purity_min = [1/2**(N//2+1)]*len(times)
    pu=np.array(DATA[k][l]["purity"])
    av= np.mean(pu,axis=0),np.var(pu, axis=0)
 #   av_purity=  (np.array(DATA[k][l]["purity"]).sum(axis=0))/(50*purity_min[0]) 

    for n in range(samples):
         ax1.plot(t,pu[n],linewidth=0.12, c='grey')      
    ax1.plot(t,pu[cond][0],linewidth=2, c='red',label= r"$\Delta E_{min}$")
    ax1.plot(t,pu[cond2][0],linewidth=2, c='green',label= r"$\Delta E_{max}$")
    #ax1.plot(times,pu[cond3][0],linewidth=2, c='yellow')

    ax1.errorbar(t, av[0], av[1], capsize=5, fmt='o',\
    markersize=1, ecolor='blue', markeredgecolor = "blue", color='w',label= r'$<P>$')
    #ax1.plot(times,Pty,c='y', linestyle='-.', label=r'$P_\infty$')
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$t/\beta$', fontsize=18)
    ax1.set_ylabel(r'$P(t)/P_{min}$', fontsize=15)
    ax1.set_title('Purity',fontsize=21)
    ax1.legend(fontsize= size)

    #Von Neumann entropy
    entropy_max= np.log(2**(N//2+1))
    ent=np.array(DATA[k][l]["entropy"])
    ent= ent/entropy_max
    av_e= np.mean(ent,axis=0),np.var(ent, axis=0)

    for q in range(samples):
        ax2.plot(t,ent[q],linewidth=0.12, c='grey')

    ax2.plot(t,ent[cond][0],linewidth=2, c='red',label= r"$\Delta E_{min}$")
    ax2.plot(t,ent[cond2][0],linewidth=2, c='green',label= r"$\Delta E_{max}$")
    ax2.errorbar(t, av_e[0], av_e[1], linestyle='-.',label= r'$<P>$')

    ax2.plot(t,av_e[0], c='orange',linewidth= 2,label= r'$<P>$')

    ax2.set_xscale('log')
    ax2.set_ylabel("$S(t)/S_{max}$",fontsize=18)
    ax2.set_xlabel(r"$t/\beta$", fontsize=18)
    ax2.set_title("Von Neumann entropy ",fontsize=21)
    ax2.legend(fontsize= size)


    #central fermion population
    cd=np.array(DATA[k][l]["counts_d"])
    cd.shape
    samples
    cd= cd.reshape(samples, 2390)
    av_cd= np.mean(cd,axis=0),np.var(cd, axis=0)

    for r in range(samples):
        ax3.plot(t,cd[r],linewidth=0.12, c='grey')

    ax3.plot(t,cd[cond][0],linewidth=2, c='red',label= r"$\Delta E_{min}$")
    ax3.plot(t,cd[cond2][0],linewidth=2, c='green',label= r"$\Delta E_{max}$")
    ax3.errorbar(t, av[0], av[1], linestyle='-.',label= r'$<P>$')

    ax3.plot(t,av[0], c='orange',linewidth= 2,label= r'$<P>$')

    ax3.set_xscale('log')
    # ax1.set_yscale('log')

    ax3.set_ylabel("  $d^\dagger d$",fontsize=18)
    ax3.set_xlabel(r"$t/\beta$", fontsize=18)
    ax3.set_title("Occupation number prob.",fontsize=21)
    ax3.legend(fontsize = size)
    

    #Eigen values of the Liouvillian superoperator
    w = Model.w
    eig =np.array(DATA[k][l]["eigenvalues_L"])/w
    Re_eig= np.real(eig)
    Im_eig= np.imag(eig)
    
    ax4.scatter(Re_eig, Im_eig, c ="b",s=10)
    ax4.set_ylabel("Im"+r"$\frac{L}{\omega}$" ,fontsize=18)
    ax4.set_xlabel("Re"+r"$\frac{L}{\omega}$" , fontsize=18)
    ax4.set_title("Eigenvalue distribution of L ", fontsize=21)
    
    
    #norm distance
    
    norm_m= np.array(DATA[k][l]["norm min"])
    norm_G = np.array(DATA[k][l]["norm gibbs"])
    av_m= np.mean(norm_m,axis=0),np.var(norm_m, axis=0)
    av_G= np.mean(norm_G,axis=0),np.var(norm_G, axis=0)

    for s in range(samples):
      ax5.plot(t,norm_m[s],linewidth=0.1,c="grey")
    ax5.plot(t,norm_m[cond[0]],linewidth=2,c="red", label='min')
    ax5.plot(t,norm_m[cond2[0]],linewidth=2,c="green", label='max')

    ax5.plot(t, av_m[0], c='y', linewidth= 2, label="<S>")

    ax5.errorbar(t,av_m[0],av_m[1], linestyle='-.')
    ax5.set_ylabel(r"$|\rho(t)- \rho_{max}| $",fontsize=18)
    ax5.set_xlabel(r"$t/\beta$", fontsize=18)
    title = "Norm "+ r"$|\rho(t)- \rho_{max}| $"

    ax5.set_title( title, fontsize = size )
    ax5.set_yscale('log')
    ax5.set_xscale('log')
    ax5.legend(fontsize=15)

    for x in range(samples):
      ax6.plot(t,norm_G[x],linewidth=0.1,c="grey")
    ax6.plot(t,norm_G[cond[0]],linewidth=2,c="red", label='min')
    ax6.plot(t,norm_G[cond2[0]],linewidth=2,c="green", label='max')
    title_G = "Norm "+ r"$|\rho(t)- \rho_{G}| $"
    ax6.plot(t, av_G[0], c='y', linewidth= 2, label="<S>")

    ax6.errorbar(t,av_G[0], av_G[1], linestyle='-.')
    ax6.set_ylabel(r"$|\rho(t)- \rho_{G}| $",fontsize=18)
    ax6.set_xlabel(r"$t/\beta$", fontsize=18)
    ax6.set_title( title_G, fontsize = size )
    ax6.set_yscale('log')
    ax6.set_xscale('log')
    ax6.legend(fontsize=15)


  
    title = 'Quantities '+r'$\omega=%.3f$, '%(Model.w)+r'$\frac{\omega}{g}=$'+'%.3f, '%(Model.w_g)\
                +'N=%.1f, '%(N)+r'$\frac{J}{\omega}=$'+'%.3f, '%(Model.gamma_w)+ \
     r'$\frac{J}{\beta}=$'+'%.6f'%(Model.gamma_beta)
    
    name_g = 'ImagesBatchParallel/Quantities/N_%s/'%(N)\
                +'gain/Quantities w%.3f_w_g%.3f_'%(Model.w, Model.w_g)\
                +'N%.1f_'%(N)+'gamma_w%.3f_gamma_beta%.6f.jpg'%(Model.gamma_w, Model.gamma_beta)
    
    name_lg = 'ImagesBatchParallel/Quantities/N_%s/'%(N)\
                +'loss_gain/S_%s'%(samples)+'/Quantities w%.3f_w_g%.3f_'%(Model.w, Model.w_g)\
                +'N%.1f_'%(N)+'gamma_w%.3f_gamma_beta%.6f.jpg'%(Model.gamma_w, Model.gamma_beta)
    fig.suptitle(title, fontsize=23)
    plt.savefig(name_lg)
    plt.close(fig)
    
    
    
    







