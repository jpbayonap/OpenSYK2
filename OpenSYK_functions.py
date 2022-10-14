# Parameters 
"""
Functions implemented for the simulations
"""
import numpy as np
import qutip as qt
import os
from itertools import combinations
import matplotlib.pyplot as plt
from scipy import integrate
from mpmath import *

### Parameters ###

P = {"w": [1e-1,1,1e1],"gamma/w":[1,1e1,1e2], "w/g": np.array([1e-1, 2e-1, 1, 2, 1e1]),\
     "N":[10,4,2], \
     "gamma/beta" : np.array([1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]),\
     "AverageNumber":[350,500]}

len_k= len(P["w/g"])
len_l= len(P["gamma/beta"])
#Evolution time
end= 90000
# end = 15000

#times = np.append(np.linspace(0,10,4000),np.arange(10,end,1))
times= np.append(np.linspace(0,10,5000),np.arange(10,end,1))
times.size
#############################################################################

#############################################################################
'''
Open SYK class:
generates the diagonalized Hamiltonian of the system, and
several Lindbladian supeoperators for the RWA regime, RWA + Lamb shift
and generalized master equation.

'''
#System variables

# w = 1e-1 default value
# gamma = 1e-1 default value

class OpenSYK():
  def __init__(self, w, gamma_w, w_g, gamma_beta, N, model):
    # Model
    self.Model= model
   
    # Central fermion strength 
    self.w= w
    self.w_g= w_g

    # Interaction strength
    self.g = (1/self.w_g)* self.w

    # Majorana fermions number
    self.N= N

    # Coupling function strength 
    self.gamma_w= gamma_w
    self.gamma= self.w*self.gamma_w
    self.gamma_beta = gamma_beta
    # Bath temperature
    self.beta = self.gamma*(1/self.gamma_beta)

    #number of fermions = [complex fermions + central fermion(model 1), complex fermions(model 2)]
    
    self.n1 = self.N//2+1
    self.n2 = self.N//2


  
  #Define fermion operators using the Jordan-Wigner transform (n or N//2)
  @property
  def fermion_operators(self):
    if self.Model==1:
      return [qt.tensor(*[qt.destroy(2) if i == j\
                  else (qt.sigmaz() if j < i\
                      else qt.identity(2))\
                          for j in range(self.n1)])\
                              for i in range(self.n1)]
    if self.Model==2:
      return [qt.tensor(*[qt.destroy(2) if i == j\
                  else (qt.sigmaz() if j < i\
                      else qt.identity(2))\
                          for j in range(self.n2)])\
                              for i in range(self.n2)]

  ### Lamb shift ###
  def LambS(self,arg, step):
    threshold = 5e-7
    Sp = lambda x : (1+np.tanh(x+arg))/x -(1+np.tanh(-x+arg))/x
    Sm = lambda x : (1-np.tanh(x+arg))/x -(1-np.tanh(-x+arg))/x
    
    Sp_sing, Sp_singerr= integrate.quad(Sp,0,step)
    Sp_pos,  Sp_poserr = integrate.quad(Sp,step,step**2)
    Sm_sing, Sm_singerr= integrate.quad(Sm,0,step)
    Sm_pos,  Sm_poserr = integrate.quad(Sm,step,step**2)
    Sp_err= Sp_singerr+ Sp_poserr
    Sm_err = Sm_singerr + Sm_poserr
    
    Sp_result = -( Sp_sing + Sp_pos )/(2*np.pi)
    Sm_result = -( Sm_sing + Sm_pos )/(2*np.pi)
    if Sp_err > threshold or Sm_err> threshold:
        print('Absolute error of integration is larger than the tolerance.'\
              , Sp_err, Sm_err)
    return Sp_result, Sm_result


    
    
  #Hamiltonian of the system SYK+central fermion
  @property
  def Hamiltonian1(self,seed):
    N = self.N
    c_fermi = self.fermion_operators
    wg = np.zeros(self.n1)
    wg[0] = self.w/2 # isolated fermion energy
    wg[1:]= self.g   # interaction strength

    #random interaction matrix 
    
    state=np.random.RandomState(seed)
    K = state.randn(N,N) 
    K = 0.5*(K - K.T)

    # K_ab eigenenergies
    lbd= np.linalg.eigvals(K)
    lbd=[np.imag(lbd)[x] for x in range(0,len(lbd),2)]

    #define M
    M=np.zeros((self.n1)**2)
    M= M.reshape(self.n1,self.n1)
    M[0]= wg
    for x in range(self.n2): 
      M[x+1][x+1] =lbd[x]/2
    M = M + M.T
    epsilon = np.linalg.eigvals(M)

    #coupling constants and quasifermions normalizations constants
    
    norms= np.sqrt([1+self.g**2*sum(1/(lbd[i]-epsilon[j])**2 for i in range(len(lbd)))\
                    for j in range(len(epsilon))])
    random_c = np.array([1+self.g*sum(1/(lbd[i]-epsilon[j]) for i in range(len(lbd)))\
                    for j in range(len(epsilon))])
    random_c = random_c/norms
    random_cps = np.array([self.gamma*(1+np.tanh(0.5*self.beta*epsilon[i]))*(random_c[i])**2 \
                 for i in range(len(random_c))])
    # high float precision
    exponents= np.array([exp(-self.beta*e) for e in epsilon])
    random_cps_m = exponents*random_cps
#     random_cps_m =[self.gamma*(1-np.tanh(0.5*self.beta*epsilon[i]))*(random_c[i])**2 \
#                   for i in range(len(random_c))]
    
    # diagonalized Hamiltonian
   #MISTAKE H_d = sum([ epsilon[i]*(c_fermi[i].dag()*c_fermi[i]) for i in range(self.n2)]) 
    H_d = sum([ epsilon[i]*(c_fermi[i].dag()*c_fermi[i]) for i in range(self.n1)]) 
    return  H_d, epsilon, lbd, random_cps, random_cps_m, norms, c_fermi
  
    
    
  #@property
  # Liovillian for generalized dissipator
  def Liouvilian(self,step,seed):
    N = self.N
    c_fermi = self.fermion_operators
    wg = np.zeros(self.n1)
    wg[0] = self.w/2 # isolated fermion energy
    wg[1:]= self.g   # interaction strength

    #random interaction matrix 
    #--------------------------------#
   # np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    #--------------------------------#
    state=np.random.RandomState(seed)
    K = state.randn(N,N) 
    K = 0.5*(K - K.T)

    # K_ab eigenenergies
    lbd= np.linalg.eigvals(K)
    lbd=[np.imag(lbd)[x] for x in range(0,len(lbd),2)]

    #define M
    M=np.zeros((self.n1)**2)
    M= M.reshape(self.n1,self.n1)
    M[0]= wg
    for x in range(self.n2): 
      M[x+1][x+1] =lbd[x]/2
    M = M + M.T
    epsilon = np.linalg.eigvals(M)

    #decay rates and quasifermions normalizations constants
    
    norms= np.sqrt([1+self.g**2*sum(1/(lbd[i]-epsilon[j])**2 for i in range(len(lbd))) for j in range(len(epsilon))])
    random_c = np.array([1+self.g*sum(1/(lbd[i]-epsilon[j]) for i in range(len(lbd))) for j in range(len(epsilon))])
    random_c = random_c/norms
    d = [c_fermi[i]/norms[i] for i in range(N//2+1)]
    D = sum(d)
    D_op= D.dag()*D  
    # jump operators
    J_op = [random_c[i]*c_fermi[i] for i in range(N//2+1)]
    # decay rates
    random_cps = np.array([self.gamma*(1+np.tanh(0.5*self.beta*epsilon[i])) 
                 for i in range(N//2+1)])
    # high float precision
    exponents= np.array([exp(-self.beta*e) for e in epsilon])
    random_cps_m = exponents*random_cps
#     random_cps_m =[self.gamma*(1-np.tanh(0.5*self.beta*epsilon[i])) \
#                   for i in range(N//2+1)]
    
    
    # diagonalized Hamiltonian
 # MISTAKE  H_d = sum([ epsilon[i]*(c_fermi[i].dag()*c_fermi[i]) for i in range(self.n2)]) 
    H_d = sum([ epsilon[i]*(c_fermi[i].dag()*c_fermi[i]) for i in range(self.n1)]) 
     
    # Liouvillian    
    L = -1.0j * (qt.spre(H_d) - qt.spost(H_d))
    
    
    ###  RWA terms ###
    #Lamb shift term with RWA
    LS= L
    for op in range(len(J_op)):
        J= J_op[op]
        Jd_J= J.dag()*J
        J_Jd = J*J.dag()
        g_p= random_cps[op]
        g_m = random_cps_m[op]
        S_p, S_m= self.LambS(0.5*self.beta*epsilon[op],step)
        S_p, S_m= random_c[op]*S_p, random_c[op]*S_p
        
        D_RWA= (qt.spre(J)*qt.spost(J.dag()) -0.5*qt.spre(Jd_J)\
              - 0.5*qt.spost(Jd_J))*g_p \
            +(qt.spre(J.dag())*qt.spost(J) -0.5*qt.spre(J_Jd)\
             -  0.5*qt.spost(J_Jd))*g_m 

        L += D_RWA
        #
        LS += -1.0j * (qt.spre(S_p*Jd_J - S_m*J_Jd) - qt.spost(S_p*Jd_J - S_m*J_Jd))
    ### RWA Liouvillian ###
    L_RWA= L 
    ### RWA plus lamb shift Liouvillian ###
    #L_RWAS = L+ LS
    
    '''
     Evolution without RWA
    '''
    ### General Liouvillian ###
#     L_G = L_RWAS
    
#     # phase terms for master equation without RWA
#     comb = combinations(np.arange(len(J_op)),2)
#     comb = [c for c in comb]
    
#     for jump in comb:
#         i,j = jump
#         a,b= J_op[j], J_op[i]
#         # gamma matrix 
#         XI_ij= random_c[i]*random_c[j]
#       #  XI_ij= 1
#         Spos_i, Sneg_i= self.LambS(0.5*self.beta*epsilon[i],step)
#         Spos_i, Sneg_i= XI_ij*Spos_i, XI_ij*Sneg_i
            
# #        Spos_i, Sneg_i= 1000,1000
        
#         Spos_j, Sneg_j= self.LambS(0.5*self.beta*epsilon[j],step)
#         Spos_j, Sneg_j= XI_ij*Spos_j, XI_ij*Sneg_j
# #        Spos_j, Sneg_j= 1000,1000
#         gpos_i, gpos_j= random_cps[i], random_cps[j] 
#         gneg_i, gneg_j= random_cps_m[i], random_cps_m[j]
# #         gpos_i, gpos_j= 1000,1000
# #         gneg_i, gneg_j= 1000,1000
#         bd_a= b.dag()*a
#         a_bd= a*b.dag()
      
#   # generalized dissipator superoperator
#         D= (0.5*(gpos_j+gpos_i)+1j*(Spos_j-Spos_i))*(qt.spre(a)*qt.spost(b.dag()))\
#             - (0.5 * gpos_j + 1j * Spos_j) * qt.spre(bd_a)\
#             - (0.5 * gpos_i - 1j * Spos_i ) * qt.spost(bd_a)\
#            +(0.5*(gneg_j+gneg_i)+1j*(Sneg_j-Sneg_i))*(qt.spre(b.dag()) * qt.spost(a))\
#             - (0.5*gneg_i - 1j*Sneg_i)*qt.spre(a_bd)\
#             - (0.5*gneg_j + 1j*Sneg_j)*qt.spost(a_bd)                                                 
#         L_G += D
        
    # return L_RWA, L_RWAS, L_G, D_op, epsilon, random_c, H_d
    return L_RWA, D_op, epsilon, random_c, H_d
    
    

 # Generate an specific model

def create_Model(a_0,a_1,a_2,a_3,a_4):
    
  return OpenSYK(P["w"][a_0], P["gamma/w"][a_1], P["w/g"][a_2],\
                 P["gamma/beta"][a_3], P["N"][a_4],  1 )

#############################################################################

#############################################################################
'''
Master equation solver 
This function returns an object with the states of the evolved system given a time interval "times",
 an object with the expectation value of the central fermion occupation number, and the eigenvalues of
the Liouvillian operator.
This solver is only adapted for the master equation in Lindblad form (with RWA).
inputs:
times : time interval for the simulation
coupl_c: coupling constants for the quasifermionic operators Xi_i (\gamma_i(\epsilon_i))
coupl_cm: coupling constants for Xi_^\dagger
c_fermi: fermion annihilation operators in qubit form
N: Majorana fermion number
H: diagonalized Hamiltonian
norms: norms of the Xi_i operators
psi: initial state
options: integration step setting
'''

#Time evolution of the system, linear

def solve2(times, coupl_c, coupl_cm, c_fermi, N, H, norms, psi, options):

  #jump operators
  J_op = [np.sqrt(coupl_c[i])*c_fermi[i] for i in range(N//2+1)]
  J_opd=[np.sqrt(coupl_cm[i])*c_fermi[i].dag() for i in range(N//2+1)]
  J =[ J_op, J_opd]
  J= [ J[i][j] for i in range(len(J)) for j in range(2)]
  #central fermion annihilation operator
  d = [c_fermi[i]/norms[i] for i in range(N//2+1)]
  D = sum(d)
  D_op= D.dag()*D
  #collapse operators
#gain model
  C_op= [J_op[i].dag()*J_op[i] for i in  range(N//2+1)]
#loss-gain model
  C= [J[i].dag()*J[i] for i in range(len(J))] 

  #initial state
  #psi_0 =qt.tensor(*[qt.basis(2,1) for j in range(N//2+1)])

  #Lindbladian operator
  # gain
#   Lindbladian_g = qt.liouvillian(H, C_op)
#   eps_Lg =Lindbladian_g.eigenenergies()
  # loss-gain
  Lindbladian_lg = qt.liouvillian(H, J)
  eps_Llg = Lindbladian_lg.eigenenergies()
    
  #evolve and calculate expectation values for each quasi-fermion,d 
  
  out_d = qt.mesolve(H, psi, times, J, D_op, options= options)
  rho= qt.mesolve(H, psi, times, J, [])
 
        
  return out_d, rho, eps_Llg, Lindbladian_lg
  


#############################################################################

#############################################################################



# intrinsic time scale
'''
Returns the smallest and largest intrinsic time scale for a given random sampling
evolution.
'''
def Internal_TR(values):
    T_S= []
    T_M =[]
    for sample in range(samples):
        comb = combinations(values[sample], 2)
        rates= [abs(c[0]-c[1]) for c in comb]
        T_S.append([min(rates)])
        T_M.append([max(rates)])
    T_S = np.array(T_S)
    T_M = np.array(T_M)
    minimum= np.where(T_S == min(T_S))[0]
    maximum = np.where(T_M == max(T_M))[0]
    
    return minimum, maximum

#############################################################################

#############################################################################

## Use arbitrary float precision to avoid nan values ##
'''
Returns the steady state values for physical quantities when N=2.
'''
from mpmath import *

def S_infty(Beta, eps):
    e0, e1= eps
    #e0 = abs(e0)
    #e1 = abs(e1)
    e= e0+e1
    return  Beta*((e0*exp(-Beta*e0)+e1*exp(-Beta*e1)\
           +e*exp(-Beta*e) )/(1+exp(-Beta*e0)\
           +exp(-Beta*e1)+exp(-Beta*e)) )+log(1+exp(-Beta*e0)\
           +exp(-Beta*e1) +exp(-Beta*e))

def P_infty(Beta,eps): 
    e0, e1= eps
    e= e0+e1
    L= 1+exp(-Beta*e0)+exp(-Beta*e1)\
        +exp(-Beta*e)
    return (1+exp(-2*Beta*e0)+exp(-2*Beta*e1)\
               +exp(-2*Beta*e))/L**2
   
#############################################################################

#############################################################################

## Parallelization

def batch_data(seed, *args):

    model,step,op,time= args

    #generate the system Liouvilian and Gibbs state
    # L, LS, LG, D, epsilon, random_c, H_d = model.Liouvilian(step)
    L, D, epsilon, random_c, H_d = model.Liouvilian(step,seed)
    return [L, H_d, D, epsilon]
    
def eig_L(l): 
  return l.eigenenergies()    


#############################################################################

#############################################################################






# Density matrix histogram

def matrix_histogram_complex(M, xlabels, ylabels, title, size, limits=None, ax=None):
    """
    Draw a histogram for the amplitudes of matrix M, using the argument of each element
    for coloring the bars, with the given x and y labels and title.

    Parameters
    ----------
    M : Matrix of Qobj
        The matrix to visualize

    xlabels : list of strings
        list of x labels

    ylabels : list of strings
        list of y labels

    title : string
        title of the plot

    limits : list/array with two float numbers
        The z-axis limits [min, max] (optional)

    ax : a matplotlib axes instance
        The axes context in which the plot will be drawn.
    
    Returns
    -------

        An matplotlib axes instance for the plot.

    Raises
    ------
    ValueError
        Input argument is not valid.

    """

#    if isinstance(M, Qobj):
        # extract matrix data from Qobj
#    M = M.full()

    n= np.size(M) 
    xpos,ypos= np.meshgrid(range(M.shape[0]),range(M.shape[1]))
    xpos=xpos.T.flatten()-0.5 
    ypos=ypos.T.flatten()-0.5 
    zpos = np.zeros(n) 
    dx = dy = 0.8 * np.ones(n) 
    Mvec = M.flatten()
    dz = abs(Mvec) 
    
    # make small numbers real, to avoid random colors
    idx, = np.where(abs(Mvec) < 0.001)
    Mvec[idx] = abs(Mvec[idx])

    if limits: # check that limits is a list type
        z_min = limits[0]
        z_max = limits[1]
    else:
        phase_min = -np.pi
        phase_max = np.pi
        
    phase_min = -np.pi
    phase_max = np.pi    
    norm= mpl.colors.Normalize(phase_min, phase_max) 

    # create a cyclic colormap
    cdict = {'blue': ((0.00, 0.0, 0.0),
                      (0.25, 0.0, 0.0),
                      (0.50, 1.0, 1.0),
                      (0.75, 1.0, 1.0),
                      (1.00, 0.0, 0.0)),
            'green': ((0.00, 0.0, 0.0),
                      (0.25, 1.0, 1.0),
                      (0.50, 0.0, 0.0),
                      (0.75, 1.0, 1.0),
                      (1.00, 0.0, 0.0)),
            'red':   ((0.00, 1.0, 1.0),
                      (0.25, 0.5, 0.5),
                      (0.50, 0.0, 0.0),
                      (0.75, 0.0, 0.0),
                      (1.00, 1.0, 1.0))}
    cmap = mpl.colors.LinearSegmentedColormap('phase_colormap', cdict, 256)

    colors = cmap(norm(np.angle(Mvec)))  

    if ax == None:

        fig = plt.figure(figsize=size)
        ax = Axes3D(fig, azim=-35, elev=35)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)
    plt.title(title, fontsize= 24)

    # x axis
    ax.axes.w_xaxis.set_major_locator(ticker.IndexLocator(1,-0.5))
    ax.set_xticklabels(xlabels) 
    ax.tick_params(axis='x', labelsize=1)

    # y axis
    ax.axes.w_yaxis.set_major_locator(ticker.IndexLocator(1,-0.5)) 
    ax.set_yticklabels(ylabels) 
    ax.tick_params(axis='y', labelsize=1)

    # z axis
    #ax.axes.w_zaxis.set_major_locator(ticker.IndexLocator(-1,-0.5))
    ax.set_zlim3d([z_min, z_max])
    ax.set_zlabel('abs',size=22)
    
    ax.tick_params(labelsize=15)
    # color axis
    cax, kw = mpl.colorbar.make_axes(ax, shrink=.41, pad=.11)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    cb.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cb.set_ticklabels((r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'))
    cb.set_label('arg',size=22)
    cb.ax.tick_params(labelsize=22) 
    return ax
