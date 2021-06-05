# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 10:54:28 2021

@author: kevin
"""



import numpy as np
import scipy as sp
from scipy.optimize import minimize
import itertools
import copy

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=35) 
matplotlib.rc('ytick', labelsize=35) 

import matplotlib as mpl
mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True

# %% Equilibium Ising model
###############################################################################
# %% Stimuli
T = 10000
D = 3
smooth = 100
noise = np.random.randn(T,D)
X = noise.copy()
for ii in range(D):
    X[:,ii] = np.convolve(noise[:,ii],np.ones(smooth),'same')/smooth *0
    #X[:,ii] = np.convolve(noise[:,1],np.ones(smooth),'same') + np.random.randn(T)*3  #correlated

# %% Network settings
N = 20
Phi = np.random.randn(D,N)
J = np.random.randn(N,N)/N**0.5
J = 0.5*(J + J.T)  #make symmetric
J = J-np.diag(J)
#vv = np.random.randn(N)
#J = np.outer(vv,vv)*0.1 + np.random.randn(N,N)*0.05
#J = J_nMF.copy()

def Current(h,J,s):
    theta = h + J @ s
    return theta

def Transition(si,thetai,beta):
    P = np.exp(-si*thetai*beta)/(2*np.cosh(thetai*beta))
    rand = np.random.rand()
    if P>rand:
        s_ = -si.copy()
    elif P<=rand:
        s_ = si.copy()
    return s_

# %% Dynamics
def Kinetic_Ising(X,Phi,J,kbT):
    beta = 1/kbT
    N, T = Phi.shape[1], X.shape[0]
    S = np.ones((N,T))
    for tt in range(0,T-1):
        Ht = X[tt,:] @ Phi
        Theta = Current(Ht, J, S[:,tt])
        for ss in range(N):
            S[ss,tt+1] = Transition(S[ss,tt],Theta[ss],beta)
    return S

kbT = 0.5
S = Kinetic_Ising(X,Phi,J,kbT)

plt.figure()
plt.imshow(S, aspect='auto')

# %% Iterations
###############################################################################
# %% initialization
si = copy.deepcopy(S)
m = np.mean(S,1)
Cij = si @ si.T - np.outer(m,m)#np.cov(si)
#np.fill_diagonal(Cij,np.zeros(N))
J0 = -np.linalg.pinv(Cij)

# %% IP method
J_ip = np.zeros((N,N))
for ii in range(N):
    for jj in range(N):
        temp = ((1+m[ii])*(1+m[jj])+Cij[ii,jj])*((1-m[ii])*(1-m[jj])+Cij[ii,jj])/ \
        ((1-m[ii])*(1+m[jj])+Cij[ii,jj])/((1+m[ii])*(1-m[jj])+Cij[ii,jj])
        J_ip[ii,jj] = 0.25*np.log(temp)

# %% TAP method


# %% looping
its = 500
gamma = 0.01
Jij = copy.deepcopy(J0)  #Jij connectivity
ht = (X @ Phi).T #np.zeros((N,T))  #local field through time
for ii in range(its):
    dLt = 0
    for tt in range(T-1):
        ht[:,tt] = X[tt,:] @ Phi  #cheating here for now
        #ht[:,tt] = ht[:,tt] + (1)*gamma*(si[:,tt+1] - np.tanh(Current(ht[:,tt],Jij,si[:,tt])))
        dLt = dLt + (1)*(si[:,tt+1][:,None] - np.tanh(Current(ht[:,tt],Jij,si[:,tt]))[:,None]) @ si[:,tt][:,None].T
    dL = dLt/T
    Jij = Jij + gamma*dL

# %%
J_mf = J0.copy()
J_it = Jij.copy()
J_tr = J.copy()
np.fill_diagonal(J_mf,np.ones(N)*np.nan)
np.fill_diagonal(J_tr,np.ones(N)*np.nan)
np.fill_diagonal(J_it,np.ones(N)*np.nan)
plt.figure()
plt.plot(J_tr[:],J_mf[:],'o')
plt.figure()
plt.plot(J_tr[:],J_it[:],'o')



# %% Equilibrium numerics
#def numeircal_dF(kbT):
#    N = W.shape[0]
#    spins = list(itertools.product([-1, 1], repeat=N))
#    Esf = np.zeros(len(spins))
#    Psf = np.zeros(len(Esf))
#    Esi = np.zeros(len(spins))
#    Psi = np.zeros(len(Esi))
#    for ii in range(len(spins)):
#        vv = np.array(spins[ii])
#        Esf[ii] = -0.5* vv @ W @ vv - 1*Vf @ vv + np.ones(N)*U @ vv
#        Psf[ii] = np.exp(-1/kbT*Esf[ii])
#        Esi[ii] = -0.5* vv @ W @ vv - 1*Vi @ vv + np.ones(N)*U @ vv
#        Psi[ii] = np.exp(-1/kbT*Esi[ii])
#    
#    # computing free-energy
#    Zf = sum(Psf)
#    Psf = Psf/Zf
#    Zi = sum(Psi)
#    Psi = Psi/Zi
#    dF = (-kbT*np.log(Zf)+kbT*np.log(Zi))
#    
#    return dF
