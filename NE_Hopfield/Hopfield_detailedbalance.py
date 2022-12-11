# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 05:26:54 2020

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import itertools
from scipy.stats import norm
from scipy import stats
from sklearn.linear_model import LinearRegression
import sklearn

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30) 

import matplotlib as mpl
mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True

# %% functions
def sigma(h,T):
    """
    Stochastic nonlinearity following Boltzmann machine
    """
    P = 1/(1+np.exp(-2*h/T))
    ll = len(h)
    rand = np.random.rand(ll)
    pos = np.where(P-rand>0)[0]
    r = -np.ones(ll)
    r[pos] = +1
    return r

def state_index(vt,N):
    """
    Return index of a neural state in iterated all possible state configuration
    """
    spins = list(itertools.product([-1, 1], repeat=N))
    ID = spins.index(tuple(vt))
    return ID

def KL(p, q):
	return sum(p[i] * np.log2(p[i]/q[i]) for i in range(len(p)) if np.isnan(p[i])==False and  np.isnan(q[i])==False)

# %%
###############################################################################
### Boltzmann machine
###############################################################################
# %% settings
N = 10
#targets
Vi = np.random.randint(0,2,N)
Vi[Vi==0] = -1
Vf = np.random.randint(0,2,N)
Vf[Vf==0] = -1
#Vf = -Vi
#weights
weighted = 0.2
#W = 1.*((1-weighted)*np.outer(Vi,Vi) + weighted*np.outer(Vf,Vf)) + np.random.randn(N,N)*.5  #two-state setup
W = 1.*((1-weighted)*np.outer(Vi,Vi) + weighted*np.outer(Vf,Vf)) + np.outer(Vi,Vf)*.0  #with weighted sequential step
W = W-np.diag(W)

# %% dynamics
#trials
rep = 10
max_it = 3000
# physical
kbT = 15  #temperature
U = 0  #threshold
bb = .01  #input weights
#dynamics and protocol
wf = np.zeros(rep)
wr = np.zeros(rep)
#test recording time series
rec_v = np.zeros((rep,max_it))

for rr in range(rep):
    ###forward (i->f)
    overlap_f = 0  #initial overlap
    u = np.matmul(W, Vi)  #inital current
    Et = -0.5*Vi @ W @ Vi #0 #intial energy
    ww = 0  #counting steps
    I = 0  #initial input
    while ww < max_it-1: #overlap_f != N:
        Vc = sigma(u-U, kbT)  #voltage
        Ii = bb*Vf  #np.matmul(W,Vf)  #input current
        I = Ii + I
        u = np.matmul(W,Vc) + I  #Vf  #current
        overlap_f = int(np.dot(Vf,Vc))  #overlapping
        #u = sigma(np.matmul(W,u) + Vf - U,kbT)
        dw = np.dot(Vc,I) - Et  #work is difference of energy as a function of input change
        Et = np.dot(Vc,I)  #update E
        wf[rr] = wf[rr] + np.dot(Vc,Ii) #+dw #work trajectory
        ww = ww + 1
        
        # ### Recording ###
        rec_v[rr, ww] = state_index(Vc.astype(int).tolist(), N)
        
        #if ww>max_it:
        #    break
    
#    ###reverse (f->i)
#    overlap_r = 0
#    u = np.matmul(W, Vf)
#    #u = Vf.copy()
#    Et = -0.5*Vf @ W @ Vf
#    ww = 0
#    I = 0
#    while overlap_r != N:
#        Vc = sigma(u-U, kbT)  #voltage
#        Ii = bb*Vi #np.matmul(W,Vi)  #input current
#        I = Ii + I
#        u = np.matmul(W,Vc) + I  #current
#        overlap_r = int(np.dot(Vi,Vc))
#        dw = np.dot(Vc,I) - Et
#        Et = np.dot(Vc,I)
#        wr[rr] = wr[rr] + np.dot(Vc,Ii)
#        ww = ww + 1
#        if ww>max_it:
#            break
#    wr[rr] = -wr[rr]  #as P(-W)
    
# %% Markov setup
H,xe,ye = np.histogram2d(rec_v[2,1:],rec_v[2,:-1],(2**N,2**N))
#Pxy = H/H.sum()   #joint prob
Ptr12 = H / H.sum(axis=1)[:, np.newaxis]  #transition prob
H,xe,ye = np.histogram2d(rec_v[2,:-1],rec_v[2,1:],(2**N,2**N))
#Pxy = H/H.sum()   #joint prob
Ptr21 = H / H.sum(axis=1)[:, np.newaxis]  #transition prob
ct,bs = np.histogram(rec_v, 2**N)
Px = ct/ct.sum()

# %% detailed balance
p1 = Ptr12 @ Px
p2 = Ptr21 @ Px
p1 = p1/np.nansum(p1)
p2 = p2/np.nansum(p2)
plt.figure()
plt.plot(p1)
plt.plot(p2,'r')
plt.xlabel('states', fontsize=30)
plt.ylabel('$W(\sigma\',\sigma) P_{eq}(\sigma)$, $W(\sigma,\sigma\') P_{eq}(\sigma\')$', fontsize=30)

sklearn.metrics.mutual_info_score(p1,p2)  # KL divergrence (?)
KL(p1,p2)

# %%
###############################################################################
# Computing housekeeping heat as a function of asymmetry connections (how much it is out of equilibrium)
###############################################################################
# %%
def network_2state(N,alpha,gamma,eta):
    """
    Returns network weight matrix W that is NxN, constructed with two patterns and a bias term gammma
    Another term alpha governs the strength of overlapping between two patterns
    The last term eta controls the strength of a unrelated Gaussian noise
    Return the matrix and two patterns
    """
    #targets
    Vi = np.random.randint(0,2,N)
    Vi[Vi==0] = -1
    Vf = np.random.randint(0,2,N)
    Vf[Vf==0] = -1
    #weights
    weighted = gamma
    W = (1-weighted)*np.outer(Vi,Vi) + weighted*np.outer(Vf,Vf) + np.outer(Vi,Vf)*alpha + np.random.randn(N,N)*eta
    W = W-np.diag(W)
    return W, Vi, Vf

def equilibrium_states(W,U,kbT,bb, Vi, Vf, max_it):
    """
    Function that recieves the connectivity, threshold, temperature, input, and two neural states.
    Outputs the "equilibrium state" computed numerically through time
    """
    #dynamics and protocol
    wf = np.zeros(rep)
    #test recording time series
    rec_v = np.zeros(max_it)
    ###forward (i->f)
    u = np.matmul(W, Vi)  #inital current
    ww = 0  #counting steps
    I = 0  #initial input
    while ww < max_it-1: #overlap_f != N:
        Vc = sigma(u-U, kbT)  #voltage
        Ii = bb*Vf  #np.matmul(W,Vf)  #input current
        I = Ii + I
        u = np.matmul(W,Vc) + I  #Vf  #current
        wf[rr] = wf[rr] + np.dot(Vc,Ii) #+dw #work trajectory
        ww = ww + 1
        # ### Recording ###
        rec_v[ww] = state_index(Vc.astype(int).tolist(), N)

    return rec_v


def detailed(rec_v):
    """
    Given the time serise of neural states near equilibrium, compute the transition matrix for forward and backward processes
    """
    H,xe,ye = np.histogram2d(rec_v[1:],rec_v[:-1],(2**N,2**N))
    #Pxy = H/H.sum()   #joint prob
    Ptr12 = H / H.sum(axis=1)[:, np.newaxis]  #transition prob
    H,xe,ye = np.histogram2d(rec_v[:-1],rec_v[1:],(2**N,2**N))
    #Pxy = H/H.sum()   #joint prob
    Ptr21 = H / H.sum(axis=1)[:, np.newaxis]  #transition prob
    ct,bs = np.histogram(rec_v, 2**N)
    Px = ct/ct.sum()
    p1 = Ptr12 @ Px
    p2 = Ptr21 @ Px
    p1 = p1/np.nansum(p1)
    p2 = p2/np.nansum(p2)
    return p1, p2

def QHK(p, q):
    """
    Computing housekeeping heat from forward and backward probabilities
    """
    return np.sum(np.where( np.logical_and(np.isnan(p) != 1, np.isnan(q) != 1), np.log(p / q), 0))


# %% Experiments
###recordings
scans = np.arange(0.1,1,0.1)  #scanning parameter  #np.array([0.1,0.9])#
Qs = np.zeros(len(scans))
max_it = 3000
###network
N = 10
U = 0
kbT = 5
bb = 0.001
#alpha = 0.1
gamma = 0.2
eta = 0.

W_,Vi,Vf = network_2state(N,0,gamma,eta)
for ii in range(len(scans)):
    ###exp
    alpha = scans[ii]
    W = W_ + np.outer(Vi,Vf)*alpha  #asymmetric networks
#    W,Vi,Vf = network_2state(N,alpha,gamma,eta)
    allvs = equilibrium_states(W,U,kbT,bb, Vi, Vf, max_it)
    ###analysis
    p1, p2 = detailed(allvs)
    Qs[ii] = QHK(p2,p1)

# %%
plt.figure()
plt.plot(scans, Qs, 'o', markersize=15)
plt.xlabel(r'$\alpha$',fontsize=40)
plt.ylabel('$EP$',fontsize=40)

# %%
W = W_ + np.outer(Vi,Vf)*0.5
vs_S = equilibrium_states(W,U,5,0, Vi, Vf, max_it)
W = W_ + np.outer(Vi,Vf)*0
vs_A = equilibrium_states(W,U,5,0, Vi, Vf, max_it)
plt.figure()
plt.plot(vs_A[:500],label=r'$\alpha$=0.5')
plt.plot(vs_S[:500],label=r'$\alpha$=0')
plt.xlabel(r'time steps',fontsize=40)
plt.ylabel('$m^{\mu}$',fontsize=40) 
plt.legend(fontsize=30, loc='upper center', bbox_to_anchor=(0.15, 1.01))
