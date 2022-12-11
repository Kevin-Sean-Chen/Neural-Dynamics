# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:38:16 2020

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from dotmap import DotMap

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

# %% functions
def Activation(w,x,nu):
    """
    Response nonlinearity of the single neuron
    """
    S = WeightedSum(w,x)
    if S+nu>0:
        y = +1
    else:
        y = -1
    return y

def WeightedSum(w,x):
    """
    Linear sum of the weighted patterns
    """
    return np.dot(w,x)

def Est_y_s(w,x,T):
    """
    Estimated expected response given weight, pattern, and effective temperature
    """
    S = WeightedSum(w,x)
    Est = (np.exp(S/T)-1)/(np.exp(S/T)+1)
    return Est

def WeightChange(r,y,x,w,pars):
    """
    Associative reward-penalty learning algorithm
    Given learning parameters, pattern x, response y, and reward r, return weight change with RL rule
    """
    lam, rho, T = pars
    Est = Est_y_s(w,x,T)
    if r==+1:
        dw = rho*(r*y - Est)*x
    elif r==-1:
        dw = lam*rho*(r*y - Est)*x
#    else:
#        dw = 0
    return dw

def Environment(x,y,dxy):
    """
    Given pattern x and response y, feedback reward according to environment dxy (with lookup table of P(reward))
    """
    patterns = dxy.x   #list of N patterns
    probs = dxy.p  #probabilities Nx2 (for +1 and -1)
    pos = FindPattern(x, patterns)  #find the matching pattern
    if y==+1:
        prob = probs[pos,1]
        if prob>np.random.rand():
            rt = +1
        else:
            rt = -1
    elif y==-1:
        prob = probs[pos,0]
        if prob>np.random.rand():
            rt = +1
        else:
            rt = -1
    return rt

def FindPattern(x,patterns):
    """
    Given a list of array patterns, return the index in the list that has the matching pattern to x
    """
    for pi,pp in enumerate(patterns):
        comparison = x == pp
        equal_arrays = comparison.all()
        if equal_arrays == True:
            pos = pi
    return pos

# %% parameters
#learning parameters
lam = 0.01
rho = 0.5
T = 0.15
pars = lam, rho, T
nu = 0.001
#trials structures
trials = 3000
seqs = 200
seq = np.random.choice(2,seqs)
Rs = np.zeros(trials)
#environment setting
P = 2  #two patterns for now
patterns = np.array([1,0]), np.array([1,1])
Pr = np.array([[0.6, 0.9],\
               [0.4, 0.2]])
dxy = DotMap()
dxy.x = patterns
dxy.p = Pr

#learning
w = np.zeros(P)  #np.random.randn(2)
for tt in range(trials):
    #initialize for trial
    reward = 0  #used for reward counting
    w_ = np.zeros(P)  #template used for batch update
    for ss in range(seqs):
        x = patterns[np.random.choice(P)]  #[seq[ss]]  #  #randomly pick pattern
        y = Activation(w, x, np.random.randn()*nu)  #measure activity
        rt = Environment(x, y, dxy)  #compute reward
        dw = WeightChange(rt, y, x, w, pars)  #update weights
        w_ = w_ + dw
        if rt>0:
            reward = reward + rt
    w = w + w_
    Rs[tt] = reward/seqs  #recording the probability of getting reward

plt.figure()
plt.plot(Rs)
print(w)

# %% record stochastic dynamics
def optimal_policy(x,dxy):
    patterns = dxy.x   #list of N patterns
    probs = dxy.p  #probabilities Nx2 (for +1 and -1)
    pos = FindPattern(x, patterns)  #find the matching pattern
    opt_y = np.argmax(probs[pos,:])
    if opt_y==0:
        y = -1
    elif opt_y==1:
        y = +1
    return y

ws = np.zeros((P,trials))  #record weights
dws = np.zeros((P,trials))  #record for forcing
match = np.zeros(trials)  #a target and a result
#learning
w = np.zeros(P)  #np.random.randn(2)
for tt in range(trials):
    #initialize for trial
    reward = 0  #used for reward counting
    w_ = np.zeros(P)  #template used for batch update
    ys = np.zeros(seqs)
    opt_y = np.zeros(seqs)
    for ss in range(seqs):
        x = patterns[np.random.choice(P)]  #[seq[ss]]  #  #randomly pick pattern
        y = Activation(w, x, np.random.randn()*nu)  #measure activity
        rt = Environment(x, y, dxy)  #compute reward
        dw = WeightChange(rt, y, x, w, pars)  #update weights
        w_ = w_ + dw
        if rt>0:
            reward = reward + rt
        ys[ss] = y
        opt_y[ss] = optimal_policy(x,dxy)
    w = w + w_
    ws[:,tt] = w
    dws[:,tt] = w_
    match[tt] = np.dot(ys,opt_y)/seqs
    Rs[tt] = reward/seqs  #recording the probability of getting reward

