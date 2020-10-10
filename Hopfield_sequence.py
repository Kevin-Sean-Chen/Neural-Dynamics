# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 04:47:40 2020

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import itertools
from scipy.stats import norm
from scipy import stats
from sklearn.linear_model import LinearRegression

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


# %% Boltzmann machine
N = 10  #network size
spins = np.array([-1,1])  #spin configuration
kbT = 1  #temperatue
beta = 1/kbT
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
#    if np.random.rand()<P:
#        r = 1
#    else:
#        r = -1
    return r

# %% temporal settings
T = 1000
us = np.ones((N,T))
Vs = np.ones((N,T))
eps1 = np.random.randint(0,2,N) #np.random.randn(N)
eps2 = np.random.randint(0,2,N)#np.random.randn(N)
Ws = np.random.randn(N,N,T)
Ws[:,:,0] = np.outer(eps1,eps1)*1 + np.outer(eps2,eps2)#
eps = 0.0  #learning rate
FF = np.eye(N)*5  #simple feedforward

# %% stimulation
### total random
stim = np.random.randint(0,2,(N,T))
## sequential
#sl = 3
#temp = np.random.randint(0,2,(N,sl))
#stim_ = np.tile(temp,int(np.floor(T/sl)))
#stim[:,:stim_.shape[1]] = stim_
#stim[stim==0] = -1

# %% neural dynamics
for tt in range(1,T):
    us[:,tt] = np.matmul(Ws[:,:,tt],Vs[:,tt-1]) + np.matmul(FF,stim[:,tt])*1
    Vs[:,tt] = sigma(us[:,tt],kbT)
    Ws[:,:,tt] = Ws[:,:,tt-1] + eps*(np.outer(Vs[:,tt],Vs[:,tt]))
#                Ws[:,:,tt] + eps*(np.matmul(us[:,tt],(us[:,tt-1]-Ws[:,:,tt-1])) \
#                  + np.matmul(us[:,tt-1],(us[:,tt]-Ws[:,:,tt-1])) \
#                  + np.dot(np.outer(us[:,tt],us[:,tt-1]), (1-Ws[:,:,tt-1])) )  #seuqential learning rule

# %%
plt.figure()
plt.imshow(Vs,aspect='auto')

# %%
###############################################################################
### Work distribution
###############################################################################
# %% settings
N = 15
#targets
Vi = np.random.randint(0,2,N)
Vi[Vi==0] = -1
Vf = np.random.randint(0,2,N)
Vf[Vf==0] = -1
#Vf = -Vi
#weights
weighted = 0.1
W = 1.*((1-weighted)*np.outer(Vi,Vi) + weighted*np.outer(Vf,Vf)) + np.random.randn(N,N)*.5
#W = W-np.diag(W)

# %% dynamics
#trials
rep = 10000
max_it = 1000
# physical
kbT = 15  #temperature
U = 0  #threshold
bb = .005  #input weights
#dynamics and protocol
wf = np.zeros(rep)
wr = np.zeros(rep)
#test recording time series
rec_I = np.zeros(rep)
rec_v = np.zeros(rep)

for rr in range(rep):
    ###forward (i->f)
    overlap_f = 0  #initial overlap
    u = np.matmul(W, Vi)  #inital current
    #u = Vi.copy()
    Et = -0.5*Vi @ W @ Vi #0 #intial energy
    ww = 0  #counting steps
    I = 0  #initial input
    while overlap_f != N:
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
        
        # ### IN PROGRESS ###
#        rec_I[rr] = Vc
#        rec_v[rr] = I
        
        if ww>max_it:
            break
    #print(ww)
    #wf[rr] = wf[rr]  #/ww  #expected energy
    
    ###reverse (f->i)
    overlap_r = 0
    u = np.matmul(W, Vf)
    #u = Vf.copy()
    Et = -0.5*Vf @ W @ Vf
    ww = 0
    I = 0
    while overlap_r != N:
        Vc = sigma(u-U, kbT)  #voltage
        Ii = bb*Vi #np.matmul(W,Vi)  #input current
        I = Ii + I
        u = np.matmul(W,Vc) + I  #current
        overlap_r = int(np.dot(Vi,Vc))
        dw = np.dot(Vc,I) - Et
        Et = np.dot(Vc,I)
        wr[rr] = wr[rr] + np.dot(Vc,Ii)
        ww = ww + 1
        if ww>max_it:
            break
    wr[rr] = -wr[rr]  #as P(-W)

# %% work distribution
plt.figure()
plt.hist(wf[0:]-0*wf[0],100)
plt.hist(wr[0:]-0*wr[0],100,alpha=0.5)

# %%
plt.figure()
plt.hist(wf,100)
plt.xlabel('$W$')
plt.ylabel('$P_f(W)$')

# %% Gaussian curve fitting
plt.figure()
#mean,std = norm.fit(wf)
ae, loce, scalee = stats.skewnorm.fit(wf)
plt.hist(wf, bins=100, normed=True, label='$P_f(W)$')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
#y = norm.pdf(x, mean, std)
y = stats.skewnorm.pdf(x , ae, loce, scalee)
plt.plot(x, y, 'b')
#mean,std = norm.fit(wr)
ae, loce, scalee = stats.skewnorm.fit(wr)
plt.hist(wr, bins=100, normed=True, alpha=0.5, label='$P_r(-W)$')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
#y = norm.pdf(x, mean, std)
y = stats.skewnorm.pdf(x , ae, loce, scalee)
plt.plot(x, y, 'r')
plt.xlabel('$W$')
plt.ylabel('$P_f(W), P_r(-W)$')
plt.legend()

# %% CFT plot 2
xm,xM = min(np.hstack([wf,wr])),max(np.hstack([wf,wr]))
com_bin = np.arange(xm,xM,0.05)
plt.figure()
nwf,_= np.histogram(wf,bins=com_bin,normed=True)
nwr,_= np.histogram(wr,bins=com_bin,normed=True)
y_x = np.log(nwf/nwr)
plt.plot(com_bin[:-1], y_x,'-')
plt.hlines(0,-.5,2.5,linestyle='dashed')
#plt.plot(np.cumsum(nwr))
plt.xlabel('$W$')
plt.ylabel('$log(P_f(W)/P_r(-W))$')

# %% regression
xs = com_bin[:-1]
pos_inf = np.where(np.isinf(y_x)==False)[0]
ys = y_x[pos_inf]
xs = xs[pos_inf]
pos_real = np.where(np.isnan(ys)==False)[0]
ys = ys[pos_real]
xs = xs[pos_real]
reg = LinearRegression().fit(xs[2:-2].reshape(-1, 1), ys[2:-2].reshape(-1, 1))
pred_dF = -reg.intercept_/reg.coef_ #reg.predict(np.array([0]).reshape(-1,1))
print(pred_dF)
y_ = reg.predict(xs.reshape(-1,1))
plt.figure()
plt.plot(xs, ys,'o',label='data')
plt.hlines(0,-.5,1.5,linestyle='dashed',label='$y=0$')
plt.plot(xs,y_,'-',label='fitting')
#plt.plot(np.cumsum(nwr))
plt.xlabel('$W$',fontsize=30)
plt.ylabel('$log(P_f(W)/P_r(-W))$',fontsize=30)
plt.legend(fontsize=25)

# %% Free energy calculation
def entropy(v):
    p1 = len(np.where(v==+1)[0])/len(v)
    return -(p1*np.log(p1)/np.log(2) + (1-p1)*np.log(1-p1)/np.log(2))

Fi = (-0.5*1*Vi @ W @ Vi - 1*Vf @ Vi + U*np.ones(N) @ Vi) - 1*kbT*entropy(Vi) #F=U-TS
Ff = (-0.5*1*Vf @ W @ Vf - 1*Vi @ Vf + U*np.ones(N) @ Vf) - 1*kbT*entropy(Vf)
dF = Ff - Fi
print(dF)   
print(np.mean(wf)-np.mean(wr))

# %% Sampling for free energy calculation
def sig_prob(h,T):
    """
    Boltzmann probability
    """
    P = 1/(1+np.exp(-2*h/T))
    return P

def Free_Energy(W,V,I,U,T):
    """
    An attempt for calculating free energy following RBM formula
    """
    xj = W @ V + 1*I - U
    pj = sig_prob(xj,T)
    F = np.dot(pj,xj) + T*(np.dot(pj,np.log(pj)) + np.dot((1-pj),np.log(pj)))
    return F

Fi = Free_Energy(W,Vi,Vf,U,kbT)
Ff = Free_Energy(W,Vf,Vi,U,kbT)
print(Ff-Fi)
print(np.abs(np.dot(Vi,Vf)/N))

# %% information estimation
spins = list(itertools.product([-1, 1], repeat=N))
Esf = np.zeros(len(spins))
Psf = np.zeros(len(Esf))
Esi = np.zeros(len(spins))
Psi = np.zeros(len(Esi))
for ii in range(len(spins)):
    vv = np.array(spins[ii])
    Esf[ii] = -0.5* vv @ W @ vv - bb*Vf @ vv + np.ones(N)*U @ vv
    Psf[ii] = np.exp(-1/kbT*Esf[ii])
    Esi[ii] = -0.5* vv @ W @ vv
    Psi[ii] = np.exp(-1/kbT*Esi[ii])
    
H_V = -np.dot(Psf,np.log(Psf))
H_V_I = -np.dot(Psi,np.log(Psi))

print((1/kbT)*(H_V-H_V_I))

# %% enumerating free energy
spins = list(itertools.product([-1, 1], repeat=N))
Esf = np.zeros(len(spins))
Psf = np.zeros(len(Esf))
Esi = np.zeros(len(spins))
Psi = np.zeros(len(Esi))
for ii in range(len(spins)):
    vv = np.array(spins[ii])
    Esf[ii] = -0.5* vv @ W @ vv - 1*Vf @ vv + np.ones(N)*U @ vv
    Psf[ii] = np.exp(-1/kbT*Esf[ii])
    Esi[ii] = -0.5* vv @ W @ vv - 1*Vi @ vv + np.ones(N)*U @ vv
    Psi[ii] = np.exp(-1/kbT*Esi[ii])

# computing free-energy
Zf = sum(Psf)
Psf = Psf/Zf
Zi = sum(Psi)
Psi = Psi/Zi
print((-kbT*np.log(Zf)+kbT*np.log(Zi)))
#Free = np.dot(Ps,Es) - kbT*(-np.dot(Ps, np.log(Ps)))
#print(Free)

Fj = (-kbT)*np.log(np.mean(np.exp((-1/kbT)*wf)))  # via Jarzynski's work relation
print(Fj)

# %% CFT plot 2
plt.figure()
plt.plot(F_tr,F_CFT,'o')
plt.xlabel('Numerical $\Delta F$')
plt.ylabel('$CFT \Delta F$')


# %%
###############################################################################
### XFT
###############################################################################
# %% settings
N = 15
#targets
Vi = np.random.randint(0,2,N)
Vi[Vi==0] = -1
Vf = np.random.randint(0,2,N)
Vf[Vf==0] = -1
#Vf = -Vi
#weights
weighted = 0.1
W = 1.*((1-weighted)*np.outer(Vi,Vi) + weighted*np.outer(Vf,Vf)) + np.random.randn(N,N)*2
#W = W-np.diag(W)

# %% dynamics
#trials
rep = 300000
# physical
T1 = 5  #temperature
T2 = 30
U = 0  #threshold
# heat exchange
Qs = np.zeros(rep)

for rr in range(rep):
    ###forward (i->f)
    u1 = np.matmul(W, Vf)  #inital current
    V1 = sigma(u1-U,T1)
    E1 = -0.5*V1 @ W @ V1 #0 #intial energy
    u2 = np.matmul(W, V1) 
    V2 = sigma(u2-U, T2)  #voltage
    E2 = -0.5*V2 @ W @ V2 #after heat exchange
    Qs[rr] = E2-E1

# %% dQ distribution
plt.figure()
plt.hist(Qs,50)

# %% XFT plot
bins = np.arange(-50,50,1)
aa,bb = np.histogram(Qs,bins,density=True)
#plt.figure()
plt.plot(bb[-50:-1],(np.log(aa[-49:]/np.flip(aa[2:51]))))
plt.xlabel('$\Delta Q$')
plt.ylabel('$log(P(\Delta Q)/P(-\Delta Q))$')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% Bennett method
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% BAR functions
def fx(W, x, kbT):
    """
    Bennett's optimal function
    """
    #func = np.exp(x/(2*kbT)) / (1+np.exp((x-W)/kbT))  #(Science...2008)
    func = 1/(1+1*np.exp(1/kbT*(W-x)))  #minimizes variance (PRE, 2020)
    return func

def zs(x, wbin, wf, wr, kbT):
    """
    difference of z function, where z functions are the function of x in BAR method
    """
    zfs = np.array([fx(ww,x,kbT)*np.exp(-1/kbT*ww) for ww in wbin])
    zf = np.log(np.dot(zfs,wf))
    zrs = np.array([fx(ww,x,kbT) for ww in wbin])
    zr = np.log(np.dot(zrs,wr))
    dF = (zr-zf)/np.sum(wr)  #in units of kbT
    return dF

# %%
wr_ = wr.copy()  #not taking negative value
xx = np.linspace(-.1,2,50)
xm,xM = min(np.hstack([wf,wr_])),max(np.hstack([wf,wr_]))
com_bin = np.arange(xm,xM,0.1)
dFs_BAR = np.zeros(len(xx))
pwf,_= np.histogram(wf,bins=com_bin,density=True)
pwr,_= np.histogram(wr_,bins=com_bin,density=True)
#pwf, pwr = pwf/np.sum(pwf), pwr/np.sum(pwr)
for ii,xi in enumerate(xx):
    dFs_BAR[ii] = zs(xi, com_bin[:-1], pwf, pwr, kbT)

# %%
plt.figure()
plt.plot(xx,dFs_BAR,'-o',label='$y=z_R(x)-z_F(x)$')
plt.plot(xx,xx,'--',label='$y=x$')
plt.legend(fontsize=25)
plt.xlabel('$x$',fontsize=30)
plt.ylabel('$y(x)$',fontsize=30)
