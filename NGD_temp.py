# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 00:27:57 2020

@author: kevin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:32:35 2019
@author: kschen
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

# %% Noisy damped oscillator _test
G = 0.01  #gamma for correlation time
ww = G/2.12  #omega (fixed over-damped property)
D = 2  #noise strength (fixed thermal noise)

dt = 1/60  #time steps for Euler iteration
T = 1000 #time window
time = np.arange(0,T,dt)
lt = len(time)  #length of time window

S = np.zeros(lt)
V = np.zeros(lt)
S[0] = np.random.randn()
V[0] = np.random.randn()
for tt in range(0,lt-1):
    S[tt+1] = S[tt] + dt*V[tt]
    V[tt+1] = (1-G*dt)*V[tt] - ww**2*S[tt]*dt + np.random.randn()*np.sqrt(D*dt)

plt.plot(S,V)
plt.xlabel('S_t')
plt.ylabel('V_t')
# %% functions
def HMM(G):
    """
    Input damp coefficient and output time series of S_t for stimuli
    """
    ww = G/2.12  #omega (fixed over-damped property)
    D = 2  #noise strength (fixed thermal noise)
    
    dt = 1/60  #time steps for Euler iteration
    T = 1000 #time window
    time = np.arange(0,T,dt)
    lt = len(time)  #length of time window
    
    S = np.zeros(lt)
    V = np.zeros(lt)
    S[0] = np.random.randn()
    V[0] = np.random.randn()
    for tt in range(0,lt-1):
        S[tt+1] = S[tt] + dt*V[tt]
        V[tt+1] = (1-G*dt)*V[tt] - ww**2*S[tt]*dt + np.random.randn()*np.sqrt(D*dt)
    return S

def autocorr(x):
    """
    autocorrelation of a time series
    """
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]

def exp_decay(x, a, tau, b):
    """
    exponential decay form for autocorrelation fitting
    """
    return a*np.exp(-x*tau)+b  #y=Aexp(-t/tau) + B fitting

def exp_fit(xx):
    """
    exponenital fit for correlation time scale
    """
    x = np.arange(1,len(xx)+1)*dt  #time shift axis
    xx = xx/max(xx)
    popt, pcov = sp.optimize.curve_fit(exp_decay, x, xx)#, p0=(1, 1e6, 1))  
    return popt[1]  #return tau estimate

def DNF(m,x):
    """
    delayed negative feedback with m step backwards, acting on x, and returning y predictions
    """
    b = (3+m)/2
    y = np.zeros_like(x)
    for tt in range(0,len(y)-m):
        feedback = 0
        for kk in range(0,m-1):
            ck = (kk+1)/m
            feedback += ck*y[tt-(m-kk)]
        y[tt] = b*x[tt] - feedback
    return y

def TLMI():
    """
    Time-shifted MI caculated with correlation for now
    """
    return

# %% short test
Gs = np.array([0.5, 0.03, 0.01])
ms = 5000 #np.array([50,100,200,400,800])
tau_corr = []  #correlation time of S
Corr = []  #correlation between input output
peak_pos = []  #peak position of cross-correlation
for gg in Gs:
    ###generate correlated stimuli
    S = HMM(gg)  #stimuli
    S = S - np.mean(S)
    S = S/np.std(S)
    ###generate prediction
    y = DNF(ms,S)  #make predictions
    y = y - np.mean(y)
    y = y/np.std(y)
    ###peak measurements
    xcorr = np.correlate(S,y,mode='same')  #cross-correlation
    lags = np.arange(-int(len(xcorr)/2),int(len(xcorr)/2))*dt  #time lags
    tau_corr.append(xcorr)
#    peak_pos.append(lags[np.argmax(xcorr)])
#    lag_i = np.argmax(xcorr) - int(len(xcorr)/2)  #relative response time
#    temp = np.corrcoef(S[lag_i:],y[:-lag_i])  #compute encoding
#    Corr.append(temp[1][0])
    ##plt.plot(lags,xcorr)

# %%
plt.figure()
plt.plot(lags,np.array(tau_corr).T)

# %%
plt.figure()
for ii in range(len(tau_corr)):
    plt.plot(lags,tau_corr[ii],label='correlation='+str(1/Gs[ii]))
plt.legend(fontsize=30)
plt.title('m=83',fontsize=30)
plt.xlabel('time lag',fontsize=30)
plt.ylabel('cross-correlation',fontsize=30)

# %% scanning parameters
Gs = np.array([0.2,0.1,0.05,0.025,0.02,0.01,0.005])
ms = np.array([50,100,200,400,800])
tau_corr = []  #correlation time of S
Corr = []  #correlation between input output
peak_pos = []  #peak position of cross-correlation
for mm in ms:
    for gg in Gs:
        ###generate correlated stimuli
        S = HMM(gg)  #stimuli
        ###generate prediction
        y = DNF(mm,S)  #make predictions
        ###correlation time
#        acorr = autocorr(S)
#        tau = exp_fit(acorr[:])
#        tau_corr.append(1/tau)  #measure correlation time
        ###peak measurements
        xcorr = np.correlate(S,y,mode='same')  #cross-correlation
        lags = np.arange(-int(len(xcorr)/2),int(len(xcorr)/2))*dt  #time lags
        peak_pos.append(lags[np.argmax(xcorr)])
        lag_i = np.argmax(xcorr) - int(len(xcorr)/2)  #relative response time
        temp = np.corrcoef(S[lag_i:],y[:-lag_i])  #compute encoding
        Corr.append(temp[1][0])

# %%plotting
Corr_ = np.array(Corr).reshape(len(Gs),len(ms))
peak_ = np.array(peak_pos).reshape(len(Gs),len(ms))
plt.plot(1/Gs,Corr_,'-o')
plt.xlabel('tau_corr')
plt.ylabel('correlation coefficient')
plt.figure()
plt.plot(1./Gs,peak_,'-o')
plt.xlabel('tau_corr')
plt.ylabel('delta_p')

# %%plot with label