# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:18:31 2020

@author: kevin
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

# %% variables
tauI = 2
tauDV = 2
tauo = 2
vv = 0.22
dtheta = 10
tau0 = 4.2
a,b,c = 1,0.2,0.01
sigma = np.pi/6
gamma0 = 0.5
taus = 2
taub = 4

alpha_s = -0.02
alpha_p = 0.03
alpha_g = 0.1


# %% time and initialization
T = 10000  #in ms
dt = 1  #ms
lt = int(T/dt)
time = np.arange(0,lt*dt,dt)
Vs = np.zeros((6,lt))
#simple neural network connection (should play with this assumption)
W = np.array([[0, 0, +1, 0, 0, 0],
              [0, 0, -1, 0, 0, 0],
              [0, 0, 0, +1, +1, +1],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              ])
XY = np.zeros((2,lt))

# %% functions
def neural_dynamics(Vs,S,theta):
    """
    Euler updates of the simplified linear netowrk
    (should modify as matrix form with fliexible nonlinearity)
    """
    dV = np.zeros(len(Vs))
    Vs[0:1] = S
    dV[2] = dt/tauI*(Vs[0]*W[0,2] + Vs[1]*W[1,2])
    dV[3:4] = dt/tauDV*(Vs[2]*W[2,3:4])
    dV[5] = dt/tauo*(Vs[2]*W[2,5])
    return dV

def steering(Vs,alpha_s):
    """
    slow continuous steering angle change
    """
    tau_vd = tau0/2 + alpha_s*Vs[3:4]
    return tau_vd

def Pirouette(Vs,alpha_p):
    """
    Frequency of the random turning process
    """
    lambda_p = alpha_p/(a+np.exp(b*V[5])) + c
    return lambda_p

def turn_angle(Vs,alpha_g):
    """
    Biased turning angle after Pirouette
    """
    gammaB = gamma0 + alpha_g*Vs[5]
    gammaA = 1 - gammaB
    f_theta = gammaA/(np.pi*2) + gammaB/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-(th-np.pi)/(2*sigma**2))
    #############################
    return f_theta

def environment(xy):
    """
    a simple Gaussian diffusion 2-D environement
    """
    M = 200   #max concentration
    sig2 = 100  #width of Gaussian
    target = np.array([30,30])   #target position
    NaCl = M*np.exp(-((xy[0]-target[0])**2+(xy[1]-target[1])**2)/sig2)
    return NaCl

def sensory(NaCl):
    """
    On-OFF response of ASE
    """
    baseline = 1
    S = 0
    if NaCl>=baseline:
        S = 60
    return S

def V_NaCl():
    """
    additional neurons response here~~
    """
    return

# %% iterations
Vs[:,0] = np.random.rand(Vs.shape[0])
XY[:,0] = np.random.randn(2)
for tt in range(lt-1):
    NaCl = environment(XY[:,0])
    S = sensory(NaCl)
    dV = neural_dynamics(Vs[tt],S[tt],theta)
    Vs[:,tt+1] = Vs[:,tt] + dV
    lambda_p = Pirouette(Vs[:,tt+1],alpha_p)
    if lambda_p>=np.random.rand():
        dth = turn_angle(Vs[:,tt+1],alpha_g)
    else:
        dth = steering(Vs[:,tt+1],alpha_s)
    XY[:,tt+1] = XY[:,tt] + vv*np.array([np.cos(dth), np.sin(dth)])
    
# %% steering with heading dynamics
def wrap2deg(dth):
    #sign = np.sign(theta)
    #rem = np.abs(np.mod(theta, 180))
    if dth > 180:
        dth = dth-360  #bounded by angle measurements
    if dth < -180:
        dth = dth+360
    return dth #sign*rem
deg2pi = np.pi/180

def sinusoid_path(vv,dtheta,dt,path_len,offset):
    xys = np.zeros((2,path_len))
    ths = np.zeros(path_len)
    phase = np.zeros(path_len)
    C = np.zeros(path_len)
    for tt in range(path_len-1):
        xys[:,tt+1] = xys[:,tt] + dt*(np.array([vv*np.cos(ths[tt]*deg2pi), vv*np.sin(ths[tt]*deg2pi)]))
        ths[tt+1] = wrap2deg( ths[tt] + dt*(dtheta*np.sign(np.sin(2*np.pi*(dtheta/360)*tt*dt)+offset)) )
        phase[tt+1] = np.sign(np.sin(2*np.pi*(dtheta/360)*tt*dt+offset))
        C[tt+1] = environment(xys[:,tt+1])  ##this can be feedback to the phase offset!
        # * np.sign(np.sin(ths[tt]))
    return xys, ths, phase, C

xys, ths, phase, C = sinusoid_path(vv,dtheta,0.1,3000,-0.0)
plt.plot(xys[0,:], xys[1,:])
plt.scatter(xys[0,:],xys[1,:],c=C)

# %% plotting tracks
xys, ths, phase, C = sinusoid_path(vv,dtheta,0.1,2000,0)
