# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:18:31 2020

@author: kevin
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

# %% Appleby model
###############################################################################
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
    lambda_p = alpha_p/(a+np.exp(b*Vs[5])) + c
    return lambda_p

def turn_angle(Vs,alpha_g):
    """
    Biased turning angle after Pirouette
    """
    gammaB = gamma0 + alpha_g*Vs[5]
    gammaA = 1 - gammaB
    th = 0
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
    dV = neural_dynamics(Vs[:,tt],S,0)
    Vs[:,tt+1] = Vs[:,tt] + dV
    lambda_p = Pirouette(Vs[:,tt+1],alpha_p)
    if lambda_p>=np.random.rand():
        dth = turn_angle(Vs[:,tt+1],alpha_g)
    else:
        dth = steering(Vs[:,tt+1],alpha_s)
    XY[:,tt+1] = XY[:,tt] + vv*np.squeeze(np.array([np.cos(dth), np.sin(dth)]))

plt.figure()
plt.plot(XY[0,:],XY[1,:]) 
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

# %%
###############################################################################
# %% Effective model
###############################################################################
# %%
#simple switch
def sigmoid(x,w,t):
    ss = 1/(1+np.exp(-w*x-t))
    return ss
J = np.array([[0.1,-.5],[-.5,0.1]])*10
dt = 0.1
T = 1000
lt = int(T/dt)
Vs = np.zeros((2,lt))
tau = 1
Is = np.random.randn(lt)*1
for tt in range(lt-1):
    Vs[:,tt+1] = Vs[:,tt] + dt/tau*( -Vs[:,tt] + Is[tt] + J @ sigmoid(Vs[:,tt],1,0) ) \
    + np.random.randn(2)*np.sqrt(dt)*1.

#local stability
tt = np.matrix.trace(J)
aa = np.linalg.det(J)
lambs = 0.5*(tt+np.sqrt(tt**2-4*aa))

plt.figure()
plt.plot(Vs.T)

# %%
#functions
def environment(xx,yy):
    """
    a simple Gaussian diffusion 2-D environement
    """
    M = 50   #max concentration
    sig2 = 20  #width of Gaussian
    target = np.array([30,30])   #target position
    NaCl = M*np.exp(-((xx-target[0])**2+(yy-target[1])**2)/2/sig2**2)
    return NaCl+np.random.randn()*0.

def steering(vv,alpha_s,dcp,K):
    """
    slow continuous steering angle change
    """
    dth_s = alpha_s*dcp*np.abs(vv) + np.random.randn()*K
    return dth_s

def Pirouette(vv,alpha_p,lamb0):
    """
    Frequency of the random turning process
    """
    lambda_p = lamb0 + alpha_p*vv
    lambda_p = min(1,max(lambda_p,0))
    th = -0
    lambda_p = 0.023/(1+np.exp(alpha_p*(vv-th))) + lamb0
    return lambda_p

def turn_angle(vv,alpha_g,gamma0):
    """
    Biased turning angle after Pirouette
    """
    gammaB = gamma0 + alpha_g*vv
    gammaA = np.max(1 - gammaB,0)
    dth_b = gammaA*(np.random.rand()*2*np.pi-np.pi) + gammaB*(sigma*np.random.randn()-np.pi) #opposite direction
    #f_theta = gammaA/(np.pi*2) + gammaB/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-(th-np.pi)/(2*sigma**2))
    return dth_b

def dc_measure(dxy,xx,yy):
    """
    perpendicular concentration measure
    """
    perp_dir = np.array([-dxy[1], dxy[0]])  #perpdendicular direction
    perp_dir = perp_dir/np.linalg.norm(perp_dir)*1 #unit norm vector
    perp_dC = environment(xx+perp_dir[0], yy+perp_dir[1]) - environment(xx-perp_dir[0], yy-perp_dir[1])
    return perp_dC

def ang2dis(x,y,th):
    e1 = np.array([1,0])
    vec = np.array([x,y])
    theta = math.acos(np.clip(np.dot(vec,e1)/np.linalg.norm(vec)/np.linalg.norm(e1), -1, 1)) #current orienation relative to (1,0)
    v = vv + vs*np.random.randn()
    dd = np.array([v*np.sin(th), v*np.cos(th)])  #displacement
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,s), (-s, c)))  #rotation matrix, changing coordinates
    dxy = np.dot(R,dd)
    return dxy

# %% 
#with behavior
Vs = np.zeros((2,lt))
Cs = np.zeros(lt)
ths = np.zeros(lt)
XY = np.random.randn(2,lt)
proj = np.array([.1,.1])*1
lamb0 = 0.005
gamma0 = 0.5
alpha_p, alpha_s, alpha_g = -.00005, -.01, 0.0001
dxy = np.random.randn(2)
vv,vs = 0.55,0.05
K = np.pi/6
J = np.array([[0.1,-2.],[-2.,0.1]])*10
for tt in range(lt-1):
    ###neural dynamics
    Vs[:,tt+1] = Vs[:,tt] + dt/tau*( -Vs[:,tt] + proj*Cs[tt] + J @ sigmoid(Vs[:,tt],1,0) ) \
    + np.random.randn(2)*np.sqrt(dt)*.1
    ###behavior
    lambda_p = Pirouette(Vs[0,tt+1],alpha_p,lamb0)  #Pirouette #Cs[tt]
    if lambda_p>=np.random.rand():
        dth = turn_angle(Vs[0,tt+1],alpha_g,gamma0)  #bias
    else:
        dcp = dc_measure(dxy,XY[0,tt],XY[1,tt])
        dth = steering(Vs[1,tt+1],alpha_s,dcp,K)  #weathervaning #Vs[1,tt+1]
    ths[tt+1] = ths[tt]+dth
    ###environment
    #dxy = np.squeeze(np.array([np.cos(dth), np.sin(dth)]))
    dxy = ang2dis(XY[0,tt],XY[1,tt],ths[tt+1])
    XY[:,tt+1] = XY[:,tt] + dxy*dt
    Cs[tt+1] = environment(XY[0,tt+1],XY[1,tt+1])
    
# %%
plt.figure()
plt.plot(Cs)
plt.figure()
y, x = np.meshgrid(np.linspace(-10, 50, 60), np.linspace(-10, 50, 60))
plt.imshow(environment(x,y),origin='lower',extent = [-10,50,-10,50])
plt.plot(XY[0,:],XY[1,:],'blue')
plt.figure()
plt.plot(Vs.T)

# %%
###############################################################################
# %% Even simplar stochastic model
###############################################################################
# %% functions
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

def Kinetic_Ising(X,Phi,S,J,kbT):
    beta = 1/kbT
    N = len(S)
    Ht = X*Phi
    Theta = Current(Ht, J, S)
    S_ = np.zeros(N)
    for ss in range(N):
        S_[ss] = Transition(S[ss],Theta[ss],beta)
    #for tt in range(0,T-1):
    return S_

# %% dynamics
N = 2
J = np.random.randn(N,N)*0.05
J = np.array([[0.1,-2.],[-2.,0.1]])*.1
#J = np.array([[0.1,-2.],[-2.,0.1]])*0.1
h = np.random.randn(N)
kbT = .1
T = 10000
X = np.random.randn(N,T)
Phi = np.random.randn(N)*0.01
S = np.random.randint(0,2,size=(N,T))
S[S==0] = -1
for tt in range(T-1):
    S[:,tt+1] = Kinetic_Ising(X[:,tt],Phi,S[:,tt],J,kbT)
plt.figure()
plt.imshow(S,aspect='auto')

# %% for chemotaxis
lamb0 = 0.005
gamma0 = 0.5
alpha_p, alpha_s, alpha_g = -.00005, -.01, 0.0001
Cs = np.zeros(T)
ths = np.zeros(T)
XY = np.random.randn(2,T)
dxy = np.random.randn(2)
for tt in range(T-1):
    ###neural dynamics
    proj = Cs[tt]*Phi
    S[:,tt+1] = Kinetic_Ising(proj,Phi,S[:,tt],J,kbT)
    ###behavior
    if np.prod(S[:,tt+1])>0: #same state... change this~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        lambda_p = Pirouette(proj[0],alpha_p,lamb0)  #Pirouette #Cs[tt]
        if lambda_p>=np.random.rand():
            dth = turn_angle(proj[0],alpha_g,gamma0)  #bias
        else:
            dcp = dc_measure(dxy,XY[0,tt],XY[1,tt])
            dth = steering(proj[1],alpha_s,dcp,K)  #weathervaning #Vs[1,tt+1]
    else:
        lambda_p = Pirouette(proj[0],alpha_p*.1,lamb0)  #Pirouette #Cs[tt]
        if lambda_p>=np.random.rand():
            dth = turn_angle(proj[0],alpha_g,gamma0)  #bias
        else:
            dcp = dc_measure(dxy,XY[0,tt],XY[1,tt])
            dth = steering(proj[1],alpha_s*10.,dcp,K)  #weathervaning #Vs[1,tt+1]
            
    ###environment
    ths[tt+1] = ths[tt]+dth
    dxy = ang2dis(XY[0,tt],XY[1,tt],ths[tt+1])
    XY[:,tt+1] = XY[:,tt] + dxy*dt
    Cs[tt+1] = environment(XY[0,tt+1],XY[1,tt+1])

# %%
plt.figure()
plt.plot(XY[0,:],XY[1,:],'blue')
plt.figure()
plt.subplot(211)
plt.imshow(S,aspect='auto',interpolation='None')
plt.subplot(212)
plt.plot(np.prod(S,axis=0))
