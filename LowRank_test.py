# -*- coding: utf-8 -*-
"""
Created on Mon May 24 08:38:41 2021

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as signal

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=25) 
matplotlib.rc('ytick', labelsize=25) 

# %% low-rank network test
N = 30
dt = 0.1
T = 1000
lt = int(T/dt)
n = np.random.randn(N)
m = np.random.randn(N)
g= .1
tau = 0.1
noise = 0.
J = np.outer(n,m)/N + g**2*np.random.randn(N,N)/N
xs = np.zeros((N,lt))
rs = np.zeros((N,lt))
Is = np.ones((N,lt))#np.random.randn(N,lt)
for tt in range(lt-1):
    rs[:,tt] = np.tanh(xs[:,tt])
    xs[:,tt+1] = xs[:,tt] +  dt*(1/tau)*(-xs[:,tt] + J @ rs[:,tt] + n/N*Is[:,tt] + noise*np.random.randn(N)*np.sqrt(dt))
    
plt.figure()
plt.imshow(rs,aspect='auto')

# %%
def LowD(Is, J, dt, T, noise=0):
    N = J.shape[0]
    lt = int(T/dt)
    xs = np.zeros((N,lt))
    rs = np.zeros((N,lt))
    for tt in range(lt-1):
        rs[:,tt] = np.tanh(xs[:,tt])
        xs[:,tt+1] = xs[:,tt] +  dt*(1/tau)*(-xs[:,tt] + J @ rs[:,tt] + Is[:,tt] + noise*np.random.randn(N)*np.sqrt(dt))
    return rs, xs

def rotmnd(v,theta):
    nn = v.shape[0]
    M = np.eye(nn)
    for c in range(0,nn-2):
        for r in range(nn-1,c+1,-1):
            t = np.arctan2(v[r,c], v[r-1,c])
            R = np.eye(nn)
            #R[[r, r-1], [r, r-1]] = np.array([[np.cos(t), -np.sin(t)],[np.sin(t), np.cos(t)]])
            R[r,r] = np.cos(t)
            R[r,r-1] = -np.sin(t)
            R[r-1,r] = np.sin(t)
            R[r-1,r-1] = np.cos(t)
            v = R @ v
            M = R @ M
    R = np.eye(nn)
    #R[[n-1, n], [n-1, n]] = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    R[nn-2,nn-2] = np.cos(theta)
    R[nn-2,nn-1] = -np.sin(theta)
    R[nn-1,nn-2] = np.sin(theta)
    R[nn-1,nn-1] = np.cos(theta)
    M = np.linalg.solve(M, R @ M)
    return M
#function M = rotmnd(v,theta)
#    n = size(v,1);
#    M = eye(n);
#    for c = 1:(n-2)
#        for r = n:-1:(c+1)
#            t = atan2(v(r,c),v(r-1,c));
#            R = eye(n);
#            R([r r-1],[r r-1]) = [cos(t) -sin(t); sin(t) cos(t)];
#            v = R*v;
#            M = R*M;
#        end
#    end
#    R = eye(n);
#    R([n-1 n],[n-1 n]) = [cos(theta) -sin(theta); sin(theta) cos(theta)];
#    M = M\R*M;
    
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.abs(np.dot(v1_u, v2_u)), -1.0, 1.0))

def xcorr(x,y):
    """
    Perform Cross-Correlation on x and y
    x    : 1st signal
    y    : 2nd signal

    returns
    lags : lags of correlation
    corr : coefficients of correlation
    """
    corr = signal.correlate(x, y, mode="full")
    lags = signal.correlation_lags(len(x), len(y), mode="full")
    return lags, corr

# %%
basis = np.eye(N)
basis = basis[:,:N-2]
R = rotmnd(basis,np.pi/2)

# %%
## prep for orthogonal normal vector and rotation matrix
randM1 = np.random.randn(N,N)
randM2 = np.random.randn(N,N)
uu,ss,vv = np.linalg.svd(randM1)
qq,rr = np.linalg.qr(randM2)
n = uu[:,0] #np.random.randn(N)
m = uu[:,1] #@ qq#np.random.randn(N,N) #np.random.randn(N)
g= 1.
tau = 0.1
noise = 0.01
J = np.outer(n,m)/N + g**2*np.random.randn(N,N)/N + np.outer(uu[:,2], uu[:,3])/N*0
pe = 20
In_ang = np.convolve(np.random.randn(lt),np.ones(pe),'same')/pe*np.pi  #target
In_ang = (In_ang-np.min(In_ang))/(np.max(In_ang)-np.min(In_ang))
In_ang = np.cos(np.arange(0,T,dt)/pe)
In_ang = (In_ang + 1)/2
Im_ang = np.convolve(np.random.randn(lt),np.ones(pe),'same')/pe*np.pi  #decoding angle
Im_ang = (Im_ang-np.min(Im_ang))/(np.max(Im_ang)-np.min(Im_ang))
#Im_ang = signal.sawtooth(np.arange(0,T,dt)/pe, 0.5)
#Im_ang = np.cos(np.arange(0,T,dt)/pe)#*np.pi
#Im_ang = (Im_ang + 1)/2

#test = np.sin(np.arange(0,T,dt)/pe)
#ang = np.sin(np.arange(0,T,dt)/pe)*np.pi
#Is = np.zeros((N,lt))
#for ii in range(lt):
#    R = rotmnd(basis,ang[ii])  #rotating the angle
#    Is[:,ii] = n @ R * I[ii]
#nI = n#n @ rr #np.random.randn(N,N)
In = np.matmul(n[:,None], (In_ang)[None,:])
Im = np.matmul(m[:,None], (Im_ang)[None,:])
Is = In*(1-Im_ang*1) + Im*1
angs = np.zeros(lt)
for ii in range(lt):
    angs[ii] = angle_between(n,Is[:,ii])
rs, xs = LowD(Is,J,dt,T,noise)

rec_I = rs.T @ m
err = np.abs(In_ang-rec_I)**2
#rec_angs = np.zeros(lt)
#for ii in range(lt):
#    rec_angs[ii] = angle_between(n,rs[:,ii])

plt.figure()
plt.imshow(rs,aspect='auto')
plt.figure()
plt.plot(In_ang)
plt.plot(rec_I)
plt.plot(err,'--')
plt.plot(angs,'k--')
plt.figure()
plt.plot(Im_ang,err,'o')

ww = np.abs(np.diff(angs)/dt)
lag = np.abs(rec_I[:-1] - In_ang[:-1])**1
plt.figure()
plt.plot(ww,lag,'o',alpha=0.5)


# %%
###############################################################################
###############################################################################

# %% Low-rank fields
T = 1000
dt = 0.1
tau = 0.1
lt = int(T/dt)
N = 50
spe = 3.5
g = .1
dd = np.arange(0,N)
aa, bb = np.sin(dd/spe), np.cos(dd/spe)
J = 1*np.outer(aa,bb)/N + g**2*np.random.randn(N,N)/N
#plt.imshow(J,aspect='auto')
I = np.zeros((N,lt))#np.random.randn(N,lt)
I[22:28,0:100] = 1
x = np.zeros((N,lt))
def sigm(x):
    return 1/(1+np.exp(x))
for tt in range(lt-1):
    x[:,tt+1] = x[:,tt] + dt*(-x[:,tt]/tau + J @ sigm(x[:,tt]) - 0*x[:,tt]**3 + I[:,tt] + np.random.randn(N)*0.01)
    
plt.figure()
plt.imshow(x,aspect='auto')

