# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 14:38:30 2019

@author: kevin
"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifierCV
#import Adaptive_Subunits

sns.set_style("white")
sns.set_context("talk")

# %% initialize model structure and parameters
###These should not be changed/re-run when evaluating across stimuli
#time
dt = 0.1  #in units of 10 ms for speed
T = 1000
time = np.arange(0,T,dt)
#time scales
tau_c = 5  #fast time scael in 10 ms
tau_s = 500  #slow time scale
#connectivity (image to subunits)
gamma = 0.05  #overlapping of subunit receptive field
p0 = 100  #input synaptic strength
q = 0.7  #probability to connect subunits and neuron (sparsoty of connection)
#q0 = 1  #strength of subunit-neuron synapse
K = 50  #number of neurons
m_avg = 5  #number of subunits per neurons
M = int(K*m_avg)  #total number of subunits
Q = np.random.rand(K,M)  #subunit-neuron conection
Q[Q>q] = 1
Q[Q<=q] = 0
#check for input unity
for kk in range(K):
    sumsyn = sum(Q[kk,:])
    if sumsyn==0:
        Q[kk,:] = 1/M
    else:
        Q[kk,:] = Q[kk,:]/sumsyn      
#adding inhibitory connnections
pi = 0.1   #probability of being inhibitory
signM = np.random.rand(Q.shape[0],Q.shape[1])
signM[signM>pi] = 1
signM[signM<=pi] = -1
Q = Q*signM
m = 5  #subunits per stimulus pool
N = int(M/m)  #possible unique images
P = np.ones((N,M))*gamma  #image-subunit connection
temp = 0
for nn in range(N):
    #P[nn,temp:temp+m] = 1
    P[nn,np.random.randint(0,M,(m))] = p0
    temp = temp + m
    
# %% dynamics
def stimuli():
    """
    Making marks for repeating sequences
    """
    ###marks
    fnum = 4  #number of unique frames in a sequence
    dur = int(20/dt)  #duration of each frame in ms
    L = 10  #repeating the sequence
    mark = np.random.choice(N, fnum, replace=False)  #choosing mark for the sequence
    seq_trans = mark[-1]
    mark2 = np.repeat(mark,dur,axis=0)
    marks_ = np.matlib.repmat(np.expand_dims(mark2,axis=1),L,1).reshape(-1)  #resacled marks
    novel = np.arange(0,N)
    novel = np.delete(novel,mark)  #remove used images
    novel_mark = np.random.choice(N-fnum, fnum, replace=False)  #novel frames in the novel sequence
    marksub = novel[novel_mark]
    #marksub = np.array([0,1,2,3])  #np.array([0,1,2,3,8,5,6,7])
    marksub = np.repeat(marksub,dur,axis=0)  #resacled for novel marks
    marks = np.concatenate((marks_, marksub))  #concatenating the last period
    ### mapping to identify subunits
    if len(marks)>len(time):
        marks = marks[:len(time)]
    else:
        marks = np.concatenate((marks,-np.ones(len(time)-len(marks))))
    seq_mark = np.zeros_like(marks)
    seq_mark[np.where(marks==seq_trans)[0]] = 1  #delta function at that subunit
    return marks, seq_mark  #marks for each frame and seq_mark for each sequence

def NL(x):
    """
    Nonlinearity, using ReLu here
    """
    return np.array( [max(xx,0) for xx in x] )

def subunit_model(marks):
    """
    Input parameters for the subunit model and return time trace
    with a random set of stumulus set
    """
    #dynamics
    xs = np.zeros((M,len(time)))  #subunit activity
    alphas = np.zeros((M,len(time)))  #subunit adaptation
    ys = np.zeros((K,len(time)))  #neurons
    xs[:,0] = np.random.rand(M) #0.1
    alphas[:,0] = np.random.rand(M) #.1
    ys[:,0] = np.random.rand(K) #0.1
    noise = 0.005  #noise strength for variation
    ### w/ inhibitory feedforward
    #kk = np.random.randn(subu)
    for tt in range(0,len(time)-1):
        I_index = int(marks[tt])  #stimulus index
        if I_index<0:
            xs[:,tt+1] = xs[:,tt] + dt*(1/tau_c)*(-xs[:,tt] + alphas[:,tt]*0) + np.random.randn(M)*noise*1  #subunit
            alphas[:,tt+1] = alphas[:,tt] + dt*(1/tau_s)*((1-alphas[:,tt]) - alphas[:,tt]*0)  #adaptation
        else:
            xs[:,tt+1] = xs[:,tt] + dt*(1/tau_c)*(-xs[:,tt] + alphas[:,tt]*P[I_index,:]) + np.random.randn(M)*noise*1  #subunit
            alphas[:,tt+1] = alphas[:,tt] + dt*(1/tau_s)*((1-alphas[:,tt]) - alphas[:,tt]*P[I_index,:])  #adaptation
        ys[:,tt+1] = ys[:,tt] + dt*(1/tau_c)*(-ys[:,tt] + NL(np.matmul(Q,xs[:,tt])) ) + np.random.randn(K)*noise #neurons
    return ys

# %% Measurements
def Measure_rij(rep, dur):
    """
    Compute neurons x sequences response profile for decoding
    rep repetition of the same sequence
    dur as the time window to measure yss
    yss[i][j] for i tirals and j=0 being transient and j=1 being sustain periods
    """
    #max_frameID = np.max(np.unique(marks))-2
    #dur = 500
    marks, seq_mark = stimuli()
    transitions = np.where(np.diff(seq_mark)>0)[0] #np.where(np.diff(marks)<0)[0]  #should be otherwize
    trans = transitions[5]  #transient sequence
    sust = transitions[-1]  #sustain sequence
    yss = []
    for jj in range(0,rep):
        marks, seq_mark = stimuli()
        ys = subunit_model(marks)
        yss.append(np.array([ys[:,sust:sust+dur], ys[:,trans:trans+dur]]))  #sustain and transient response
#        yss.append(np.array([ys[:,trans[0]:trans[0]+dur],ys[:,sust[0]:sust[1]]]))
    return yss  #2 x K x T (0 for transient and 1 for sustain measurements)

def Measure_rijk(seq, rep, dur):
    """
    Compute neurons x sequences response profile for decoding
    seq sets of different sequences
    rep repetition of the same sequence
    dur as the time window to measure yss
    yss[i][j][k] for ith sequence, jth repetition for that sequence, and k=0 for sustained k=1 for transient
    """
    #max_frameID = np.max(np.unique(marks))-2
    #dur = 500
    marks, seq_mark = stimuli()
    transitions = np.where(np.diff(seq_mark)>0)[0] #np.where(np.diff(marks)<0)[0]  #should be otherwize
    sust = transitions[5]  #sustain sequence
    trans = transitions[-1]  #transient sequence
    ys_seq_rep = []
    for ii in range(0,seq):
        marks, seq_mark = stimuli()  #construct one sequence stimuli
        y_rep = []
        for jj in range(0,rep):
            ys = subunit_model(marks)  #stimulate subunits with this set of input
            y_rep.append(np.array([ys[:,sust+dur:sust+dur*4], ys[:,trans+dur:trans+dur*4]]))  #sustain and transient response
        ys_seq_rep.append(y_rep)
#        yss.append(np.array([ys[:,trans[0]:trans[0]+dur],ys[:,sust[0]:sust[1]]]))
    return ys_seq_rep  #seq list of rep list of (2 x K x dur) (0 for sustained and 1 for transient measurements)
    
def MLE_decording(r_ij):
    """
    Neural response r from neuron i under sequence j and estimates the 
    optimal linear weight and returns predicted j'
    """
    #n_neurons = r_ij.shape[0]
    n_sequences = r_ij.shape[1]
    ###following the method in preprint for un-optimized decoding weight
    w_ = r_ij - np.repeat(np.mean(r_ij,axis=1),n_sequences).reshape(r_ij.shape)
    j_est = np.argmax(w_.T @ r_ij, axis=0)
    return j_est


# %% EXP
sequences = 10
repetitions = 10
dur = 200  #duration per image
yss = Measure_rijk(sequences, repetitions, dur)

# %% storing measurements
trials_t = np.zeros((sequences, repetitions, K))  #measurements for seq rep and from K neurons
trials_s = np.zeros((sequences, repetitions, K))
for ss in range(0,sequences):
    for rr in range(0,repetitions):
        trials_s[ss,rr,:] = np.mean(yss[ss][rr][0], axis=1)
        trials_t[ss,rr,:] = np.mean(yss[ss][rr][1], axis=1)

# %% sorting and training
def sort_heat(unsorted):
    """
    Sorting the cell-sequence response heatmap with the max value of activation
    """
    max_r = np.max(unsorted, axis=0)  #max response of each neuron
    temp = unsorted[:, np.argsort(max_r)]
    max_i = np.max(temp, axis=1)  #max reponse to an image
    temp2 = temp[np.argsort(max_i),:]
    sortedd = np.fliplr(np.flipud(temp2))
    return sortedd

sorted_s = np.array([sort_heat(trials_s[:,ii,:]) for ii in range(repetitions)])
sorted_t = np.array([sort_heat(trials_t[:,ii,:]) for ii in range(repetitions)])

plt.figure()
plt.subplot(211)
plt.title('sustained type')
plt.imshow(np.mean(sorted_s,axis=0), aspect='auto')
plt.subplot(212)
plt.title('transient type')
plt.imshow(np.mean(sorted_t,axis=0),aspect='auto')
plt.xlabel('sorted cell index')
plt.ylabel('sorted stimuli index')

# %% training
#clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1],cv=5)
clf = SVC(gamma='auto')
yy = np.arange(0,sequences)  #IDs of the sequences
cellnum = np.arange(2,K,5)
boot = 10
performance_t = np.zeros((boot,len(cellnum)))
performance_s = np.zeros((boot,len(cellnum)))
for bb in range(boot):
    temp = np.random.choice(repetitions,2,replace=False)
    for ki,kk in enumerate(cellnum):
        temp_rt = sorted_t[temp[0],:,:kk] #[:,None]
        temp_rs = sorted_s[temp[0],:,:kk]
        clf.fit(temp_rt, yy)
        performance_t[bb,ki] = clf.score(sorted_t[temp[1],:,:kk],yy)
        clf.fit(temp_rs, yy)
        performance_s[bb,ki] = clf.score(sorted_s[temp[1],:,:kk],yy)
        #performance_s.append(clf.score(sorted_s[temp[1],:,:kk],yy))

plt.title('population decoding')
plt.plot(cellnum,performance_t,'b-o')
plt.plot(cellnum,performance_s,'r-o')
plt.xlabel('number of cells')
plt.ylabel('performance')
plt.legend()

# %% OLD
###############################################################################
# %% experimental trials
trials = 30
dur = 200*2
yss = Measure_rij(trials, dur)
trials_t = []
trials_s = []
for k in range(0,trials):
    trials_t.append(np.mean(yss[k][0],axis=1))
    trials_s.append(np.mean(yss[k][1],axis=1))
trials_t = np.array(trials_t)
trials_s = np.array(trials_s)

# %% sorting confusion matrix
plt.figure()
plt.subplot(211)
plt.title('sustained type')
#temp2 = temp[np.argmax(temp, axis=0)[:temp.shape[0]],:]
plt.imshow(sort_heat(trials_s), aspect='auto')
plt.subplot(212)
plt.title('transient type')
plt.imshow(sort_heat(trials_t),aspect='auto')
#plt.imshow(trials_t[np.argmax(trials_t, axis=0),:],aspect='auto')
plt.xlabel('sorted cell index')
plt.ylabel('sorted stimuli index')

# %% clustering sequential response
sorted_s = sort_heat(trials_s)
sorted_t = sort_heat(trials_t)
kmeans_t = KMeans(n_clusters=trials).fit(trials_t)

# %%
##### linear classifiers
seq_id = np.arange(0, sorted_s.shape[0])
clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1],cv=10).fit(sorted_s[:,:10], seq_id)
clf.score(sorted_s[:,:10], seq_id)

# %%
clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1],cv=5)
yy = np.arange(0,trials)
performance_t = []
performance_s = []
cellnum = np.arange(2,K,5)
for kk in cellnum:
    temp_rt = sorted_t[:,:kk]
    temp_rs = sorted_s[:,:kk]
    clf.fit(temp_rt, np.arange(0,trials)) 
    performance_t.append(clf.score(temp_rt,yy))
    clf.fit(temp_rs, np.arange(0,trials)) 
    performance_s.append(clf.score(temp_rs,yy))
 
plt.title('population decoding')
plt.plot(cellnum,performance_t,'-o',label='transient')
plt.plot(cellnum,performance_s,'-o',label='sustain')
plt.xlabel('number of cells')
plt.ylabel('performance')
plt.legend()

# %% SVM classification
clf = SVC(gamma='auto')
clf.fit(sorted_t, np.arange(0,trials)) 
print(clf.score(sorted_s, np.arange(0,trials)) )
clf.fit(sorted_s, np.arange(0,trials)) 
print(clf.score(sorted_t, np.arange(0,trials)) )

# %% scaling of population coding
clf = SVC(gamma='auto')
yy = np.arange(0,trials)
performance_t = []
performance_s = []
cellnum = np.arange(2,K,5)
for kk in cellnum:
    temp_rt = trials_t[:,:kk]
    temp_rs = trials_s[:,:kk]
    clf.fit(temp_rt, np.arange(0,trials)) 
    performance_t.append(clf.score(temp_rs,yy))
    clf.fit(temp_rs, np.arange(0,trials)) 
    performance_s.append(clf.score(temp_rt,yy))
 
plt.title('population decoding')
plt.plot(cellnum,performance_t,'-o',label='sustain')
plt.plot(cellnum,performance_s,'-o',label='transient')
plt.xlabel('number of cells')
plt.ylabel('performance')
plt.legend()

# %% with cross-validation
###############################################################################
# %% SVM classification
clf = SVC(gamma='auto')
half = int(trials/2)
clf.fit(trials_t[:half,:], np.arange(0,half)) 
print(clf.score(trials_t[half:,:], np.arange(0,half)) )
clf.fit(trials_s[:half,:], np.arange(0,half)) 
print(clf.score(trials_s[half:,:], np.arange(0,half)) )

# %% scaling of population coding
clf = SVC(gamma='auto')
yy = np.arange(0,half)
performance_t = []
performance_s = []
for kk in range(5,K,5):
    temp_rt = trials_t[:half,:kk]
    temp_rs = trials_s[:half,:kk]
    clf.fit(temp_rt, np.arange(0,half)) 
    performance_t.append(clf.score(trials_t[half:,:kk],yy))
    clf.fit(temp_rs, np.arange(0,half)) 
    performance_s.append(clf.score(trials_s[half:,:kk],yy))
 
plt.title('population decoding')
plt.plot(performance_t,'-o',label='transient')
plt.plot(performance_s,'-o',label='sustain')
plt.xlabel('number of cells')
plt.ylabel('performance')
plt.legend()