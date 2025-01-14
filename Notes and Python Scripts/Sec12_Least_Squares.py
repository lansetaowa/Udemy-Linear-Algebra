# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 09:55:31 2024

@author: elisa
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as sio

# Sec12 topic: Least Square for Statistics Model Fitting:
    # X@b = y, solve X.T@X@b = X.T@y for beta
        # because X is rectangular

# 125. example: use least square to calculate mean
# data
data = np.array([[-4,0,-3,1,2,8,5,8]]).T
N = len(data)

# design matrix
X = np.ones((N,1))
# fit the model
b = np.linalg.solve(X.T@X,X.T@data)

# compare against the mean
m = np.mean(data)

# print the results
print(b,m)

# compute the model-predicted values
yHat = X@b

# plot data and model prediction
plt.plot(np.arange(1,N+1),data,'bs-',label='Data')
plt.plot(np.arange(1,N+1),yHat,'ro--',label='Model pred.')

plt.legend()


# design matrix
X = np.concatenate( [np.ones([N,1]),
                     np.array([np.arange(0,N)]).T
                     ],axis=1)
# fit the model
b = np.linalg.solve(X.T@X,X.T@data)

# compute the model-predicted values
yHat = X@b

# plot data and model prediction
plt.plot(np.arange(1,N+1),data,'bs-',label='Data')
plt.plot(np.arange(1,N+1),yHat,'ro--',label='Model pred.')

plt.legend()

# 126. example2: 
data = sio.loadmat("data\EEG_RT_data.mat")
rts = data['rts'][0] # reaction time, 99
eeg = data['EEGdata'] # 30*99
frex = data['frex'][0] # frequency, 99

nTrials = len(rts)
nFreq = len(frex)

# fitting one model of eeg to rts
X = np.concatenate((np.ones((nTrials-1,1)), 
                    np.reshape(rts[:-1], (nTrials-1,1)), # previous reaction time
                    np.reshape(eeg[5,:-1], (nTrials-1,1))),
                   axis=1)

y = rts[1:] # current reaction time

beta1 = np.linalg.solve(X.T@X, X.T@y)
beta2 = np.linalg.lstsq(X,y,rcond=None)

print(beta1)
print(beta2)

# 127. least square via QR decomposition
m=10
n=3

X = np.random.randn(m,n)
y = np.random.randn(m,1)

beta1 = np.linalg.solve(X.T@X, X.T@y)

Q,R = np.linalg.qr(X)
beta2 = np.linalg.inv(R)@Q.T@y

print(beta1-beta2)






















