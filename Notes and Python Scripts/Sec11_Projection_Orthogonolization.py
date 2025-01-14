# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:13:33 2024

@author: elisa
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# Sec11 topic: Projection and Orthogonolization:
    # projection of a dot on a line (in R2): beta = np.dot(a.T,b)/np.dot(a.T,a)
    # projection of a vector on a matrix (in R^n): x = inverse(A)@b
    # decompose w to orthogonal and parallel to v:
        # w-parallel-to-v = (w.T@v/v.T@v) * v
        # w-orthogonal-to-v = w - w-parallel-to-v
    # orthogonal matrices
        # typically noted in Q
        # all columns are orthogonal pairwise (e.g. dot product of any two columns is 0)
        # all columns have the norm/magnitude of 1
        # Q.T@Q = I
    # Gram-Schimdt process: turn any matrix into orthogonal
    # QR decomposition:
        # A = Q@R:
            # where Q is orthogonal
            # R is considered "Residual" matrix, R = Q.T@A
        # A.T@A = R.T@R
    

# 108. projection in R^2
b = np.array([4,1]) # a point
a = np.array([2,5]) # a line
beta = np.dot(a.T,b)/np.dot(a.T,a)

plt.plot(b[0],b[1],'ko',label='b')
plt.plot([0,a[0]],[0,a[1]], 'b', label='a')

plt.plot([b[0], beta*a[0]], [b[1],beta*a[1]], 'r--', label = r'b-$\beta$a')
plt.axis('square')
plt.legend()

# 109. projection in R^n

# 111. decompose vectors w into parallel and orthogonal to v
w = np.array([2,3])
v = np.array([4,0])

# compute w-parallel-to-v
w_p = (np.dot(w.T,v)/np.dot(v.T,v))*v
print(w_p)

# compute w-orthogonal-to-v
w_o = w - w_p
print(w_o)

# confirm result
print(w_o+w_p)
print(w_o@w_p)

# plot 4 vectors
plt.plot([0,w[0]],[0,w[1]],'r',label='w')
plt.plot([0,v[0]],[0,v[1]],'g',label='v')
plt.plot([0,w_o[0]],[0,w_o[1]],'r--',label='w_o')
plt.plot([0,w_p[0]],[0,w_p[1]],'r:',label='w_p')
plt.axis('square')
plt.axis([-1,5,-1,5])
plt.legend()

# 114. QR decomposition
A = np.array([[1,0],
              [1,0],
              [0,1]])

# "full" QR decomposition
Q,R = np.linalg.qr(A, 'complete')
print(Q.shape)

# "economy" QR decomposition
Q,R = np.linalg.qr(A, 'reduced')
print(Q.shape)

M = np.array([[1,1,-2],
              [3,-1,1]])
Q,R = np.linalg.qr(M, 'complete')

print(np.round(R,4))
print(np.round(Q.T@M,4))

# 115. Gram-Schimdt procedure           
# for square matrix
def gs_square(A):
    # A: input matrix
    m = A.shape[0]
    
    Q = np.zeros((m,m))
    for i in range(m):
        if i==0:
            Q[:,i] = A[:,i]
        else:
            v_i = A[:,i]
            for j in range(i):
                v_j = Q[:,j]
                v_proj_j = (np.dot(v_i,v_j)/np.dot(v_j,v_j))*v_j
                v_i = v_i - v_proj_j
            
            Q[:,i] = v_i

    for i in range(Q.shape[1]):
        Q[:,i] = Q[:,i]/np.linalg.norm(Q[:,i])
    
    return Q

# test
m=5
A = np.random.randn(m,m)
Q = gs_square(A)
Q.T@Q
plt.imshow(Q.T@Q)

Q_r, R = np.linalg.qr(A)
Q_r-Q

# 117. compute inverse through QR decompose
m = 10
A = np.random.randn(m,m)

A_inv = np.linalg.inv(A)

Q,R = np.linalg.qr(A)
A_inv2 = np.linalg.inv(R)@Q.T

















