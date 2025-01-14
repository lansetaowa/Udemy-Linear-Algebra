# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:00:46 2024

@author: elisa
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# Sec6 topic: Matrix Rank
    # rank(A), or r: dimensionality of information
    # rank(A) is a non-negative int, 0<=r<=min(m,n)
    # definition: the largest # of rows/columns that can form a linearly independent set
    # methods to compute rank:
        # 1. the number of columns in a linearly independent set
        # 2. apply row reduction to make the matrix echelon form, and count the number of pivots
        # 3. compute singular value decomposition and count the number of non-zero singular values
        # 4. compute eigen decomposition and count the number of non-zero eigen values
    # limit of rank(A+B) is rank(A)+rank(B)
    # limit of rank(A@B) is min(rank(A),rank(B))
    # rank(A) = rank(A.T) = rank(A@A.T) = rank(A.T@A)
        # A.T@A is symmetric, square
    # shifting a matrix with a noise to make it full-rank
        # A~ = A + lambda*I

# 63. computing rank
m=4
n=6

A = np.random.randn(m,n)
M = [[1,1,2],
     [2,2,4],
     [2,3,5]]

np.linalg.matrix_rank(A)
np.linalg.matrix_rank(M)

# adding noise to M, a reduced matrix, making it full-rank
noiseamp= 0.0000001
B = M + noiseamp*np.random.randn(3,3)

np.linalg.matrix_rank(B)

# 65. reduced-rank matrix via multiplication
# generalize the procedure to create m*n matrix with rank r

A1 = np.random.randn(10,4)
A2 = np.zeros((10,6))

A = np.concatenate((A1, A2), axis=1)
np.linalg.matrix_rank(A)

B = np.eye(10)
np.linalg.matrix_rank(A@B)

def rankr_mat(m,n,r):
    # assume m<=n
    if r>m:
        print('r must not be larger than m')
    elif r==m:
        M1 = np.random.randn(m,m)
        M2 = np.concatenate((np.eye(m), np.zeros((m,n-m))), axis=1)
        M = M1@M2
        print(np.linalg.matrix_rank(M))
        return M
    elif r<m:
        M1 = np.concatenate((np.random.randn(m,r), np.zeros((m,m-r))), axis=1)
        M2 = np.concatenate((np.eye(m), np.zeros((m,n-m))), axis=1)
        M = M1@M2
        print(np.linalg.matrix_rank(M))
        return M

def rankr_mat2(m,n,r):
    M1 = np.random.randn(m,r)
    M2 = np.random.randn(r,n)
    M = M1@M2
    return M

M = rankr_mat(5, 6, 3)
M = rankr_mat2(5, 6, 3)

# 66. rank of scaler multiplied-matrix
m=5
r=3
M1 = np.random.randn(m,m) # full-rank
M2 = np.random.randn(m,r)@np.random.randn(r,m) # reduced-rank

lda = 0.4
np.linalg.matrix_rank(M2)
        
# 67. rank of A.T@A
m=4
n=2

A = np.random.randn(m,n)
A.shape
np.linalg.matrix_rank(A)
(A@A.T).shape
np.linalg.matrix_rank(A@A.T)
(A.T@A).shape
np.linalg.matrix_rank(A.T@A)      
  
# 69. making a matrix full-rank by shifting: A~ = A + lambda*I
m = 30
A = np.random.randn(m,m)
A = np.round(A.T@A)
np.linalg.matrix_rank(A)

# reduce the rank
A[:,0] = A[:,1]  
np.linalg.matrix_rank(A)  
  
# shift
l = 0.01
B = A + l*np.eye(m)
np.linalg.matrix_rank(B)  
    
# 70. determine if v is in the span of the sets
v = np.array([[1,2,3,4]]).T

S = np.array([[4,3,6,2],[0,4,0,1]]).T 
T = np.array([[1,2,2,2],[0,0,1,2]]).T

np.linalg.matrix_rank(S)
np.linalg.matrix_rank(T)

np.linalg.matrix_rank(np.concatenate((S, v), axis=1))    
np.linalg.matrix_rank(np.concatenate((T, v), axis=1))    
 
    
  
    
  
    
  
    
  