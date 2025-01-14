# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 19:56:14 2024

@author: elisa
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.linalg 

# Sec13 topic: Eigen Decomposition:
    # eigen-decomposition is only for square matrix
    # Av = lambda*v, (A-lambda*I)v = 0 vector
    # solve for eigenvalues: det(A-lambda*I) = 0  
    # eigenvector: 
        # invariant direction
        # subspaces with maximal covariance
        # each eigenvalue is the key to its eigenvector
        # for each lambda, find v that belongs to N(A-lambda*I)
        # eigenvector can flip signs
    # matrix diagnolization: A = V@L@inv(V), L is lambda diag matrix
    # distinct eigenvalues have distinct vectors
    # repeated eigenvalues may have different vectors (i.e. forms a eigenplane)
    # symmetric matrix:
        # eigenvectors are orthogonal to each other
        # all eigenvalues are real
    # singular matrix: at least one eigenvalue is 0
    # det(A) = the product of A's eigenvalues
    # trace(A) = the sum of A's eigenvalues
    # Generalized Eigendecomposition: for S and R, Sv = lambda*Rv
    

# 129. finding eigenvalues
A = [[1,5],[2,4]]

eigdecomp = np.linalg.eig(A)
eig_vals = eigdecomp[0] # eigenvalues, a list

# 131. eigenvalues for diag and triangular matrices
# diag matrix
n=5
A = np.diag(np.random.randn(n))
eig_vals = np.linalg.eig(A)[0]
print(eig_vals)

# upper triangular matrix
n=4
A = np.random.randn(n,n)
Aupper = np.triu(A)
eig_vals = np.linalg.eig(Aupper)[0]
print(eig_vals)

# lower triangular matrix
n=4
A = np.random.randn(n,n)
Alower = np.tril(A)
eig_vals = np.linalg.eig(Alower)[0]
print(eig_vals)

# 132. plot eigenvalues for random matrix
plt.figure(figsize=(6,6))
for i in range(100):
    n=50
    A = np.random.randn(n,n)
    eig_vals = np.linalg.eig(A)[0]
       
    for v in eig_vals:
        plt.plot(v.real, v.imag,'o', markersize=1)

# 133. Finding Eigenvectors
A = [[1,2],[2,1]]

eig_vals, eig_vecs = np.linalg.eig(A)
# L,W / D,V = np.linalg.eig(A)  common notation 1
print(eig_vals)
print(eig_vecs)

eigens = {}

# eigenvectors are of norm 1
np.linalg.norm(eig_vecs[:,0])
np.linalg.norm(eig_vecs[:,1])

# 135. Diagnolization, A = V@L@inv(V)
A = np.round(10*np.random.randn(4,4))

eig_vals, eig_vecs = np.linalg.eig(A)
Ap = eig_vecs @ np.diag(eig_vals) @ np.linalg.inv(eig_vecs)
print(A-Ap)

# 136. Matrix powers via diagnolization
A = np.random.randn(3,3)
print(np.linalg.matrix_power(A, 3)) # A@A@A

D,V = np.linalg.eig(A)
D = np.diag(D)
print(V@ np.linalg.matrix_power(D, 3) @ np.linalg.inv(V))

plt.subplot(121)
plt.imshow(np.linalg.matrix_power(A, 3))
plt.title('A3')

plt.subplot(122)
plt.imshow(np.real(V@ np.linalg.matrix_power(D, 3) @ np.linalg.inv(V)))
plt.title('A3 via eig decomp')

# 137. eigendecomposition for matrix diff
# claim: (A-B)v = lambda*v -> (A^2-AB-BA-B^2)v = lambda^2*v
n=3
A = np.random.randn(n,n)
B = np.random.randn(n,n)

eig_vals, eig_vecs = np.linalg.eig(A-B)
l,v = eig_vals[0], eig_vecs[:,0]

res1 = (A@A-A@B-B@A+B@B)@v
res2 = (l**2)*v
print(res1-res2)

# 140. eigendecomposition for symmetric matrix
n=5
A = np.random.randn(n,n)
A = A+A.T 
eig_vals, eig_vecs = np.linalg.eig(A)

plt.imshow(A)
plt.imshow(eig_vecs)
plt.imshow(eig_vecs@eig_vecs.T) # I matrix

# 142. reconstruct a matrix from eigenlayers
n=5
A = np.random.randn(n,n)
A = A+A.T 
eig_vals, eig_vecs = np.linalg.eig(A)

# norm of outer product of any v_i
i=3
np.linalg.norm(np.outer(eig_vecs[:,i],eig_vecs[:,i]))

# one layer of A as lvv'
eig_vals[i]*np.outer(eig_vecs[:,i],eig_vecs[:,i])

# summing up layers to get A
layers = [eig_vals[i]*np.outer(eig_vecs[:,i],eig_vecs[:,i]) for i in range(n)]
A_sum = sum(layers)
print(A-A_sum)

# 144. trace and det of a matrix
n=5
A = np.random.randn(n,n)
eig_vals, eig_vecs = np.linalg.eig(A)

print(np.trace(A))
print(sum(eig_vals))

print(np.linalg.det(A))
print(np.prod(eig_vals))

