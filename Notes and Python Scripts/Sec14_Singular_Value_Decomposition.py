# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 12:47:58 2024

@author: elisa
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# Sec14 topic: Singular Value Decomposition:
    # SVD: A = U@S@V.T
    # singular value interpretation: sum is the variance of A
    # use svd for computing pseudo-inverse: V@inv(S)@U.T
    # condition number of a matrix: ratio between the max and min singular value

# 147. SVD
A = [[3,0,5],[8,1,3]]

U,S,V = np.linalg.svd(A)
print(U)
print(S)
print(V) # V is actually V.T, therefore A = U@np.diag(S)@V

np.allclose(U@np.diag(S)@V[:2,:],A)

# 148. eig vs. svd for square symmetric matrices
n=5
A = np.random.randn(n,n)
B = A+A.T 

# eig(W,L)
L,W = np.linalg.eig(B)
U,S,V = np.linalg.svd(B)

# compare all matrices
fig, ax = plt.subplots(2,3,figsize=(15,10))

ax[0,0].imshow(W)
ax[0,0].set_title('W-eig')

ax[0,1].imshow(np.diag(L))
ax[0,1].set_title('$\lambda$-eig')

ax[1,0].imshow(U)
ax[1,0].set_title('U-svd')

ax[1,1].imshow(np.diag(S))
ax[1,1].set_title('$\Sigma$-svd')

ax[1,2].imshow(V.T)
ax[1,2].set_title('V-svd')

# 149. relation between singular values and eigenvalues

# 159. create matrix with desired conditional numbers



