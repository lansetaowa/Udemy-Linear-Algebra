# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:44:45 2024

@author: elisa
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# Sec9 topic: Matrix Determinant:
    # Notation: det(A)
    # only for square matrix, is a single value
    # det(A)=0 if linearly dependent (or rank-deficient, reduced-rank, singular)
    # has no inverse if det(A)=0
    # det(AB) = det(A)*det(B)

# 89. determinant of singular (reduced-rank) matrix
# 2*2 matrix
A = np.random.randint(0,100, size=(2,2))
A[:,1] = A[:,0]*3
np.linalg.det(A)

# m*m matrices
m = 6
A = np.random.randint(0,10000, size=(m,m))
A[:,1] = A[:,0]
np.linalg.det(A)

m = 6
A = np.random.randn(m,m)
A[:,1] = A[:,0]
np.linalg.det(A)

# 91. matrix determinant with row exchanges
A = np.random.randn(6,6)
# swap 2 rows: mth and nth row
row_list = list(range(A.shape[1]))
m,n = 3,4
row_list[m], row_list[n] = row_list[n], row_list[m]
A2 = A[row_list]

np.linalg.det(A)
np.linalg.det(A2) # flipped sign

# swap rows twice
A = np.random.randn(6,6)
row_list = list(range(A.shape[1]))
h,j,k = 2,3,4
row_list[h], row_list[j], row_list[k] = row_list[k], row_list[h], row_list[j]
A2 = A[row_list]

np.linalg.det(A)
np.linalg.det(A2) # same sign

# 93. det of shifted matrices
 
lda_list = np.linspace(0.01,0.1,30)
detavg_list = []
    
for lda in lda_list:
    det_list = []
    for i in range(5000):
        A = np.random.randn(20,20)
        A[:,-1] = A[:,0]
        A2 = A + lda*np.eye(20)
        det_A = np.abs(np.linalg.det(A2))
        det_list.append(det_A)
    
    det_avg = np.average(det_list)
    detavg_list.append(det_avg)
        
plt.plot(lda_list, detavg_list, 'o-')        
plt.xlabel('fraction of identity shifting')
plt.ylabel('det')

# 94. det of matrix product
# illustrate that det(AB) = det(A)*det(B), loop up to size of 40
for m in range(3,41):
    A = np.random.randn(m,m)
    B = np.random.randn(m,m)
    
    det1 = np.linalg.det(A@B)
    det2 = np.linalg.det(A)*np.linalg.det(B)
    
    print("diff for size of {0} is {1}".format(m, det1-det2))






