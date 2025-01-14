# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 09:18:36 2024

@author: elisa
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import sympy

# Sec10 topic: Matrix Inverse:
    # inverse(A)*A = I
    # inverse(ABC) = inverse(C)*inverse(B)*inverse(A)
    # invertible if: 1. square matrix; 2. full rank
    # if possible, should avoid explicitly using inverse in Python because of potential inaccuracy
    # MCA to compute any inverse:
        # 1. M: the minor matrix, a matrix of determinant
        # 2. C: the cofactors matrix, hadamard-multiply with a matrix with alternating signs
        # 3. A-1: the adjugate matrix, C.T divided by determinant of original matrix
    # using rref of augmented matrix
    # left and right inverse for rectangular matrix:
        # tall matrix: m*n, m>n, has a left inverse if column full-rank
        # wide matrix: m*n, m<n, has a right inverse if row full-rank
    # pseudo-inverse: A*, A*@A is close to identical matrix
        # pseudo-inverses are not unique
    # deal with rank-deficient matrix:
        # dim reduction with PCA, etc., so that it has a true inverse
        # then project back to the full space
        
        
# 96. computing inverse
m=5
A = np.random.randint(low=0,high=100,size=(m,m))
A_inv = np.linalg.inv(A)

plt.imshow(A_inv@A)

# 99. implement MCA to compute inverse
# a matrix to start with
m=5
A = np.random.randint(low=0,high=100,size=(m,m))

# wrap into a function
def inv_mca(input_mat):
    if np.linalg.det(input_mat) == 0:
        print('not invertible')
    else:
        # get dimension
        m = input_mat.shape[0]
        
        # M step: minor matrix with determinants
        minor = np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                A_del1 = np.delete(A, i, axis=0)
                A_del2 = np.delete(A_del1, j, axis=1)
                det = np.linalg.det(A_del2)
                minor[i,j] = det

        # C step: cofactor matrix
        checkerbox = np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                checkerbox[i,j] = (-1)**(i+j)

        cofactor = minor*checkerbox

        # A step: adjugate matrix divided by det to compute inverse
        A_det = np.linalg.det(A)
        inverse = cofactor.T / A_det
        
        return inverse

# check result
A_inv = inv_mca(A)
A_inv2 = np.linalg.inv(A)
A_inv - A_inv2

plt.imshow(A_inv@A)

# 100. inverse via rref
m = 4
A = np.random.randint(low=0,high=100,size=(m,m))

# compute inverse with rref
def inv_rref(input_mat):
    if np.linalg.det(input_mat) == 0:
        print('not invertible')
    
    else:
        m = input_mat.shape[0]
        
        # augmented with diagonal matrix
        augmented = sympy.Matrix(np.concatenate((A, np.eye(m)), axis=1))
        
        # rref to get inverse on the right side
        rref = augmented.rref()[0]
        inverse = np.array(rref)[:,m:]
        
        # convert dtype
        inverse = inverse.astype('float64')
        
        return inverse

# check result
A_inv = inv_rref(A)
A_inv2 = np.linalg.inv(A)
A_inv - A_inv2

plt.imshow(A_inv@A)

# 101. diagonal matrix inverse
m = 5
diag = np.random.randint(low=1,high=10,size=m)

M_diag = np.zeros((m,m))
for i in range(m):
    M_diag[i,i] = diag[i]

print(M_diag)
print(np.linalg.inv(M_diag))

# 103. one-sided inverse for rectangular matrices
# m>n, left inverse
m=6
n=3

A = np.random.randn(m,n)
A_leftinv = (np.linalg.inv(A.T@A))@A.T
A_leftinv@A

# m<n, right inverse
m=3
n=6

A = np.random.randn(m,n)
A_rightinv = A.T@np.linalg.inv(A@A.T)
A@A_rightinv

# 105. pseudo-inverse
m = 4
A = np.random.randint(low=0,high=100,size=(m,m))
A[:,-1] = A[:,1]
np.linalg.matrix_rank(A)

A_pseudoinv = np.linalg.pinv(A)
plt.imshow(A_pseudoinv@A)
plt.imshow(A@A_pseudoinv)

# 106. pseudo-inverse for invertible matrix
# square
m = 4
A = np.random.randint(low=0,high=100,size=(m,m))
A_inv = np.linalg.inv(A)
A_pseudoinv = np.linalg.pinv(A)
A_inv - A_pseudoinv

# rectangular
m=6
n=3

A = np.random.randn(m,n)
A_leftinv = (np.linalg.inv(A.T@A))@A.T
A_pseudoinv = np.linalg.pinv(A)
A_leftinv - A_pseudoinv











