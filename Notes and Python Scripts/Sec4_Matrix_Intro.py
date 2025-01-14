# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:20:06 2024

@author: elisa
"""

import numpy as np

# Sec4 topic: Matrix Introduction
    # matrix diagonal/off-diagonal
    # block matrix
    # size: M*N
    # a zoo of matrices:
        # square: M*M
        # rectangular: M*N (M != N)
        # symmetric & skew-symmetric (sign flipped)
        # I: identity matrix (all 1's on the diagonal, and 0's for the rest)
        # 0: zero matrix
        # diagonal matrix (0's for all off-diag)
        # triangular matrix, upper & lower
        # concatenate/augmented
    # matrix addition and subtraction
    # shift a matrix with identity matrix, only changes diagonal elements
    # matrix scaler multiplication
    # transpose
    # complex matrix: transpose .T and Hermitian transpose .H
    # diagonal and trace
    # matrix broadcast arithmetic

# 32. a zoo of matrices
# square
S = np.random.randint(low=1,high=5,size=(4,4))

# rectangular
R = np.random.randint(low=1,high=5,size=(2,4))
R2 = np.random.randn(2,4)

# identity matrix
I = np.eye(5)

# zero matrix
Z = np.zeros(shape=(4,2))
Z_c = np.zeros(shape=(4,2), dtype=complex)

# diagonal matrix
D = np.diag(np.array([2,5,3,4,7]))

# triangular matrix
Tu = np.triu(S)
Tl = np.tril(S)

# 33. addition/subtraction
A = np.random.randint(low=100, size=(5,3))
B = np.random.randint(low=100, size=(5,3))
C = np.random.randint(low=100, size=(5,3))

A+B-C

# shifting a matrix
l = 0.3 # lambda
N = 5
D = np.random.randn(N,N)

D_shifted = D + l*np.eye(N)

# 36. transpose
M = np.random.randint(low=100, size=(5,3))
M
M.T
M.T.T
np.transpose(M)

# transpose a complex matrix
C = np.array([[4+1j, 3, 2-4j],
              [5+6j, 4-2j, 1]])

# regular transpose
C.T
np.transpose(C)

# Hermitian transpose
np.matrix(C).H

# 38. diagonal and trace
M = np.round(np.random.randn(4,4)*5, 0)

# extract diagonal elements
d = np.diag(M) # input a matrix, output its diag vector
D = np.diag(d) # input a vector, output a diag matrix

# trace as sum of diag elements
tr = np.trace(M)
np.sum(np.diag(M))

# 39. linearity of trace
A = np.random.randint(low=100, size=(5,5))
B = np.random.randint(low=100, size=(5,5))

print(np.trace(A) + np.trace(B))
print(np.trace(A+B))

l = np.random.rand() # lambda
print(np.trace(l*A))
print(np.trace(A)*l)

# 40. matrix broadcast arithmetic
A = np.arange(1,13).reshape(3,4)

# add two vectors
v1 = [10,20,30,40]
v2 = [100,200,300]

A+v1 # broadcast on rows 
A+np.array(v2).reshape(3,1)




















