# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:24:39 2024

@author: elisa
"""

import numpy as np
import math
import sympy

# Sec8 topic: solve systems of equations
    # step1: convert systems of equations to matrix-vector equations
    # step2: create augmented matrix with coefficients and results
    # step3: Gaussian elimination for an upper triangle matrix
    # step4: back-substitute to solve
    
    # Echelon form:
    # pivots: number of pivots equals the matrix rank
    # reduced row echelon form

# 84. reduced-row echelon form (RREF or rref(A))
A = sympy.Matrix(np.random.randn(4,4))
A.rref() # returns a tuple
A.rref()[0] # the output rref matrix
A.rref()[1]
np.array(A.rref()[0]) # turns into np array

# 85. RREF of different matrix (random)
# - square
A = np.random.randint(0,100, size=(5,5))
rref = sympy.Matrix(A).rref()[0] # identity
rref

# - rectangular (tall, wide)
A = np.random.randint(0,100, size=(6,4))
rref = sympy.Matrix(A).rref()[0] 
rref # diagonal with 1 on the pivots

A = np.random.randint(0,100, size=(4,7))
rref = sympy.Matrix(A).rref()[0] 
rref # diagonal with 1 on the pivots

# - linear dependencies (column, row)
A = np.random.randint(0,100, size=(7,7))
A[:,-1] = A[:,0]
np.linalg.matrix_rank(A)
rref = sympy.Matrix(A).rref()[0] 
rref 

A = np.random.randint(0,100, size=(7,7))
A[-1,:] = A[0,:]
np.linalg.matrix_rank(A)
rref = sympy.Matrix(A).rref()[0] 
rref 
