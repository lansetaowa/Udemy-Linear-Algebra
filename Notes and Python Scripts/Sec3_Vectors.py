# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 22:35:45 2024

@author: elisa
"""

import numpy as np

#### Note both the algebraic and geometric meaning 
#### of each application

# Sec3 topic: Vector
    # Vector: an ordered list of numbers
        # column vectors, can be transposed to row vector
        # row vectors
    # dimensionality: the number of numbers
    # vector addition and substraction
    # scaler multiplication
        # scaler is a number, usually represented by greek letters, such as alpha
    # vector multiplication: dot/Hadamard/outer/cross product
        # dot product is a single number
            # distributive property 分配律 a*(b+c) = a*b + a*c
            # not associative property 不符合结合律， (a*b)*c != a*(b*c)
        # Hadamard multiplication product is the same length vector
            # np.multiply
        # outer product:
            # creates a N*M matrix product, np.outer
        # cross product:
            # for vectors, only defined for 2 3D vectors, resulting in another 3D vector
            # np.cross
    # length and magnitude of a vector
    # vector orthogonality: two vectors meet at 90 degrees
    # complex vector/matrix
        # conjugate vector
    # unit vector: length is 1
        # zero vector has no unit vector
    # field: a set on which addition, substraction, multiplication and division can be applied
        # R: real numbers
        # C: complex numbers
        # Z: integers
    # subspace:
        # subspace of vectors: a linear combination of vectors
        # contain the zero vector
        # must be closed under addition and scaler multiplication
    # span
    # linear independence/dependence



v1 = np.array([1,4])
v2 = np.array([5,3])

a = np.random.randn(5)
b = np.random.randn(5)
c = np.random.randn(5)

# 8. addition/substraction
v1+v2
v1-v2

# 9. scaler multiplication
v1*7

# 10. dot product
# 11. not associative
np.dot(np.dot(a,b),c)
np.dot(a, np.dot(b,c))

# distributive
np.dot(a,(b+c))
np.dot(a,b)+np.dot(a,c)

# commutative
np.dot(a,b)
np.dot(b,a)

############### exercise；
# dot product for each pair of columns between two 4*6 matrix 
M1 = np.random.randn(24).reshape(4,6)
M2 = np.random.randn(24).reshape(4,6)

for i in range(M1.shape[1]):
    for j in range(i,M2.shape[1]):
        dot_prod = np.dot(M1[:,i],M2[:,j])
        print(f'prod_{i}_{j}:', dot_prod)

# 14. length/norm of a vector
len_v1 = np.sqrt(sum(np.multiply(v1,v1)))
len_v12 = np.linalg.norm(v1) 

# len(v1) will give the dimension of v1

# 15. geometry of dot product
angle = np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    
# 18. Hadamard multiplication
np.multiply(a,b)
np.multiply(a,c)

# 19. outer product
np.outer(a,v1)
np.outer(v1,a)

# 20. cross product
x = [1, 2, 3]
y = [4, 5, 6]
np.cross(x, y)

# 21. complex numbers
import cmath
z = 5+3j
z2 = complex(4,3)
z.real
z.imag
cmath.phase(z)
z.conjugate() # conjugate

# conjugate of z
z_conj = 5-3j

# complex in numpy
z_np = np.complex(3,4)
np.linalg.norm(z_np)
np.transpose(z_np)*z_np
z_np.conjugate()*z_np

# complex vector
v = np.array([3, 4j, 5+6j, 2-8j])
v.T
np.transpose(v)
np.transpose(v.conjugate())*v
v.conjugate()*v

# 22. Hermitian transpose (aka conjugate transpose)
v.conjugate()

# 23. create unit vector
v1 = np.array([5,8,4,10])
mu = 1/np.linalg.norm(v1) # scaler to convert to unit
v1_unit = v1*mu
print(v1_unit)
    
################# 24. exercise:
# create two random integer vectors
v1 = np.random.randn(4)
v2 = np.random.randn(4)

# compute the lengths of each, and create dot product
v1_len = np.linalg.norm(v1)
v2_len = np.linalg.norm(v2)

dot_prod = np.dot(v1, v2)

# normalize vectors
v1_norm = v1/v1_len
v2_norm = v2/v2_len

# the magnitude of dot product
dpm = np.abs(dot_prod)

# the magnitude to two unit vectors
dpm2 = np.abs(np.dot(v1_norm,v2_norm))


