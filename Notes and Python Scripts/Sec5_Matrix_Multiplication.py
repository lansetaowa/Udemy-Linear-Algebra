# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:26:05 2024

@author: elisa
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# Sec5 topic: Matrix Multiplication
    # standard matrix multiplication
    # 4 ways to understand matrix multiplication (M*N, N*K -> M*K):
        # element perspective: the dot product between all combinations of 1st matrix's rows and 2nd matrix columns
        # layer perspective: the sum of outer product of 1st' cols and 2nd's rows
        # column perspective: for i in 1 to K, ith column is 1st matrix weighted by ith column of 2nd matrix
        # row perspective: for i in 1 to M, ith row is 2nd matrix weighted by ith row of 1st matrix
    # multiplication with a diagonal matrix
        # A*Diag: columns
        # Diag*A: rows
    # order of operation on matrices
        # (LIVE).T = (E.T)*(V.T)*(I.T)*(L.T)
        # T can be any operation
    # matrix-vector multiplication produces a vector
        # same direction as input vector
        # if a diag matrix, then post or pre multiplies a vector gives the same result
    # rotate matrix: [[cos, -sin],[sin, cos]]
    # if M@v = lambda*v, then lambda is eigenvalue, v is eigen vector
    # create a symmetric matrix:
        # S = A.T + A
        # S = A.T@A  or A@A.T
    # multiplication of two symmetric matrices is not symmetric
        # exception: diagonal is constant for 2*2 matrices
    # a diag matrix multiplies another diag creates a diag matrix
    # Frobenius dot product
        # three methods to calculate:
            # 1. element-wise multiplication, then add up
            # 2. vectorize matrix, then dot product
            # <A,B>F = trace(A.T@B)
        # vectorize a matrix: concatenate column-wise
        # matrix norm/Euclidean-norm: tr(A.T@A)
    # matrix norm
        # Euclidean norm/Frobenius norm: sqrt(trace(A.T@A))
        # induced 2-norm:
            # norm(A) = supreme( norm(A@x)/norm(x) ), x is non-zero vector
            # interpret: how much A scales vector x
        # schatten p-norm:
    # self-adjoint operator A satisfies: <Av,w> = <v,Aw>
        # A is m*m, symmetric
    # matrix asymmetry index (how asymmetric a matrix is)
        # A~ = (A-A.T)/2
        # asymmetry index = norm(A~)/norm(A)
            
            
    

# 41. standard matrix multiplication
m=4
n=3
k=6

A = np.random.randn(m,n)
B = np.random.randn(n,k)
C = np.random.randn(m,k)

np.matmul(A,B)
np.matmul(A,C)
np.matmul(A,C.T)
np.matmul(B,C)
np.matmul(B,C.T)

A*B # element-wise multip
A@B # matrix multip

# 42. 4 perspectives of matrix multiplication of A and B
# element perspective
AB_prod = np.zeros((A.shape[0], B.shape[1]))

for i in range(0, AB_prod.shape[0]):
    for j in range(0, AB_prod.shape[1]):
        AB_prod[i,j] = np.dot(A[i,:],B[:,j])

# layer perspective
AB_prod = np.zeros((A.shape[0], B.shape[1]))

for i in range(0, B.shape[0]):
    a = A[:,i] # A's ith column
    b = B[i,:] # B's ith row
    ab_outer = np.outer(a,b) # creates a layer
    AB_prod = AB_prod + ab_outer

# column perspective
AB_prod = np.zeros((A.shape[0], B.shape[1]))

for i in range(0, B.shape[1]):
    AB_prod[:,i] = np.matmul(A,B[:,i])

# row perspective
AB_prod = np.zeros((A.shape[0], B.shape[1]))

for i in range(0, A.shape[0]):
    AB_prod[i,:] = np.matmul(A[i,:],B)

# 45. order of operations
M1 = np.matmul(A,B).T 
M2 = np.matmul(B.T, A.T)
M1-M2

# 46. matrix-vector multiplication
m = 4
N = np.random.randint(-10,11,(m,m))
S = np.round(N.T*N / m**2) # scaled symmetric
w = np.array([1,4,3,-2])

S@w
S.T@w
w@S
w.T@S.T
w.T@S

w@N
N@w
w.T@N
N.T@w
(N@w).T
N@w.T

# 47. 2D transformation matrices
# transformation matrix, rotate and stretch
v = np.array([3,-2])
M = np.random.randint(-10,10, size=(2,2))

w = M@v
plt.plot([0,v[0]],[0,v[1]],label='v')
plt.plot([0,w[0]],[0,w[1]],label='Av')
plt.title('rotate + strech')

# pure strech
theta = np.pi/3
A = np.array([[math.cos(theta),-math.sin(theta)],
              [math.sin(theta),math.cos(theta)]])
w = A@v
plt.plot([0,v[0]],[0,v[1]],label='v')
plt.plot([0,w[0]],[0,w[1]],label='Av')
plt.title('pure rotate')
plt.legend()

# 48. test for strech rotate effect
v = np.array([3,-2])
for t in range(1,24):
    th = np.pi*((t+1)/12) # pi/12 to 2pi
    A = np.array([[2*math.cos(th),-math.sin(th)],
                  [math.sin(th),math.cos(th)]])
    w = A@v
    plt.plot([0,w[0]],[0,w[1]],label= 'w_{}'.format(th)) 

# 54. multiplication of 2 symmetric matrices
a,b,c,d,e,f,g,h,k,l,m,n,o,p,q,r,s,t,u = symbols('a b c d e f g h k l m n o p q r s t u', real=True)

# symmetric and constant-diagonal matrices
A = Matrix([ [a,b,c,d],
             [b,a,e,f],
             [c,e,a,h],
             [d,f,h,a]   ])

B = Matrix([ [l,m,n,o],
             [m,l,q,r],
             [n,q,l,t],
             [o,r,t,l]   ])
    
init_printing()

# but AB neq (AB)'
A@B - (A@B).T
    
# 55. self-multiply for full and diag matrices
A = np.random.randint(-20,20, size=(4,4))
D = np.diag([1,3,2,5])
D2 = np.diag([5,8,4,9])

A@A
A*A

D@D
D*D
D@D2

# 56. Fourier transform with matrix multiplication
n=30
F = np.zeros((n,n), dtype=complex)
omega = exp(-2*np.pi*1j/n)

for i in range(0,n):
    for j in range(0,n):
        m = i*j
        F[i,j] = omega**m

plt.imshow(F.real, cmap='jet')
plt.show()

x = np.random.randn(n)
X1 = F@x
X2 = np.fft.fft(x)
X1-X2

# 57. Frobenius dot product
m=9
n=4

A = np.random.randn(m,n)
B = np.random.randn(m,n)

# method1: addition of element-wise multiplication
np.sum(A*B)

# method2: vectorize then dot product
Av = np.reshape(A, m*n, order='F') # 'F' means column-wise, 'C' is row-wise and default
Bv = np.reshape(B, m*n, order='F')
np.dot(Av,Bv)

# method3: trace(A.T@B)
np.trace(A.T@B)

# matrix norm/Frobenius norm/Euclidean norm
np.linalg.norm(A, ord='fro') # default ord is 'fro'
np.sqrt(np.trace(A.T@A)) # same result
np.sqrt(np.sum(A*A)) # same result

# 58. matrix norm
A = np.array([ [11,10,3], [4,5,6], [7,8,9] ])

# 1. Euclidean norm/Frobenius norm
norm_frob = np.linalg.norm(A, ord='fro')

# 2. induced 2-norm
norm_ind2 = np.linalg.norm(A, ord=2)
np.sqrt(np.max(np.linalg.eig(A.T@A)[0])) # same as above

# schatten p-norm
p=2
s = np.linalg.svd(A)[1] # get singular values
norm_schat = np.sum(s**p)**(1/p)

print(norm_frob, norm_ind2, norm_schat)

# orthogonal matrix, 2-norm is always 1
Q,R = np.linalg.qr(A)
np.linalg.norm(Q, ord=2) # 2-norm is 1

# 59. conditions and proof of self-adjoint operators 
m=5
M = np.random.randn(m,m)
A = np.round(M.T@M,0)
v = np.round(np.random.randn(m),0)
w = np.round(np.random.randn(m),0)

np.dot(A@v,w)
np.dot(v,A@w)

# 60. matrix asymmetry index (how asymmetric a matrix is)




