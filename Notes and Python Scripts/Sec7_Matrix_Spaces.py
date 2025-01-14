# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 20:07:22 2024

@author: elisa
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

# Sec7 topic: Matrix Spaces:
    # C(A): the subspace spanned by all A's column vectors
    # R(A) or C(A.T): the subspace spanned by all A's row vectors
    # Null space:
        # if a non-trivial vector exists for a matrix such that Av = 0 and v != 0
        # N(A) contains all these vectors {v}
        # some matrices have empty null space
    # left Null space:
        # N{A}: the set of vectors {v} such that
        # v.T@A = 0 and v != 0
        # also written as: A.T@v = 0
    # interpretation of null space: 
        # vector sent in with no return
    # column/left-null and row/null spaces are orthogonal:
        # v is orthogonal to C(A) means: 
            # np.dot(v.T, any column of A) = 0
            # and, np.dot(v.T, any linear combination of A columns) = 0
    # dimensions of column/row/null-spaces:
        # 

# 73. visualize column spaces
# matrix S
S = np.array( [ [3,0],
                [5,2],
                [1,2] ] )

# vector v
v = np.array([-3, 1, 5])
# v = np.array([1, 7, 3])


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# draw plane corresponding to the column space
xx, yy = np.meshgrid(np.linspace(-5,5,10),np.linspace(-5,5,10))
cp = np.cross(S[:,0],S[:,1])
z1 = (-cp[0]*xx - cp[1]*yy)/cp[2]
ax.plot_surface(xx,yy,z1,alpha=.2)


## plot the two vectors from matrix S
ax.plot([0, S[0,0]],[0, S[1,0]],[0, S[2,0]],'k')
ax.plot([0, S[0,1]],[0, S[1,1]],[0, S[2,1]],'k')

# and the vector v
ax.plot([0, v[0]],[0, v[1]],[0, v[2]],'r')


ax.view_init(elev=150,azim=0)
plt.show()
