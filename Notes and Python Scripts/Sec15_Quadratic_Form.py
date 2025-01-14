# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:23:22 2025

@author: elisa
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# Sec15 topic: Quadratic form:
    # quadratic form: qf(w,S) = w.T@S@w, resulting in a single value, S is square matrix
        #  if S is identity matrix: qf(w,I) = w.T@w
    # normalized quadratic form: argmax {qf(w,S)/w.T@w}
    # for symmetric matrix:
        # eigenvectors points to the ridge/valley directions of qf surface

# 165. eigenvectors and qf surface for a symmetric matrix

    