#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:48:31 2019

@author: mauri
"""

import numpy
from scipy.optimize import root

v = 18.33
d = 2/12
ro = 1.94
mu = 2.09e-5

Re = v*d*ro/mu
#Re = 63500
eps = 0.003

def f(x):
    return (-2*numpy.log10((2.51/(Re*numpy.sqrt(x))) + (eps/(3.71))) - 1.0/numpy.sqrt(x))



print(root(f, 0.02))
    


