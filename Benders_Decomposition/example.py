""" This file provides a sample problem for using Benders decomposition algorithm from main.py. """

import numpy as np 
from main import benders_decomposition

# First stage data
A = np.array([[-1,1,1,0],[1,2,0,1]])
b = np.array([130,2000])
c = np.array([2,1,1,1])

# Second stage data
T_1 = np.array([1,0,0,0]).reshape(1,4)
T_2 = np.array([0,1,0,0]).reshape(1,4)
T = []
T.append(T_1)
T.append(T_2)

W_1 = np.array([1,1]).reshape(1,2)
W_2 = np.array([1,1]).reshape(1,2)
W = []
W.append(W_1)
W.append(W_2)

h_1 = np.array([100])
h_2 = np.array([100])
h = []
h.append(h_1)
h.append(h_2)

q_1 = np.array([1,1])
q_2 = np.array([1,1])
q = []
q.append(q_1)
q.append(q_2)

p = np.array([0.5,0.5])


# Apply the algorithm to the problem
print(benders_decomposition(A,b,c,T,W,h,q,p))
