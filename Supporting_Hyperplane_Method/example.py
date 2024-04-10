""" 
This file illustrates the use of the supporting hyperplane method from main.py. 
Therefore, the following two-dimensional sample problem is solved:

minimize      -2x - y
subject to    x <= 6
              y <= 4
              (x-3)^2  +  (y-2)^2  <= 2
              0.2*(x-3)^2  +  (y-2)^2  <= 1 

Moreover, the result is also visualized. 
Note that in order for the plot to work the nonlinear constraints must not be changed. However, the linear constraints and the 
objective function can be modified to visually explore the effect of a different cost vector or different linear constraints.
"""

import numpy as np
from main import supporting_hyperplane_method
from plot import plot_supporting_hyperplane_method


# Linear constraints are given as simple box constraints
A = np.array([[1,0],[0,1]])
b = np.array([6,4])

# Objective
c = np.array([-2,-1])

# Methods to evaluate nonlinear functions and their gradients for the nonlinear constraints
def fct_circle(x):
    return -(x[0]-3)**2 - (x[1]-2)**2 + 2
def gradient_circle(x):
    return np.array([-2*(x[0]-3), - 2*(x[1]-2)])

def fct_ellipse(x):
    return -0.2*(x[0]-3)**2 - (x[1]-2)**2 + 1 
def gradient_ellipse(x):
    return np.array([-0.4*(x[0]-3), -2*(x[1]-2)])

nonlin_constr = {"circle": [fct_circle, gradient_circle], "ellipse": [fct_ellipse, gradient_ellipse]} 

# Strictly feasible point
x_int = np.array([3,2])

# Solve the problem
result = supporting_hyperplane_method(A,b,c,nonlin_constr,x_int)

# Print the results
print("Optimal solution: ", result["x_opt"])
print("Relative optimality gap: ", result["gap"])
print("Number of performed iterations: ", result["iter"])
print("Reason for termination: ", result["termination_reason"])

# Plot the results
plot_supporting_hyperplane_method(result["x_opt"],  x_int, result["A"], result["b"], np.shape(A)[0])


