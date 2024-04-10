""" 
This file plots the result of the supporting hyperplane method for a two-dimensional sample problem.
The linear constraints are plotted in a general manner whereas the visualization of the concave constraints relies on the 
concrete functions given in example.py.
"""

import numpy as np 
import matplotlib.pyplot as plt 


def plot_supporting_hyperplane_method(x_opt, x_int, constr_matrix, constr_rhs, num_orig_constr):
    # For each linear constraint compute two points to plot the half space
    plot_points = np.zeros((np.shape(constr_matrix)[0],2,2))        

    for constr in range(np.shape(constr_matrix)[0]):        # Fix one component at 0 and 8, respectively, and compute the second one
        if constr_matrix[constr,1] != 0:
            plot_points[constr,0,:] = np.array([0, constr_rhs[constr] / constr_matrix[constr,1]])
            plot_points[constr,1,:] = np.array([8, (constr_rhs[constr] - constr_matrix[constr,0] * 8) / constr_matrix[constr,1]])
        else:
            plot_points[constr,0,:] = np.array([constr_rhs[constr] / constr_matrix[constr,0], 0])
            plot_points[constr,1,:] = np.array([(constr_rhs[constr] - constr_matrix[constr,1] * 8) / constr_matrix[constr,0], 8])

    # Plot the constraints
    # Distinguish between linear constraints of the original problem and cutting planes added during the algorithm
    plt.rcParams["figure.figsize"] = (10,10)
    for i in range(np.shape(constr_matrix)[0]):
        if i <= num_orig_constr - 1:
            plt.plot(plot_points[i,:,0], plot_points[i,:,1], 'b', linewidth = 5)
        else:
            plt.plot(plot_points[i,:,0], plot_points[i,:,1], 'k', linewidth = 1)

    # Plot the feasible set of the nonlinear concave constraints
    x_circle = np.linspace(1.585787, 4.41421355, 100)
    y_circle_1 = np.array([2 + np.sqrt(2 - (x-3)**2) for x in x_circle])
    y_circle_2 = np.array([2 - np.sqrt(2 - (x-3)**2) for x in x_circle])
    plt.plot(x_circle, y_circle_1, 'b', linewidth = 5)
    plt.plot(x_circle, y_circle_2, 'b', linewidth = 5)

    x_ellipse = np.linspace(0.7639320226, 5.236067976, 100)
    y_ellipse_1 = np.array([2 + np.sqrt(1 - 0.2*(x-3)**2) for x in x_ellipse])
    y_ellipse_2 = np.array([2 - np.sqrt(1 - 0.2*(x-3)**2) for x in x_ellipse])
    plt.plot(x_ellipse, y_ellipse_1, 'b', linewidth = 5)
    plt.plot(x_ellipse, y_ellipse_2, 'b', linewidth = 5)

    # Plot the strictly feasible point and the returned solution
    plt.plot(x_int[0], x_int[1],'bo', markersize=10)
    plt.plot(x_opt[0], x_opt[1],'ko', markersize=10)

    plt.xlim(0,7)
    plt.ylim(0,6)
    
    plt.savefig("plot.png")
    plt.show()