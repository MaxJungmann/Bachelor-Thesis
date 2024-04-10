""" 
This file implements three auxiliary functions for the supporting hyperplane method implemented in main.py. 

    1. eval_nonlin_constr to evaluate the nonlinear constraints
    2. bisection to find a boundary point which is required in each iteration of the supporting hyperplane method
    3. stopping_criterion to implement several stopping criteria for the supporting hyperplane method
"""

import numpy as np

# Constants
MAX_ITER = 10000
TOL_OBJ_REL = 0.001
TOL_BISEC = 0.001


def eval_nonlin_constr(nonlin_constr, x, type):
    """ 
    Provide access to function and gradient evaluations of the nonlinear constraints.

    Input:
        nonlin_constr: Dictionary with one entry for each nonlinear constraint
                       Each entry has the form {"constraint": [function, gradient]}. It provides
                       a method to evaluate the value of the constraint at a given point and
                       a method to evaluate the gradient of the constraint at a given point.
        x : Point (numpy array) at which constraints or one particular gradient are evaluated 
        type: One of "feasible", "strictly_feasible", or "gradient"
    
    Output:
        Boolean expression for "feasible" or "strictly_feasible"
        Numpy array as a gradient evaluation at x for "gradient"
    """

    # Evaluate all constraints at x
    constr_fct_values = np.array([nonlin_constr[constr][0](x) for constr in nonlin_constr.keys()])

    if type == "feasible":
        return np.all(constr_fct_values >= 0)
    
    if type == "strictly_feasible":
        return np.all(constr_fct_values > 0)
    
    if type == "gradient":
        # Find the constraint with the smallest function value (which will be 0) and return its gradient at x
        return nonlin_constr[list(nonlin_constr.keys())[np.argmin(constr_fct_values)]][1](x)    



def bisection(x_int, x_out, nonlin_constr):
    """ 
    Execute a bisection search and return a boundary point.
    
    Compute the maximum value for lamda s.t. x_int + lamda * (x_out - x_int) is feasible for the original problem.
    Therefore, the interval [0,1] is successively halved and the constraints are evaluated at the center of the interval.
    If the center is feasible, then the right half of the interval is chosen in the next step, else the left half.

    Input:
        x_int: Strictly feasible point for the original problem 
        x_out: Relaxed solution that is infeasible for the original problem 
        nonlin_constr: Dictionary for function and gradient evaluation of the nonlinear constraints

    Output:
        A boundary point 
    """

    lower = 0
    upper = 1
    delta = x_out - x_int
    max_steps = -np.log2(TOL_BISEC / np.linalg.norm(delta))   
    
    for _ in range(int(max_steps)):
        lamda = (upper + lower) / 2
        if eval_nonlin_constr(nonlin_constr, x_int + lamda * delta, "feasible"):
            lower = lamda
        else:
            upper = lamda
    
    return x_int + lower * delta



def stopping_criterion(iter, x_bd, x_out, c, reason=False):
    """ 
    If reason = False, test if one of the stopping criteria is met. 
    If reason = True, return the reason for termination. 
    """

    gap = c@x_bd - c@x_out
    ground_truth = c@x_out
    if np.abs(gap/ground_truth) <= TOL_OBJ_REL:
        if reason == True:
            return "Relative objective tolerance reached"
        return True

    if iter > MAX_ITER:
        if reason == True:
            return "Maximum number of iterations reached"
        return True
    
    return False


