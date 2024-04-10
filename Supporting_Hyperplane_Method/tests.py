"""
This file tests the implementation of the supporting hyperplane method from main.py.
Therefore, the optimal value is compared with the result of the benchmark solver GUROBI.
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from main import supporting_hyperplane_method

# Relative tolerance up to which an instance is considered to be solved correctly
TOL = 0.01


def test_supporting_hyperplane_method(num):
    """
    For a negative semidefinite matrix D the convex optimization problem
        minimize    c*x
        subject to  Ax <= b
                    x >= 0
                    x@D@x + e*x + f >= 0
    is solved num times with randomized problem data.

    Input:
        num: The number of test runs to be performed

    Output:
        Textual message how many of the tests returned a correct solution (within some tolerance)
    """

    counter = 0     # stores the number of correct tests
    for _ in range(num):  
        # Build random linear constraints 
        n = np.random.randint(10,100)
        m = np.random.randint(5,n)
        A = np.random.randint(1,10,(m,n))
        b = np.random.randint(5*n,50*n,m)
        c = np.random.randint(-5,5,n)

        # Build random concave constraint 
        D_ = np.random.randint(-5,5,(np.random.randint(5,20),n))
        D = -np.dot(D_.transpose(),D_)      # -D_^T@D_ is always a negative semidefinite matrix
        e = np.random.randint(-5,5,n)
        f = np.random.randint(1,5)
        def fct_eval(x):
            return x@D@x + e@x + f 
        def gradient_eval(x):
            return 2*D@x + e
        nonlin_constr = {"concave_constraint": [fct_eval, gradient_eval]} 

        # f > 0 and b >= 0 imply that 0 is feasible for the problem and strictly feasible for the nonlinear constraint
        x_int = np.zeros(n)

        # Solve the problem with the supporting hyperplane method
        x_opt_veinott = supporting_hyperplane_method(A,b,c,nonlin_constr, x_int)["x_opt"]

        # Solve the problem with Gurobi
        env = gp.Env(empty=True)
        env.setParam("OutputFlag",0)    # suppress any Gurobi console output
        env.start()
        model = gp.Model(env=env)
        x = model.addMVar(shape = n)      
        model.setObjective(c@x, GRB.MINIMIZE)
        model.addConstr(A@x <= b)        
        model.update()
        x = model.getVars()
        model.addQConstr(x@D@x + e@x + f >= 0)
        model.optimize()
        x_opt_gurobi = model.getAttr("X", model.getVars())

        # Check if the optimal values coincide
        gap = c@x_opt_veinott - c@x_opt_gurobi
        ground_truth = c@x_opt_gurobi
        if np.abs(gap/ground_truth) <= TOL:
            counter += 1
    
    return f"{counter} out of {num} test instances were solved correctly."


# Test the algorithm
print(test_supporting_hyperplane_method(100))
