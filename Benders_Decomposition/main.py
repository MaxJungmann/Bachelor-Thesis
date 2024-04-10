""" This file implements the Benders decomposition method to solve two-stage models with finite discrete distribution. """

import numpy as np
from gurobipy import GRB 
from aux_fct import init_master, init_scenarios, stopping_criterion

# Constants
MAX_ITER = 10000
TOL_OPT = 0.001


def benders_decomposition(A,b,c,T,W,h,q,p):
    """ Solve a block-structured linear program using Benders decomposition.

    The linear program is of the form
        minimize     c*x  +  p_1 * q_1*y_1  +  ...  +  p_N * q_N*y_N
        subject to   Ax                                         = b
                     T_1 x + W_1 y_1                            = h_1
                     .                .                            .
                     .                     .                       .
                     .                          .                  .
                     T_N x +                        + W_N y_N   = h_N                     
                     x >= 0
                     y_i >= 0, i=1,...,N.

    Such linear programs occur in two-stage models with finite discrete distribution in the context of stochastic linear programming.
    Therefore, N is considered as the number of scenarios. 

    In each iteration the method solves a master problem yielding a vector x. 
    Then, for this fixed x, the dual problems for the scenarios 1,...,N are solved independently of each other.
    Based on the dual solutions, in each iteration a cut is generated to remove infeasible or suboptimal points.

    Input:
        A: Technology matrix 
        b: Right-hand side vector 
        c: Cost vector 
        T: List where each entry corresponds to a matrix T_i, i=1,..,N of a scenario
        W: List where each entry corresponds to a matrix W_i, i=1,..,N of a scenario
        h: List where each entry corresponds to a right hand-side vector h_i, i=1,..,N of a scenario
        q: List where each entry corresponds to a cost vector q_i, i=1,..,N of a scenario
        p: Array of length N storing the probability of each scenario

    Output:
        A dictionary containing
            solution: Optimal solution (x,y_1,...,y_N)
            opt_val: Objective value of the optimal solution
            termination_reason: Reason for termination of the algorithm
            iter: Number of iterations performed during the algorithm
    
    An exception is raised if the problem turns out to be unsolvable.
    """
  
    # Initialize and solve master problem
    master, x = init_master(A,b,c)
    theta_set = False      

    if master.Status == GRB.OPTIMAL:
        x_master = x.X 
    else:
        raise Exception("Initial relaxation is not solvable.")

    # Initialize the dual problem for each scenario
    N = len(W)      # Number of scenarios
    models, variables = init_scenarios(N, W, q)

    # Main algorithm
    theta = None
    iter = 0
    while not stopping_criterion(iter, MAX_ITER, theta, models, p, N, TOL_OPT):
        # Solve the dual problems of the scenarios given the master solution 
        optimal_solutions = []
        for scenario in range(N):
            models[scenario].setObjective((h[scenario]-T[scenario]@x_master)@variables[scenario], GRB.MAXIMIZE)
            models[scenario].optimize()

            # If the dual is infeasible, the primal problem is infeasible or unbounded (and hence not solvable)
            if models[scenario].Status == GRB.INFEASIBLE:
                raise Exception("Problem is not solvable")

            # If the dual is unbounded (and hence the primal infeasible), add a feasibility cut
            if models[scenario].Status == GRB.UNBOUNDED:
                ray = variables[scenario].UnbdRay
                master.addConstr((T[scenario].T@ray)@x >= np.dot(ray,h[scenario]))
                break

            # If the dual problem is solvable, store the optimal solution of the scenario
            if models[scenario].Status == GRB.OPTIMAL:
                optimal_solutions.append(variables[scenario].X)   

        # If all scenarios have an optimal solution, add an optimality cut
        if len(optimal_solutions) == N:                                         
            # Introduce the auxiliary variable theta if it is not set yet
            if not theta_set:
                theta = master.addMVar(shape=1, vtype=GRB.CONTINUOUS, obj=1, lb=-np.inf)
                theta_set = True

            # Add optimality cut 
            master.addConstr(theta + np.sum([p[model] * T[model].T@optimal_solutions[model] for model in range(N)],axis=0)@x 
                                >= np.sum([p[model] * np.dot(h[model],optimal_solutions[model]) for model in range(N)]))
            
        # Reoptimize the master model 
        master.optimize()
        if master.Status == GRB.OPTIMAL:
            x_master = x.X
        else:
            raise Exception("Problem is unsolvable")

        iter += 1


    # Get the optimal primal solutions to the optimal dual solutions for the scenarios and return the result
    y = []
    for scenario in range(N):
        y.append(models[scenario].getAttr("Pi", models[scenario].getConstrs()))

    return {"solution": np.hstack((x_master, np.ravel(y))),"opt_val": master.ObjVal, 
            "termination_reason":stopping_criterion(iter, MAX_ITER, theta, models, p, N, TOL_OPT, reason=True),"iter": iter}


