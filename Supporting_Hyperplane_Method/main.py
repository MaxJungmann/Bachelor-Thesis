"""
This file implements the supporting hyperplane method of Veinott to solve (generalized) convex optimization problems.
Linear programs are solved with Gurobi.
"""

import numpy as np 
import gurobipy as gp
from gurobipy import GRB 
from aux_fct import eval_nonlin_constr, bisection, stopping_criterion


def supporting_hyperplane_method(A, b, c, nonlin_constr, x_int):
    """ Execute the supporting hyperplane method of Veinott. 

    Solve the following convex optimization problem:
        minimize    c*x
        subject to  Ax <= b
                    x >= 0
                    g_i(x) >= 0 , i = 1,...,l

    The method starts with only the linear constraints and then sequentially adds linear inequalities (a.k.a. cutting planes).

    Input:
        A: Technology matrix 
        b: Right hand side vector 
        c: Objective vector 
        nonlin_fct: Dictionary for function and gradient evaluation of the nonlinear constraints
                    which are assumed to be pseudoconcave differentiable functions
        x_int: A feasible point satisfying g_i(x_int) > 0 for i = 1,...,l 

    Output:
        Dictionary containing the following entries:
            x_opt: (Approximately) optimal solution 
            A: Technology matrix of the final relaxation 
            b: Right-hand side of the final relaxation 
            gap: Relative optimality gap between best boundary point and relaxed vertex solution
            iter: Number of iterations
            termination_reason: Reason for termination of the method

    Exceptions:
        1. x_int does not satisfy the requirements
        2. The initial relaxation turns out to be unbounded
    """

    # Check whether x_int is strictly feasible (and hence ensure feasibility of the original problem)
    if not np.all(A@x_int <= b) or not np.all(x_int >= 0) or not eval_nonlin_constr(nonlin_constr, x_int, "strictly_feasible"):
        raise Exception("x_int is not strictly feasible")
        
    # Build and optimize the relaxed model
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)    # suppress any Gurobi console output
    env.start()
    model = gp.Model(env=env)
    model.Params.DualReductions = 0
    x = model.addMVar(shape = len(c))      
    model.addConstr(A@x <= b)       # x >= 0 is set by default
    model.setObjective(c@x, GRB.MINIMIZE)
    model.optimize()

    # Ensure solvability of the initial relaxation (and hence boundedness of the original problem)
    if model.Status == 5:
        raise Exception("Initial polyhedral relaxation is unbounded. Please ensure a bounded initial relaxation.")

    # Get relaxed solution and corresponding boundary point
    x_out = model.getAttr("X", model.getVars())
    x_bd = bisection(x_int, x_out, nonlin_constr)

    # x_best is the feasible solution with the smallest objective value found so far
    x_best = x_bd

    iter = 0
    while not stopping_criterion(iter, x_best, x_out, c):
        # Add constraint and solve refined relaxation
        gradient = eval_nonlin_constr(nonlin_constr, x_bd, "gradient")
        model.addConstr(gradient@x >= gradient@x_bd)
        model.optimize()
        
        # Get relaxed solution and corresponding boundary point
        x_out = model.getAttr("X", model.getVars())
        x_bd = bisection(x_int, x_out, nonlin_constr)

        # Update best solution if possible
        if c@x_bd < c@x_best:
            x_best = x_bd
    
        iter += 1

    # Return the result
    gap = c@x_best - c@x_out
    ground_truth = c@x_out
    return {"x_opt": x_best, "A": model.getA().toarray(), "b": model.getAttr("RHS", model.getConstrs()), "gap": (np.abs(gap/ground_truth)), 
            "iter": iter, "termination_reason": stopping_criterion(iter, x_best, x_out, c, reason=True)}



