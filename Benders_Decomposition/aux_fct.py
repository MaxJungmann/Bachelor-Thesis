""" This file implements several auxiliary functions for the Benders decomposition method from main.py. """

import numpy as np
import gurobipy as gp
from gurobipy import GRB 


def init_master(A,b,c):
    """ Initialize and solve the master problem. """
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)    # suppress any Gurobi console output
    env.start()
    master = gp.Model(env=env)
    x = master.addMVar(shape=np.shape(A)[1], vtype=GRB.CONTINUOUS)      # x >= 0 is set by default
    master.addConstr(A@x == b)
    master.setObjective(c@x, GRB.MINIMIZE)
    master.optimize()

    return master, x


def init_scenarios(N, W, q):
    """ Initialize the optimization model for each scenario. """
    models = []
    variables = []
    for scenario in range(N):
        env = gp.Env(empty=True)
        env.setParam("OutputFlag",0)    # suppress any Gurobi console output
        env.start()
        model = gp.Model(env=env)
        models.append(model)
        models[-1].Params.DualReductions = 0     # distinguish between infeasible and unbounded problems   
        models[-1].Params.InfUnbdInfo = 1        # obtain unbounded ray if problem is unbounded
        variables.append(models[-1].addMVar(shape=np.shape(W[scenario].T)[1], lb = -np.inf, vtype=GRB.CONTINUOUS))
        models[-1].addConstr(W[scenario].T@variables[-1] <= q[scenario])

    return models, variables


def stopping_criterion(iter, MAX_ITER, theta, models, p, N, TOL_OPT, reason=False):
    """ 
    If reason = False, test if one of the stopping criteria is met. 
    If reason = True, return the reason for termination. 
    """

    if iter > MAX_ITER:
        if reason == True:
            return "Maximum number of iterations reached"
        return True
    
    try:
        if np.abs((np.sum([p[model] * models[model].ObjVal for model in range(N)]) - theta.X) / theta.X) <= TOL_OPT:
            if reason == True:
                return "Relative objective tolerance reached"
            return True
        return False
    except:
        return False
