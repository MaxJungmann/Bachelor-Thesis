""" This file tests the implementation of the Benders decomposition from main.py by comparing its optimal value with the result of 
the general-purpose solver Gurobi. """

import numpy as np 
import gurobipy as gp
from gurobipy import GRB 
from main import benders_decomposition

# Relative tolerance up to which an instance is considered to be solved correctly
TOL = 0.01


def test_bender(n,m,s,k,N,num):
    """ Build num many randomized two-stage problems. Then solve them once with Benders decomposition and once with Gurobi 
        and compare the optimal values.
    
    All scenarios i = 1,...,N have equal size. Here is
        A: m x n matrix     + m x m slack matrix 
        b: Vector of size m
        c: Vector of size n+m
        T_i: s x n matrix   + s x m zero matrix
        W_i: s x k matrix   + s x s slack matrix
        h_i: Vector of size s
        q_i: Vector of size k+s
        p: Vector of size N storing the probability of each scenario.

    Output:
        Textual message how many problems were solved correctly (within some tolerance).
    """
    
    # Number of correctly solved instances
    counter = 0
    
    for _ in range(num):
        # Create first stage problem data
        A = np.hstack((np.random.randint(1,20,(m,n)), np.eye(m,m)))
        b = np.random.randint(n*10,n*100,m)
        c = np.hstack((np.random.randint(-10,-1,n), np.zeros(m)))

        # Create second stage problem data 
        T = [np.hstack((np.random.randint(1,20,(s,n)),np.zeros((s,m)))) for _ in range(N)]
        W = [np.hstack((np.random.randint(1,20,(s,k)),np.eye(s,s))) for _ in range(N)]
        h = [np.random.randint(n*10+k*10,n*100+k*100,s) for _ in range(N)]
        q = [np.hstack((np.random.randint(-10,-1,k),np.zeros(s))) for _ in range(N)]
        p = np.random.randint(0,100,N)
        p = p/np.sum(p)

        # Apply Benders decomposition 
        opt_bender = benders_decomposition(A,b,c,T,W,h,q,p)["opt_val"]

        # Build standard LP data (A,b,c) for Gurobi
        q = np.array(q,dtype=np.float64).flatten()
        offset = 0
        for i in range(N):
            q[offset:offset+k+s] *= p[i]
            offset += k+s
        c = np.hstack((c,q))

        b = np.hstack((b,np.array(h).flatten()))

        A_ = np.zeros((m + N * s, n+m+N*(k+s)))
        A_[0:m,0:n+m] = A
        for i in range(N):
            A_[m+i*s:m+(i+1)*s, :n+m] = T[i]
            A_[m+i*s:m+(i+1)*s, n+m+i*(k+s):n+m+(i+1)*(k+s)] = W[i]
        A = A_

        # Solve the problem with Gurobi 
        env = gp.Env(empty=True)
        env.setParam("OutputFlag",0)    # suppress any Gurobi console output
        env.start()
        model = gp.Model(env=env)
        x = model.addMVar(shape=np.shape(A)[1], vtype=GRB.CONTINUOUS)
        model.addConstr(A@x == b)
        model.setObjective(c@x, GRB.MINIMIZE)
        model.optimize()
        opt_gurobi = model.ObjVal

        # Check for correctness
        if np.abs((opt_bender - opt_gurobi) / opt_gurobi) <= TOL:
            counter += 1


    return f"{counter} out of {num} test instances were solved correctly."


# Test Benders decomposition
print(test_bender(n=100,m=50,s=10,k=20,N=10,num=100))