# What is in this repo
This repository contains the code for two optimization algorithms I implemented in my [bachelor thesis](/Bachelor_Thesis.pdf). On the one hand, I implemented a **cutting plane method** for solving convex optimization problems. On the other hand, I implemented a **decomposition algorithm** for solving large-scale block structured linear programs (LPs).

My [bachelor thesis](/Bachelor_Thesis.pdf) was about linear programming problems with the additional difficulty that some of the input data is not deterministic but uncertain and follows a known probability distribution. Such problems are typically referred to as **stochastic linear programs**. In practice, for instance customer demands are often times uncertain and have to be modeled appropriately. While it is quite straightforward to solve traditional linear programs, there is no comprehensive theory for all stochastic linear programs. Therefore, in my [bachelor thesis](/Bachelor_Thesis.pdf) I focused on special cases of stochastic linear programs which turn out to be equivalent to (deterministic) convex optimization problems that can be solved efficiently. 

## Supporting hyperplane method
The supporting hyperplane method of Veinott can be used to solve a special case of a **chance-constrained linear program**, namely the case when only the right-hand side is random and follows a nondegenerate multivariate normal distribution. In Section 2.3 of my [bachelor thesis](/Bachelor_Thesis.pdf) I show how this problem can be transformed to a (generalized) convex optimization problem. Moreover, an algorithm to solve the arising problem is derived and its convergence properties are proven.

Fortunately, the main idea of the algorithm can be summarized in a single image which can be found below.

The feasible set of the problem is a convex set (e.g. a convex ellipse). Moreover, an outer **polyhedral approximation** of this set (e.g. given by box constraints) as well as a strictly feasible point (all blue) are known. In each iteration a linear program over the outer polyhedral approximation is solved yielding an optimal vertex solution (yellow). If this vertex is not feasible for the original problem, a cutting plane (black) is computed that cuts off this vertex but nothing of the actual feasible set. The point on the boundary (black) is the new guess of an approximate solution. 

<p align="center">
<img src="/Plots_README/plot_supporting_hyperplane_method.png" alt="" width="400"/>
</p>

After several iterations one may get the following image.

<p align="center">
<img src="/Plots_README/plot_termination_of_supporting_hyperplane_method.png" alt="" width="400"/>
</p>

## Benders decomposition
Another common problem type in stochastic linear programming are so called **two-stage models**. Here a recourse action is allowed in order to compensate a possibly disadvantageous decision due to the random problem data.

These problems are solved using a discretization of the probability distribution. However, this discretization leads to a huge linear program which might not be that amenable for a general-purpose LP solver. Fortunately, the arising LP has a special structure which can be exploited to solve the LP more efficiently. A detailed outline of this method which is called Benders decomposition can be found in section 3.2 of my [bachelor thesis](/Bachelor_Thesis.pdf).


# Project organisation
## Supporting hyperplane method
1. The actual algorithm is implemented in [main.py](/Supporting_Hyperplane_Method/main.py) and uses several auxiliary functions from [aux_fct.py](/Supporting_Hyperplane_Method/aux_fct.py).
2. The usage of the algorithm is demonstrated in [example.py](/Supporting_Hyperplane_Method/example.py). The result for the sample problem is visualized using [plot.py](/Supporting_Hyperplane_Method/plot.py) and is saved to [plot.png](/Supporting_Hyperplane_Method/plot.png).
3. The implementation is tested in [tests.py](/Supporting_Hyperplane_Method/tests.py).

## Benders decomposition
1. The actual algorithm is implemented in [main.py](/Benders_Decomposition/main.py) and uses several auxiliary functions from [aux_fct.py](/Benders_Decomposition/aux_fct.py).
2. The usage of the algorithm is demonstrated in [example.py](/Benders_Decomposition/example.py). 
3. The implementation is tested in [tests.py](/Benders_Decomposition/tests.py). 
