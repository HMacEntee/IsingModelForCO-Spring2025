# imports
import networkx as nx
import numpy as np
from typing import List, Union
from util import *
# PuLP is a linear & mixed integer programming modeler used to constuct 
# optimization problems and call solvers (CPLEX, GUROBI, etc...)
# https://coin-or.github.io/pulp/main/includeme.html
from pulp import *

# IMPORTANT 
# you must locally install the GUROBI and CPLEX solvers
# pip install gurobipy
# install cplex via the CPLEX Optimiztion Studio

# problem forumlation dervied from the below 
# https://www.tcs.tifr.res.in/~prahladh/teaching/2009-10/limits/lectures/lec03.pdf
def bip(graph, solver):

    # Create dictionary of weights dependent on weight/unweighted graph
    edges = graph.edges(data=True)
    weights = {}
    weighted = nx.is_weighted(graph)
    if weighted:
        pass
        weights ={(u, v) : int(w["weight"]) for (u, v, w) in edges}
    else:
        weights = {(u, v) : 1 for (u, v, _) in edges}

    #define problem
    maxcut = LpProblem("maxcut", LpMaximize)

    # construct problem variables
    x = {node : pulp.LpVariable(name=f"x_{node}", cat="Binary") for node in graph.nodes}
    e = {(u, v): pulp.LpVariable(name=f"e_{u},{v}", cat="Binary") for (u, v, _) in edges}

    # define objective function
    maxcut += pulp.lpSum([weights[(u, v)] * e[(u, v)] for (u, v) in e]), "Sum_of_cut_edges"

    # define maxcut problem constraints
    for (u, v) in graph.edges:
        maxcut += (
            e[(u, v)] <= x[u] + x[v],
            f"cut_edge_constraint_three{u}_{v}"
        )
        maxcut += (
            e[(u, v)] <= 2 - (x[u] + x[v]),
            f"Ensure_cut_edge_principles_{u}_{v}"
        )
        
    # solve the problem using provided solver, adjuct time limit to graph size
    maxcut.solve(solver(msg=True, timeLimit=60))

    # display results
    print(f"{solver} Status:{maxcut.status}")
    cutedges = [ pulp.value(e[(u, v)]) for (u, v) in e ]
    print(f"Edges cut: {sum(cutedges)}")

if __name__ == "__main__":

    # Extract graph data from data files
    graph = read_nxgraph('.././data/syn/powerlaw_200_ID0.txt')

    # List available solvers usable locally
    print(listSolvers(onlyAvailable=True))

    # solve for undirected graph
    bip(graph, GUROBI)    
    # bip(graph, CPLEX_CMD)    
        




