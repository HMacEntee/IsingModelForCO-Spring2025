import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import os
import numpy as np
from typing import List, Union, Tuple
import networkx as nx
import torch as th
from torch import Tensor

TEN = th.Tensor
INT = th.IntTensor
TEN = th.Tensor
GraphList = List[Tuple[int, int, int]]
IndexList = List[List[int]]


# read graph file, e.g., gset_14.txt, as networkx.Graph
# The nodes in file start from 1, but the nodes start from 0 in our codes.
def read_nxgraph(filename: str) -> nx.Graph(): # type: ignore
    graph = nx.Graph()
    with open(filename, 'r') as file:
        # lines = []
        line = file.readline()
        is_first_line = True
        while line is not None and line != '':
            if '//' not in line:
                if is_first_line:
                    strings = line.split(" ")
                    num_nodes = int(strings[0])
                    num_edges = int(strings[1])
                    nodes = list(range(num_nodes))
                    graph.add_nodes_from(nodes)
                    is_first_line = False
                else:
                    node1, node2, weight = line.split(" ")
                    graph.add_edge(int(node1) - 1, int(node2) - 1, weight=weight)
            line = file.readline()
    return graph

#
def transfer_nxgraph_to_adjacencymatrix(graph: nx.Graph):
    return nx.to_numpy_array(graph)

# the returned weightmatrix has the following format： node1 node2 weight
# For example: 1 2 3 // the weight of node1 and node2 is 3
def transfer_nxgraph_to_weightmatrix(graph: nx.Graph):
    # edges = nx.edges(graph)
    res = np.array([])
    edges = graph.edges()
    for u, v in edges:
        u = int(u)
        v = int(v)
        # weight = graph[u][v]["weight"]
        weight = float(graph.get_edge_data(u, v)["weight"])
        vec = np.array([u, v, weight])
        if len(res) == 0:
            res = vec
        else:
            res = np.vstack((res, vec))
    return res

# weightmatrix: format of each vector: node1 node2 weight
# num_nodes: num of nodes
def transfer_weightmatrix_to_nxgraph(weightmatrix: List[List[int]], num_nodes: int) -> nx.Graph():# type: ignore
    graph = nx.Graph()
    nodes = list(range(num_nodes))
    graph.add_nodes_from(nodes)
    for i, j, weight in weightmatrix:
        graph.add_edge(i, j, weight=weight)
    return graph

# max total cuts
def obj_maxcut(result: Union[Tensor, List[int], np.array], graph: nx.Graph):
    num_nodes = len(result)
    obj = 0
    adj_matrix = transfer_nxgraph_to_adjacencymatrix(graph)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if result[i] != result[j]:
                obj += adj_matrix[(i, j)]
    return obj