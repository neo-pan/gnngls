import argparse
import itertools
import multiprocessing as mp
import pathlib
import uuid

import networkx as nx
import numpy as np
import torch

import gnngls
from gnngls import datasets
from gnngls import optimal_tour, sub_optimal_tour, tour_cost

from check_data import check_triangle_inequality
from generate_instances import get_solved_instances

# checkpoint = torch.load('/home/xhpan/Codes/gnngls/scripts/models/tsp50_c/checkpoint_final.pt')
# print(checkpoint['epoch'])

G = nx.DiGraph()
n_nodes = 5
coords = np.random.random((n_nodes, 2))
for n, p in enumerate(coords):
    G.add_node(n, pos=0.0)

w = 1
for i, j in itertools.permutations(G.nodes, 2):
    print(i,j, end=' ')
    if i==j:
        continue
    G.add_edge(i, j, weight=w)
    w += 1

list(get_solved_instances(20, 100))

# print("distance", nx.attr_matrix(G, edge_attr='weight'))
# print(check_triangle_inequality(G))
# print(f"Sub-optimal tour cost: {tour_cost(G, sub_optimal_tour(G))}")
# print(f"Optimal tour cost: {tour_cost(G, optimal_tour(G))}")