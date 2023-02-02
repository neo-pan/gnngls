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

# checkpoint = torch.load('/home/xhpan/Codes/gnngls/scripts/models/tsp50_c/checkpoint_final.pt')
# print(checkpoint['epoch'])

G = nx.Graph()
n_nodes = 10
coords = np.random.random((n_nodes, 2))
for n, p in enumerate(coords):
    G.add_node(n, pos=0.0)

distance = np.random.random((n_nodes, n_nodes))
for i, j in itertools.combinations(G.nodes, 2):
    if j-i == 1 or j-i == n_nodes-1:
        w = 1
    else:
        w = 10
    G.add_edge(i, j, weight=w)

print("distance", nx.attr_matrix(G, edge_attr='weight'))
print(check_triangle_inequality(G))
print(f"Sub-optimal tour cost: {tour_cost(G, sub_optimal_tour(G))}")
print(f"Optimal tour cost: {tour_cost(G, optimal_tour(G))}")