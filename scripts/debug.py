import numpy as np
import networkx as nx
import gnngls
import itertools

np.set_printoptions(linewidth=200)

G = nx.DiGraph()
n_nodes = 4

coords = np.random.random((n_nodes, 2))
for n, p in enumerate(coords):
    G.add_node(n, pos=0.0)

distance = np.random.random((n_nodes, n_nodes))
for i, j in itertools.permutations(G.nodes, 2):
    G.add_edge(i, j, weight=distance[i, j])

# opt_solution = gnngls.sub_optimal_tour(G, scale=1e6, max_trials=100, runs=10)
opt_solution = gnngls.optimal_tour(G, scale=1e6)
sub_opt_sol = gnngls.sub_optimal_tour(G, scale=1e6)

opt_len = gnngls.tour_cost(G, opt_solution)
sub_opt_len = gnngls.tour_cost(G, sub_opt_sol)

print(f'Optimal length: {opt_len}')
print(f'Sub-optimal length: {sub_opt_len}')