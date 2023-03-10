import numpy as np
import networkx as nx
import gnngls
import itertools
import tsplib95
import lkh

from gnngls import datasets

np.set_printoptions(linewidth=400)

# G = nx.read_gpickle('problem.pkl')
# n_nodes = G.number_of_nodes()

G = nx.DiGraph()
n_nodes = 10
coords = np.random.random((n_nodes, 2))
for n, p in enumerate(coords):
    G.add_node(n, pos=0.0)
distance = np.random.random((n_nodes, n_nodes))
for i, j in itertools.permutations(G.nodes, 2):
    G.add_edge(i, j, weight=distance[i, j])

sub_opt_sol = gnngls.sub_optimal_tour(G, scale=1e6)
sub_opt_len = gnngls.tour_cost(G, sub_opt_sol)
print(f'Sub-optimal length: {sub_opt_len}')

opt_solution = gnngls.optimal_tour(G, scale=1e6, verbose=True)
opt_len = gnngls.tour_cost(G, opt_solution)
print(f'Sub-optimal length: {sub_opt_len}')
print(f'Optimal length: {opt_len}')
if sub_opt_len < opt_len:
    print("SUB OPT better than OPT!")
    # save graph G
    # nx.write_gpickle(G, f'./{sub_opt_len}_{opt_len}.pkl')

# tsp_file = "tmp.tsp"
# problem = tsplib95.load(tsp_file)
# problem = lkh.LKHProblem.parse(problem.render().strip("EOF"))
# lkh_path='/home/xhpan/Tools/LKH3/LKH-3.0.8/LKH'
# solution = lkh.solve(lkh_path, problem=problem)
# print(solution)

in_solution = gnngls.tour_to_edge_attribute(G, opt_solution)
nx.set_edge_attributes(G, in_solution, 'in_solution')
datasets.set_features(G)
datasets.set_labels(G)
print(nx.attr_matrix(G, edge_attr='regret')[0])

# for e in [[3,1], [1,3]]:
#     tour = gnngls.fixed_edge_tour(G, e, scale=1e6, max_trials=100, runs=10)
#     cost = gnngls.tour_cost(G, tour)
#     regret = (cost - opt_len) / opt_len
#     print(f'Cost: {cost}, Regret: {regret}')
#     print(f"Fixed edge: {e}, Tour: {tour}")