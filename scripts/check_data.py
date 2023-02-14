import pathlib
import argparse
import random
from typing import List, Tuple
import numpy as np
import networkx as nx

from tqdm import tqdm
from pprint import pprint
from gnngls import optimal_tour, sub_optimal_tour, tour_cost


def check_knn_in_optimal_regressive(G: nx.Graph)->np.ndarray:
    cost_matrix = nx.attr_matrix(G, edge_attr='weight')[0]
    np.fill_diagonal(cost_matrix, np.inf)
    # k-nn relationship of edges in the optimal tour, represented by node sequence
    edge_max_k = np.zeros((G.number_of_nodes(), G.number_of_nodes()), np.int64)
    optimal_tour = []
    u = 0
    while len(optimal_tour) < G.number_of_nodes():
        optimal_tour.append(u)
        for v in G.neighbors(u):
            if v in optimal_tour:
                continue
            if G.get_edge_data(u,v)["in_solution"] == True:
                u = v
                break
    optimal_tour = np.array(optimal_tour)
    node_to_idx = {optimal_tour[i]:i for i in range(G.number_of_nodes())}

    for i in range(G.number_of_nodes()):
        tour = np.roll(optimal_tour, i)
        mat = cost_matrix.copy()
        u = tour[0]
        for v in tour[1:]:
            weight = mat[u,v]
            max_k_u = np.sum(cost_matrix[u]<=weight)
            edge_max_k[i, node_to_idx[u]] = max_k_u
            mat[:, u] = np.inf
            u = v
        edge_max_k[node_to_idx[u], i] = 1

    return edge_max_k


def check_knn_in_optimal(G: nx.Graph)->np.ndarray:
    cost_matrix = nx.attr_matrix(G, edge_attr='weight')[0]
    np.fill_diagonal(cost_matrix, np.inf)
    edge_max_k = np.zeros((G.number_of_nodes(), 2), np.int64)
    # k-nn relationship of edges in the optimal tour
    optimal_tour = []
    for (u, v, d) in G.edges.data("in_solution"):
        if d == True:
            optimal_tour.append((u,v))
    
    for i, (u,v) in enumerate(optimal_tour):
        weight = G.get_edge_data(u,v)["weight"]
        max_k_u = np.sum(cost_matrix[u]<=weight)
        max_k_v = np.sum(cost_matrix[v]<=weight)
        edge_max_k[i] = (max_k_u, max_k_v)

    return edge_max_k


def check_triangle_inequality(G: nx.Graph) -> bool:
    cost_matrix = nx.attr_matrix(G, edge_attr='weight')[0]
    for i in range(G.number_of_nodes()):
        for j in range(G.number_of_nodes()):
            for k in range(G.number_of_nodes()):
                if i == j or i == k or j == k:
                    continue
                if cost_matrix[i, j] + cost_matrix[j, k] < cost_matrix[i, k]:
                    print(f"Triangle inequality violated: {i} -> {j} -> {k} -> {i}, " +
                          f"{cost_matrix[i, j]} + {cost_matrix[j, k]} < {cost_matrix[i, k]}")
                    return False
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a dataset.')
    parser.add_argument('dir', type=pathlib.Path)
    parser.add_argument('--n_data', type=int, default=1000)
    args = parser.parse_args()
    cost = np.zeros((args.n_data, 2))
    is_matrix = np.zeros((args.n_data, 1))
    instances = list(args.dir.glob('*.pkl'))
    random.shuffle(instances)
    knn = []
    knn_regressive = []
    for instance_path in tqdm(instances[:args.n_data]):
        G = nx.read_gpickle(instance_path)
        knn.append(check_knn_in_optimal(G))
        knn_regressive.append(check_knn_in_optimal_regressive(G))

    knn = np.stack(knn)
    knn_regressive = np.stack(knn_regressive)

    print(knn.shape)
    print(knn_regressive.shape)

    np.savez('knn_statistics', knn=knn, knn_regressive=knn_regressive)
    # npz = np.load("knn_statistics.npz")
    # knn = npz["knn"]
    # knn_regressive = npz["knn_regressive"]
    
    print('-'*60)
    print(f"Overall statistic:")
    print(f"{'K mean:':<18} {knn.flatten().mean()}, std: {knn.flatten().std()}")
    print(f"{'Regressive K mean:':<18} {knn_regressive.flatten().mean()}, std: {knn_regressive.flatten().std()}")
    per_graph_knn = knn.max(2).max(1)
    per_graph_knn_regressive = knn_regressive.max(2).max(1)
    print('-'*60)
    print(f"Per-graph statistic:")
    print(f"{'K mean:':<18} {per_graph_knn.flatten().mean()}, std: {per_graph_knn.flatten().std()}")
    print(f"{'Regressive K mean:':<18} {per_graph_knn_regressive.flatten().mean()}, std: {per_graph_knn_regressive.flatten().std()}")
    print(f"K max: {per_graph_knn.max()}, Regressive K max: {per_graph_knn_regressive.max()}")