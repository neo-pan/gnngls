#!/usr/bin/env python
# coding: utf-8

import argparse
import itertools
import multiprocessing as mp
import pathlib
import uuid

import networkx as nx
import numpy as np

import gnngls
from gnngls import datasets
from utils import stdout_redirected


def prepare_instance(G):
    datasets.set_features(G)
    datasets.set_labels(G)
    return G


def get_solved_instances(n_nodes, n_instances):
    for _ in range(n_instances):
        G = nx.DiGraph()

        coords = np.random.random((n_nodes, 2))
        for n, p in enumerate(coords):
            G.add_node(n, pos=0.0)

        distance = np.random.random((n_nodes, n_nodes))
        for i, j in itertools.permutations(G.nodes, 2):
            G.add_edge(i, j, weight=distance[i, j])

        # opt_solution = gnngls.sub_optimal_tour(G, scale=1e6, max_trials=100, runs=10)
        opt_solution = gnngls.optimal_tour(G, scale=1e6)
        in_solution = gnngls.tour_to_edge_attribute(G, opt_solution)
        nx.set_edge_attributes(G, in_solution, 'in_solution')

        yield G


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a dataset.')
    parser.add_argument('n_samples', type=int)
    parser.add_argument('n_nodes', type=int)
    parser.add_argument('dir', type=pathlib.Path)
    args = parser.parse_args()

    if args.dir.exists():
        print(f'Output directory {args.dir} exists.')
        # raise Exception(f'Output directory {args.dir} exists.')
    else:
        args.dir.mkdir()
    pool = mp.Pool(processes=32)
    instance_gen = get_solved_instances(args.n_nodes, args.n_samples)
    for G in pool.imap_unordered(prepare_instance, instance_gen):
        nx.write_gpickle(G, args.dir / f'{uuid.uuid4().hex}.pkl')
    pool.close()
    pool.join()
