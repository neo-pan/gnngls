import concorde.tsp as concorde
import os
import lkh
import tempfile
import networkx as nx
import numpy as np
import tsplib95
from matplotlib import colors

from .utils import OutputGrabber

def tour_to_edge_attribute(G, tour):
    in_tour = {}
    tour_edges = list(zip(tour[:-1], tour[1:]))
    for e in G.edges:
        in_tour[e] = e in tour_edges
    return in_tour


def tour_cost(G, tour, weight='weight'):
    c = 0
    for e in zip(tour[:-1], tour[1:]):
        c += G.edges[e][weight]
    return c


def is_equivalent_tour(tour_a, tour_b):
    if tour_a == tour_b[::-1]:
        return True
    if tour_a == tour_b:
        return True
    return False


def is_valid_tour(G, tour):
    if tour[0] != 0:
        return False
    if tour[-1] != 0:
        return False
    for n in G.nodes:
        c = tour.count(n)
        if n == 0:
            if c != 2:
                return False
        elif c != 1:
            return False
    return True


def optimal_tour(G, scale=1e3, verbose=False, **kwargs):
    problem = tsplib95.models.StandardProblem()
    problem.name = 'TSP'
    problem.type = 'TSP'
    problem.dimension = 2 * G.number_of_nodes()
    problem.edge_weight_type = 'EXPLICIT'
    problem.edge_weight_format = 'FULL_MATRIX'

    # transform ATSP to TSP
    edge_weights, _ = nx.attr_matrix(G, edge_attr='weight')
    edge_weights = np.array(edge_weights)
    # set maximum and minimum value, idea borrowed from R package TSP
    EDGE_WEIGHTS_RANGE = edge_weights.max() - edge_weights.min()
    MAX_VALUE = edge_weights.max() + EDGE_WEIGHTS_RANGE * 2
    MIN_VALUE = edge_weights.min() - EDGE_WEIGHTS_RANGE * 2
    assert MAX_VALUE > edge_weights.max() and MIN_VALUE <= edge_weights.min()
    np.fill_diagonal(edge_weights, MIN_VALUE)
    infeasible = np.ones_like(edge_weights) * MAX_VALUE
    tsp_edge_weights = np.vstack((
        np.hstack((infeasible, edge_weights.T)),
        np.hstack((edge_weights, infeasible))
    ))
    problem_edge_weights = (tsp_edge_weights * scale).astype(np.int32)
    problem_edge_weights = problem_edge_weights - problem_edge_weights.min()
    problem.edge_weights = problem_edge_weights.tolist()

    with tempfile.TemporaryDirectory(dir="/mnt/data4/xhpan/tmp") as tmpdir:
        with open(os.path.join(tmpdir, "tmp.tsp"), "w") as fp:
            problem.write(fp)
        solver = concorde.TSPSolver.from_tspfile(os.path.join(tmpdir, "tmp.tsp"))
        solution = solver.solve(verbose, **kwargs)
        assert solution.success
        # filter the transformed tour
        tour = [i for i in solution.tour.tolist() if i < G.number_of_nodes()] + [0]

    # select the tour with smaller cost
    rev_tour = list(reversed(tour))
    tour_len = tour_cost(G, tour)
    rev_tour_len = tour_cost(G, rev_tour)
    if tour_len < rev_tour_len:
        result = tour
    else:
        result = rev_tour

    return result

def sub_optimal_tour(G, scale=1e3, lkh_path='/home/xhpan/Tools/LKH3/LKH-3.0.8/LKH', **kwargs):
    problem = tsplib95.models.StandardProblem()
    problem.name = 'TSP'
    problem.type = 'TSP'
    problem.dimension = len(G.nodes)
    problem.edge_weight_type = 'EXPLICIT'
    problem.edge_weight_format = 'FULL_MATRIX'
    edge_weights, _ = nx.attr_matrix(G, edge_attr='weight')
    problem.edge_weights = (edge_weights * scale).tolist()

    solution = lkh.solve(lkh_path, problem=problem, **kwargs)
    tour = [n - 1 for n in solution[0]] + [0]
    return tour


def optimal_cost(G, weight='weight'):
    c = 0
    for e in G.edges:
        if G.edges[e]['in_solution']:
            c += G.edges[e][weight]
    return c


def fixed_edge_tour(G, e, scale=1e3, lkh_path='/home/xhpan/Tools/LKH3/LKH-3.0.8/LKH', **kwargs):
    problem = lkh.LKHProblem()
    problem.name = 'ATSP'
    problem.type = 'ATSP'
    problem.dimension = len(G.nodes)
    problem.edge_weight_type = 'EXPLICIT'
    problem.edge_weight_format = 'FULL_MATRIX'
    edge_weights, _ = nx.attr_matrix(G, edge_attr='weight')
    problem.edge_weights = (edge_weights * scale).tolist()
    problem.fixed_edges = [[n + 1 for n in e]]

    solution = lkh.solve(lkh_path, problem=problem, **kwargs)
    tour = [n - 1 for n in solution[0]] + [0]
    return tour


def plot_edge_attribute(G, attr, ax, **kwargs):
    cmap_colors = np.zeros((100, 4))
    cmap_colors[:, 0] = 1
    cmap_colors[:, 3] = np.linspace(0, 1, 100)
    cmap = colors.ListedColormap(cmap_colors)

    pos = nx.get_node_attributes(G, 'pos')

    nx.draw(G, pos, edge_color=attr.values(), edge_cmap=cmap, ax=ax, **kwargs)
