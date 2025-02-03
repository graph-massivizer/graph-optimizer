import random

import networkx as nx
import scipy as sp
import scipy.io


def random_unique_gnm_graph(n, m):
    cache = random_unique_gnm_graph.cache = getattr(random_unique_gnm_graph, 'cache', set())
    G = nx.gnm_random_graph(n, m)
    signature = (G.number_of_nodes(), G.number_of_edges())
    if signature in cache:
        return random_unique_gnm_graph(n, m)
    cache.add(signature)
    return G


def random_gnp_graph(n, p):
    cache = random_gnp_graph.cache = getattr(random_gnp_graph, 'cache', set())
    G = nx.gnp_random_graph(n, p)
    signature = (G.number_of_nodes(), G.number_of_edges())
    if signature in cache:
        return random_gnp_graph(n, p)
    cache.add(signature)
    return G


def random_small_world_graph(n, k, p):
    cache = random_small_world_graph.cache = getattr(random_small_world_graph, 'cache', set())
    G = nx.watts_strogatz_graph(n, k, p)
    signature = (G.number_of_nodes(), G.number_of_edges())
    if signature in cache:
        return random_small_world_graph(n, k, p)
    cache.add(signature)
    return G


def write_graph_to_file(G, filename):
    M = nx.convert_matrix.to_scipy_sparse_matrix(G)
    with open(filename, 'wb') as file:
        sp.io.mmwrite(file, M)


if __name__ == "__main__":
    for i in range(1000):
        n = random.randint(2**16, 2**21)
        m = random.randint(n, min(n*n, 2**25))
        print(n, m)
        G = random_unique_gnm_graph(n, m)
        write_graph_to_file(G, f'generated/gnm/random_{n}_{m}.mtx')
