import random

import networkx as nx
import scipy as sp
import scipy.io


def random_unique_graph(n, m):
    cache = random_unique_graph.cache = getattr(random_unique_graph, 'cache', set())
    G = nx.gnm_random_graph(n, m)
    signature = (G.number_of_nodes(), G.number_of_edges())
    if signature in cache:
        return random_unique_graph(n, m)
    cache.add(signature)
    return G


def write_graph_to_file(G, filename):
    M = nx.convert_matrix.to_scipy_sparse_matrix(G)
    with open(filename, 'wb') as file:
        sp.io.mmwrite(file, M)


for _ in range(100):
    n = random.randint(8, 2**16)
    m = random.randint(8, min(n*n, 2**16))

    G = random_unique_graph(n, m)
    write_graph_to_file(G, f'random_{n}_{m}.mtx')
