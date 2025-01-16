import random

import networkx as nx
import scipy as sp
import scipy.io


def random_gnm_graph(n, m):
    G = nx.gnm_random_graph(n, m)
    return G


def random_small_world_graph(n, k, p):
    G = nx.watts_strogatz_graph(n, k, p)
    return G


def write_graph_to_file(G, filename):
    M = nx.convert_matrix.to_scipy_sparse_matrix(G)
    with open(filename, 'wb') as file:
        sp.io.mmwrite(file, M)


if __name__ == "__main__":
    node_counts = [100, 250, 500, 1000, 2500, 5000, 7500, 10000, 25000]#, 50000, 100000, 250000, 500000, 1000000]
    for n in node_counts:
        for i in range(100):
            m = random.randint(16, min(n*n, 2**16))
            p = i/100
            k = random.randint(4, 10) if n < 1000 else random.randint(10, 20)

            G = random_gnm_graph(n, m)
            write_graph_to_file(G, f'topology/gnm/random_{n}_{m}.mtx')

            G = random_small_world_graph(n, k, p)
            write_graph_to_file(G, f'topology/sw/random_{n}_{k}_{p}.mtx')
