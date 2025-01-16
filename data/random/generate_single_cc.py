import random

import networkx as nx
import scipy as sp
import scipy.io


def random_connected_gnm_graph(n, m):
    G = nx.gnm_random_graph(n, m)
    largest_cc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return largest_cc


def random_connected_small_world_graph(n, k, p):
    G = nx.watts_strogatz_graph(n, k, p)
    largest_cc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return largest_cc


def write_graph_to_file(G, filename):
    M = nx.convert_matrix.to_scipy_sparse_matrix(G)
    with open(filename, 'wb') as file:
        sp.io.mmwrite(file, M)


if __name__ == "__main__":
    for i in range(1000):
        print("Iteration:", i)
        n = random.randint(16, 2**14)
        m = random.randint(16, min(n*n, 2**14))
        p = random.random()
        k = random.randint(2, 16)

        G = random_connected_gnm_graph(n, m)
        write_graph_to_file(G, f'single_cc/gnm/random_{n}_{m}.mtx')

        G = random_connected_small_world_graph(n, k, p)
        write_graph_to_file(G, f'single_cc/sw/random_{n}_{k}_{p}.mtx')
