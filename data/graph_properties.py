import networkx as nx
import scipy as sp

def extract_graph_properties(graph_name):
    g = nx.from_scipy_sparse_array(sp.io.mmread(graph_name))

    n = len(g.nodes())
    m = len(g.edges())

    extracted_properties = {
        "name": graph_name,
        "n": n,
        "m": m,
        "directed": g.is_directed(),
        "weighted": nx.is_weighted(g),
        "diameter": nx.diameter(g),
        "clustering_coefficient": nx.average_clustering(g),
        "triangle_count": sum(nx.triangles(g).values()) // 3
    }
    
    return extracted_properties
