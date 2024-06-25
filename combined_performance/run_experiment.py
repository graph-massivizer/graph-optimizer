import os
import glob

from scipy.io import mminfo, mmread, mmwrite
import numpy as np


R = 10
IN_GRAPHS = glob.glob('../data/random/random_*.mtx')
print(f"Running experiment with {len(IN_GRAPHS)} input graphs and {R} runs.")

os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Prepare the sources vector.
for i, filename in enumerate(IN_GRAPHS):
    rows, cols, entries, _, _, _ = mminfo(filename)
    assert rows == cols
    mmwrite(f'data/{i}_in_sources.mtx', [list(range(rows))])

# BETWEENNESS CENTRALITY
graph = IN_GRAPHS
sources = [f'data/{i}_in_sources.mtx' for i in range(len(IN_GRAPHS))]
centrality = [f'data/{i}_out_centrality.mtx' for i in range(len(IN_GRAPHS))]
os.system(f"python ../autobench/run_bench.py ../bgo/bc/bc_lagr --runs {R} --data G={','.join(graph)} sources={','.join(sources)} centrality={','.join(centrality)} --output results/bc.csv")

# FIND MAXIMUM
os.system(f"python ../autobench/run_bench.py ../bgo/find_max/find_max_gb --runs {R} --data values={','.join(centrality)} --output results/find_max.csv")

# BREADTH FIRST SEARCH
source = [str(np.argmax(mmread(cmtx))) for cmtx in centrality]
level = [f'data/{i}_out_level.mtx' for i in range(len(IN_GRAPHS))]
parent = [f'data/{i}_out_parent.mtx' for i in range(len(IN_GRAPHS))]
os.system(f"python ../autobench/run_bench.py ../bgo/bfs/bfs_lagr --runs {R} --data G={','.join(graph)} source={','.join(source)} level={','.join(level)} parent={','.join(parent)} --output results/bfs.csv")

# COMBINED
os.system(f"python ../autobench/run_bench.py code/simple --runs {R} --data G={','.join(graph)} sources={','.join(sources)} level={','.join(level)} parent={','.join(parent)} --output results/combined.csv")
