#ifndef BFS_GAP_HPP
#define BFS_GAP_HPP

#include "datastructures.hpp"
#include "benchmark.h"

typedef CSRGraph<NodeID> BFSGraph;

int bfs_gap(const BFSGraph &G, int source, CArray<int> *level, CArray<int> *parent);

#endif