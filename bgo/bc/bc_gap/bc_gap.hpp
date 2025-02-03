#ifndef BC_GAP_HPP
#define BC_GAP_HPP

#include "datastructures.hpp"
#include "benchmark.h"

typedef CSRGraph<NodeID> BCGraph;

int bc_gap(const BCGraph &G, CArray<int> sources, CArray<int> *centrality);

#endif