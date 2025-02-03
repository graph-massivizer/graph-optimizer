#ifndef CC_GAP_HPP
#define CC_GAP_HPP

#include "datastructures.hpp"
#include "benchmark.h"

typedef CSRGraph<NodeID> CCGraph;

int cc_gap(const CCGraph &G, CArray<int> *components);

#endif