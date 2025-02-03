#ifndef TC_GAP_HPP
#define TC_GAP_HPP

#include "datastructures.hpp"
#include "benchmark.h"

typedef CSRGraph<NodeID> TCGraph;

int TC_gap(const TCGraph &g, int *triangles);

#endif