#ifndef PR_GAP_HPP
#define PR_GAP_HPP

#include "datastructures.hpp"
#include "benchmark.h"

typedef CSRGraph<NodeID> PRGraph;

int PR_gap(const PRGraph &G, CArray<int> *PR);

#endif