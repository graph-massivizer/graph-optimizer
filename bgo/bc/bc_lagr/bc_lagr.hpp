#ifndef BC_LAGR_HPP
#define BC_LAGR_HPP

#include "GraphBLAS.h"
#include "LAGraph.h"

#include "datastructures.hpp"

int betweenness_centrality_LAGr(LAGraph_Graph G, CArray<GrB_Index> sources, GrB_Vector *centrality);

#endif
