#include "GraphBLAS.h"
#include "LAGraph.h"

#include "datastructures.hpp"

#include "bc_lagr.hpp"

int bc_lagr(LAGraph_Graph G, CArray<GrB_Index> sources, GrB_Vector *centrality) {
    char msg[LAGRAPH_MSG_LEN];

    if (G->AT == NULL)
        LAGraph_Cached_AT(G, msg);
    return LAGr_Betweenness(centrality, G, sources.data, sources.size, msg);
}
