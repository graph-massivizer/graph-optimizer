

#include "GraphBLAS.h"
#include "LAGraph.h"

#include "datastructures.hpp"

#include "bc_lagr.hpp"

char msg[LAGRAPH_MSG_LEN];

int betweenness_centrality_LAGr(LAGraph_Graph G, CArray<GrB_Index> sources, GrB_Vector *centrality) {
    if (G->AT == NULL)
        LAGraph_Cached_AT(G, msg);
    return LAGr_Betweenness(centrality, G, sources.data, sources.size, msg);
}
