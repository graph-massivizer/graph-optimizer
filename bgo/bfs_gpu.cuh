#include "GraphBLAS.h"
#include "suitesparse/LAGraph.h"


extern "C"
void breadthFirstSearchGPU(
    GrB_Vector *level, GrB_Vector *parent,
    const GrB_Matrix A, int num_nodes, int root
);
