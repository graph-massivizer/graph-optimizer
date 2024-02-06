#ifndef UTILS_H
#define UTILS_H

#include "GraphBLAS.h"
#include "suitesparse/LAGraph.h"
#include <vector>
using namespace std;

/*
 * Convert a matrix M to a vector array G.
 */
void GB_matrix_to_vector_array(GrB_Matrix M, vector<int> *G) {
    GrB_Index num_nodes;
    GrB_Matrix_nrows(&num_nodes, M);

    G->resize(num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        G[i].resize(num_nodes);
    }

    GrB_Index nvals;
    GrB_Matrix_nvals(&nvals, M);

    GrB_Index row_indices[nvals];
    GrB_Index col_indices[nvals];
    int values[nvals];
    GrB_Matrix_extractTuples_INT32(row_indices, col_indices, values, &nvals, M);

    for (int i = 0; i < nvals; i++) {
        G[row_indices[i]][col_indices[i]] = values[i];
    }
}

/*
 * Read a graph into matrix M given a specified fileName.
 */
void read_graph_GB(GrB_Matrix *M, char *fileName) {
    FILE *fd = fopen(fileName, "r");
    char msg[LAGRAPH_MSG_LEN];

    if (GrB_SUCCESS != LAGraph_MMRead(M, fd, msg))
    {
        fprintf(stderr, "ERROR: Failed to load graph: %s\n", fileName);
        exit(-1);
    }

    if (fd != NULL)
        fclose(fd);
}

/*
 * Read a graph into vector array G given a specified fileName.
 */
void read_graph_vector_array(vector<int> *G, char *fileName) {
    GrB_Matrix A;
    read_graph_GB(&A, fileName);

    GB_matrix_to_vector_array(A, G);
}



#endif
