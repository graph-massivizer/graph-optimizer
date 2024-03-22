#ifndef UTILS_H
#define UTILS_H

#include "GraphBLAS.h"
#include "LAGraph.h"
#include <vector>
#include <iostream>

template <typename T>
void pretty_print_array(T *arr, int n, std::string name) {
    std::cout << name.c_str() << ": [";
    for (int i = 0; i < n - 1; i++) {
        std::cout << std::to_string(arr[i]) << ", ";
    }
    std::cout << std::to_string(arr[n - 1]) << "]" << std::endl;
}

/*
 * Convert a matrix M to a vector array G.
 */
template <typename T>
void GB_matrix_to_vector_array(GrB_Matrix M, std::vector<T> *G) {
    GrB_Index num_nodes;
    GrB_Matrix_nrows(&num_nodes, M);

    G->resize(num_nodes);
    for (uint i = 0; i < num_nodes; i++) {
        G[i].resize(num_nodes);
    }

    GrB_Index nvals;
    GrB_Matrix_nvals(&nvals, M);

    GrB_Index row_indices[nvals];
    GrB_Index col_indices[nvals];
    float values[nvals];

    GrB_Matrix_extractTuples_FP32(row_indices, col_indices, values, &nvals, M);

    for (uint i = 0; i < nvals; i++) {
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
template <typename T>
void read_graph_vector_array(std::vector<T> *G, char *fileName) {
    GrB_Matrix A;
    read_graph_GB(&A, fileName);

    GB_matrix_to_vector_array<T>(A, G);
}

template <typename T>
void pretty_print_vector(GrB_Vector v, std::string name) {
    GrB_Index n;
    GrB_Vector_size(&n, v);
    GrB_Index indices[n];
    int values[n];
    GrB_Vector_extractTuples_INT32(indices, values, &n, v);
    std::cout << name.c_str() << ": [";
    for (uint i = 0; i < n - 1; i++) {
        std::cout << std::to_string(values[i]) << ", ";
    }
    std::cout << std::to_string(values[n-1]) << "]" << std::endl;
}

#endif
