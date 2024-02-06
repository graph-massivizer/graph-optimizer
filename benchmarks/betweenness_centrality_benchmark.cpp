#include "GraphBLAS.h"
#include "suitesparse/LAGraph.h"
#include "../include/utils.hpp"
#include "../bgo/betweenness_centrality.hpp"
#include "../bgo/find_max.hpp"
#include <stdio.h>
#include <ctime>

char msg[LAGRAPH_MSG_LEN];
clock_t start_time;
double time_elapsed;

int main(int argc, char **argv) {
    srand(time(0));
    if (argc < 2)
    {
        fprintf(stderr, "Error\nUsage: %s <matrix_market_file.mtx>\n", argv[0]);
        return 1;
    }

    LAGraph_Init(msg);

    GrB_Matrix A;

    read_graph_GB(&A, argv[1]);

    GrB_Index num_nodes;
    GrB_Matrix_nrows(&num_nodes, A);
    vector<int> G_naive[num_nodes];
    GB_matrix_to_vector_array(A, G_naive);

    /*
     * Initialization of source nodes
     */
    int n_sources = 1000;
    GrB_Index sources_GrB[n_sources];
    int sources[n_sources];
    for (int i = 0; i < n_sources; i++) {
        int random = rand();
        int temp = random % num_nodes;
        temp = i;
        sources_GrB[i] = temp;
        sources[i] = temp;
    }

    /***************************************************************************
     * METHOD 1: GraphBLAS
     **************************************************************************/
    start_time = clock();
    GrB_Vector bc;
    BC_GB_update(&bc, A, sources_GrB, n_sources);
    time_elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    printf("Time elapsed GraphBLAS: %f\n", time_elapsed);
    find_max(bc);


    /***************************************************************************
     * METHOD 2: LAGraph (also using GraphBLAS)
     **************************************************************************/
    LAGraph_Graph G;
    LAGraph_New(&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg);

    start_time = clock();
    LAGr_Betweenness(&bc, G, sources_GrB, n_sources, msg);
    time_elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    printf("Time elapsed LAGraph: %f\n", time_elapsed);
    find_max(bc);


    /***************************************************************************
     * METHOD 3: Naive sequential
     **************************************************************************/
    float bc_naive[num_nodes];
    GrB_Index nvals;
    GrB_Matrix_nvals(&nvals, A);
    start_time = clock();
    BC_naive(bc_naive, G_naive, sources, n_sources, num_nodes);
    time_elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    printf("Time elapsed naive: %f\n", time_elapsed);
    find_max(bc_naive, num_nodes);


    /***************************************************************************
     * METHOD 4: Brandes' algorithm
     **************************************************************************/
    float bc_brandes[num_nodes];
    start_time = clock();
    BC_brandes(bc_brandes, G_naive, sources, n_sources, num_nodes);
    time_elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    printf("Time elapsed Brandes: %f\n", time_elapsed);
    find_max(bc_brandes, num_nodes);


    // Cleanup
    GrB_Matrix_free(&A);
    LAGraph_Finalize(msg);

    return 0;
}