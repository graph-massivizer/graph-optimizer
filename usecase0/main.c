#include <stdio.h>
#include <omp.h>
#include "GraphBLAS.h"
#include "LAGraph.h"
#include "betweenness_centrality.h"

void read_graph(GrB_Matrix *M, char *fileName) {
    FILE *fd = fopen(fileName, "r");

    if (GrB_SUCCESS != LAGraph_mmread(M, fd))
    {
        fprintf(stderr, "ERROR: Failed to load graph: %s\n", fileName);
        exit(-1);
    }

    if (fd != NULL)
        fclose(fd);
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Error\nUsage: %s <matrix_market_file.mtx>\n", argv[0]);
        return 1;
    }

    LAGraph_init();

    GrB_Matrix A = NULL;

    double tic[2], t;
    LAGraph_tic(tic);
    read_graph(&A, argv[1]);
    t = LAGraph_toc(tic);
    printf("Reading file took %g seconds\n", t);

    GrB_Index num_nodes;
    GrB_Matrix_nrows(&num_nodes, A);

    /*
     * BGO 1: filtering
     */

    //TODO

    /*
     * BGO 2: Betweenness centrality
     */

    GrB_Vector bc;
    GrB_Vector local_total_bc;
    GrB_Vector total_bc;

    GrB_Vector_new(&local_total_bc, GrB_FP32, num_nodes);
    GrB_Vector_new(&total_bc, GrB_FP32, num_nodes);

    LAGraph_tic(tic);

    for (GrB_Index src = 0; src < num_nodes; ++src)
    {
        BC(&bc, A, src);
        GrB_eWiseAdd(total_bc, GrB_NULL, GrB_NULL, GrB_PLUS_FP32, bc, total_bc, GrB_NULL);
    }
    

    t = LAGraph_toc(tic);

    printf("Betweenness Centrality took %g seconds\n", t);
    float max_bc = -1.;
    int max_bc_index = -1;
    for (int i = 0; i < num_nodes; i++) {
        float val;
        GrB_Vector_extractElement_FP32(&val, total_bc, i);
        if (val > max_bc) {
            max_bc = val;
            max_bc_index = i;
        }
    }
    printf("The node with the highest betweenness centrality is: %d, with a value of %f\n", max_bc_index + 1, max_bc);

    //TODO

    /*
     * BGO 3: Single-Source-Shortest-Path
     */

    //TODO

    // Cleanup
    GrB_free(&total_bc);
    GrB_free(&bc);
    GrB_free(&A);

    LAGraph_finalize();
    return GrB_SUCCESS;
}