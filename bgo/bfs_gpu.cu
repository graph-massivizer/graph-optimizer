/*
 * GPU Breadth First Search implementation based on:
 *     https://github.com/kaletap/bfs-cuda-gpu/blob/master/src/gpu/simple/bfs_simple.cu
 *         by Przemysław Kaleta (https://github.com/kaletap)
 */


#include "GraphBLAS.h"

#include "bfs_gpu.cuh"

#define cudaDeviceScheduleBlockingSync 0x04 
#define BLOCK_SIZE 32


__global__
void initDeviceArray(int *array, int size, int alt_index) {
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_index < size) {
        if (thread_index == alt_index) {
            array[thread_index] = 0;
        } else {
            array[thread_index] = -1;
        }
    }
}

__global__
void stepBreadthFirstSearchGPU(
    int *level, int *parent,
    bool *G, int num_nodes,
    int *cQueue, int *nQueue, int *cQueueSize, int *nQueueSize, int cLevel
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < *cQueueSize) {
        int u = cQueue[tid];

        for (int v = 0; v < num_nodes; v++) {
            if (G[u * num_nodes + v] && (level[v] == -1)) {  // Reachable and unvisited
                parent[v] = u;
                level[v] = cLevel + 1;

                int i = atomicAdd(nQueueSize, 1);
                nQueue[i] = v;
            }
        }
    }
}

extern "C"
void breadthFirstSearchGPU(
    GrB_Vector *level, GrB_Vector *parent,
    const GrB_Matrix A, int num_nodes, int root
) {
    const int blocks = (num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int nQueueSize = 1;
    int cLevel = 0;

    bool *d_G;
    int *d_level;
    int *d_parent;
    int *d_cQueue;
    int *d_nQueue;
    int *d_cQueueSize;
    int *d_nQueueSize;

    /* Allocate memory on the device. */
    cudaMalloc((void **) &d_G, num_nodes * num_nodes * sizeof(bool));
    cudaMalloc((void **) &d_level, num_nodes * sizeof(int));
    cudaMalloc((void **) &d_parent, num_nodes * sizeof(int));
    cudaMalloc((void **) &d_cQueue, num_nodes * sizeof(int));
    cudaMalloc((void **) &d_nQueue, num_nodes * sizeof(int));
    cudaMalloc((void **) &d_cQueueSize, sizeof(int));
    cudaMalloc((void **) &d_nQueueSize, sizeof(int));

    {   /* Create a boolean matrix and copy to device. */
        bool h_G[num_nodes * num_nodes] = {0};

        GrB_Index nvals;
        GrB_Matrix_nvals(&nvals, A);

        GrB_Index row_indices[nvals];
        GrB_Index col_indices[nvals];
        float values[nvals];

        GrB_Matrix_extractTuples_FP32(row_indices, col_indices, values, &nvals, A);

        for (unsigned int i = 0; i < nvals; i++) {
            h_G[row_indices[i] * num_nodes + col_indices[i]] = (values[i] > 0);
        }
        
        cudaMemcpy(d_G, &h_G, num_nodes * num_nodes * sizeof(bool), cudaMemcpyHostToDevice);
    }

    /* Initialize the level and parent arrays. */
    initDeviceArray<<<blocks, BLOCK_SIZE>>>(d_level, num_nodes, root);
    initDeviceArray<<<blocks, BLOCK_SIZE>>>(d_parent, num_nodes, -1);

    /* Put the root node in the queue. */
    cudaMemcpy(d_nQueue, &root, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nQueueSize, &nQueueSize, sizeof(int), cudaMemcpyHostToDevice);  // This fails?!
    cudaDeviceSynchronize();

    while (nQueueSize > 0) {
        /* Swap current and next queue. */
        int *tmp = d_cQueue;
        d_cQueue = d_nQueue;
        d_nQueue = tmp;
        tmp = d_cQueueSize;
        d_cQueueSize = d_nQueueSize;
        d_nQueueSize = tmp;

        /* Make sure the next queue is empty! */
        cudaMemset(d_nQueueSize, 0, sizeof(int));
        cudaDeviceSynchronize();

        stepBreadthFirstSearchGPU<<<blocks, BLOCK_SIZE>>>(
            d_level, d_parent,
            d_G, num_nodes,
            d_cQueue, d_nQueue, d_cQueueSize, d_nQueueSize, cLevel
        );
        cudaDeviceSynchronize();

        cudaMemcpy(&nQueueSize, d_nQueueSize, sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        if (++cLevel > num_nodes) {
            printf("Reached maximum number of iterations!\n");
            break;
        }
    }

    int tmp_level[num_nodes];
    int tmp_parent[num_nodes];
    cudaMemcpy(&tmp_level, d_level, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&tmp_parent, d_parent, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    GrB_Vector_new(level, GrB_INT32, num_nodes);
    GrB_Vector_new(parent, GrB_INT32, num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        GrB_Vector_setElement_INT32(*level, tmp_level[i], i);
        GrB_Vector_setElement_INT32(*parent, tmp_parent[i], i);
    }

    cudaFree(d_G);
    cudaFree(d_level);
    cudaFree(d_parent);
    cudaFree(d_cQueue);
    cudaFree(d_nQueue);
    cudaFree(d_cQueueSize);
    cudaFree(d_nQueueSize);
}
