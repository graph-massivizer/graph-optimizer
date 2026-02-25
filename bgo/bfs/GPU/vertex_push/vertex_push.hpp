#ifndef BFS_GPU_VERTEX_PUSH
#define BFS_GPU_VERTEX_PUSH

#include "../bfs.hpp"
#include "datastructures.hpp"

int bfs_vertex_push(CSR &g, int source, CArray<int32_t> *levels, CArray<int> *parent);

#endif