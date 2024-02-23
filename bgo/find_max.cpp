#include <GraphBLAS.h>
#include "find_max.hpp"

int find_max(GrB_Vector v) {
    GrB_Index num_nodes;
    GrB_Vector_size(&num_nodes, v);

    float max = -1.;
    int max_index = -1;
    for (uint i = 0; i < num_nodes; i++) {
        float val;
        GrB_Vector_extractElement_FP32(&val, v, i);
        if (val > max) {
            max = val;
            max_index = i;
        }
    }

    return max_index;
}

int find_max(float *v, int n) {
    float max = -1.;
    int max_index = -1;
    for (int i = 0; i < n; i++) {
        if (v[i] > max) {
            max = v[i];
            max_index = i;
        }
    }

    return max_index;
}