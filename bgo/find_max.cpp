#include <GraphBLAS.h>
#include "find_max.hpp"

int find_max(GrB_Vector v, int n) {
    // GrB_Index num_nodes;
    // GrB_Vector_size(&num_nodes, v);

    float max = -1.;
    int max_index = -1;
    float val;
    for (int i = 0; i < n; i++) {
        GrB_Vector_extractElement_FP32(&val, v, i);
        if (val > max) {
            max = val;
            max_index = i;
        }
    }

    return max_index;
}

int find_max(float *v, int n) {
    register float max = -1.;
    register int max_index = -1;
    register float val;
    for (register int i = 0; i < n; i++) {
        val = v[i];
        if (val > max) {
            max = val;
            max_index = i;
        }
    }

    return max_index;
}
